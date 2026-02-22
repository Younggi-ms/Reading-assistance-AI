# engine.py
import json
import random
from dataclasses import dataclass
from typing import List, Optional

import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# --------------------------------------------------
# 데이터 구조
# --------------------------------------------------

@dataclass
class Quiz:
    question: str
    options: List[str]
    correct_answer: int
    document_context: str
    difficulty: float = 0.5


@dataclass
class User:
    name: str
    points: int = 0
    total_questions: int = 0
    correct_answers: int = 0


# --------------------------------------------------
# PDF 처리기
# --------------------------------------------------

class DocumentProcessor:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def extract_text_from_pdf(self, path: str) -> List[str]:
        """
        파일 경로 기반 PDF 텍스트 추출
        Gradio File(type="filepath")랑 궁합 맞음.
        """
        chunks: List[str] = []
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [p.extract_text() or "" for p in reader.pages]
                full_text = "\n".join(pages)
                chunks = self._split_into_chunks(full_text)
        except Exception as e:
            print("PDF ERROR:", e)
        return chunks

    def _split_into_chunks(self, text: str) -> List[str]:
        chunks: List[str] = []
        buf = ""

        for s in text.split(". "):
            if len(buf) + len(s) < self.max_chunk_size:
                buf += s + ". "
            else:
                chunks.append(buf.strip())
                buf = s + ". "

        if buf:
            chunks.append(buf.strip())

        return chunks


# --------------------------------------------------
# 유틸: 중국어 포함 여부 체크
# --------------------------------------------------

def contains_chinese(s: str) -> bool:
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


# --------------------------------------------------
# LLM 문제 생성기 (Qwen2.5-7B-Instruct)
# --------------------------------------------------

class SPICEQuizGenerator:
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        print("Loading model:", model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )
        # pad 토큰 정리
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",  # GPU(L4) 활용
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        self.model.eval()
        print("Model Ready.")

    def build_prompt(self, doc: str, diff: float) -> str:
        """
        Qwen2.5-7B-Instruct에게 줄 프롬프트.
        - JSON 형식 그대로
        - 한국어/영어만 허용, 중국어 금지 강조
        """
        diff_desc = "쉬운" if diff < 0.3 else "중간" if diff < 0.7 else "어려운"
        return f"""
당신은 문제 제작 전문가입니다.
다음 텍스트를 기반으로 {diff_desc} 난이도의 4지선다 객관식 문제를 JSON으로 만들어주세요.

중요:
- 한국어 또는 영어만 사용하세요.
- 중국어(간체/번체 포함), 한자는 절대로 사용하지 마십시오.
- 반드시 아래 JSON 형식만 출력하세요. 설명, 마크다운, 코드블록, 해설 금지.

텍스트:
{doc[:1500]}

JSON 형식:
{{
  "question": "",
  "options": ["", "", "", ""],
  "correct_answer": 0
}}
"""

    def _parse_json_from_output(self, decoded: str) -> dict:
        """
        모델 출력에서 JSON만 뽑아서 파싱.
        - 여러 개의 { ... } 가 있어도 마지막으로 유효한 것 선택
        """
        import re

        candidates = re.findall(r"\{[\s\S]*?\}", decoded)
        if not candidates:
            raise ValueError("No JSON candidate found in model output")

        last_error: Optional[Exception] = None

        for cand in reversed(candidates):
            try:
                data = json.loads(cand)

                if (
                    isinstance(data, dict)
                    and "question" in data
                    and "options" in data
                    and "correct_answer" in data
                    and isinstance(data["options"], list)
                    and len(data["options"]) == 4
                ):
                    return data
            except Exception as e:
                last_error = e
                continue

        if last_error is not None:
            raise last_error
        else:
            raise ValueError("Failed to parse any JSON candidate")

    def _generate_once(self, chunk: str, diff: float) -> Optional[Quiz]:
        prompt = self.build_prompt(chunk, diff)
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **tokens,
                max_new_tokens=384,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # 중국어 포함되면 바로 버리기
        if contains_chinese(decoded):
            print("CHINESE DETECTED in model output, will retry.")
            print("MODEL OUTPUT (TRUNCATED):\n", decoded[:400])
            return None

        try:
            data = self._parse_json_from_output(decoded)
            return Quiz(
                question=str(data["question"]),
                options=[str(o) for o in data["options"]],
                correct_answer=int(data["correct_answer"]),
                document_context=chunk[:300],
                difficulty=diff,
            )
        except Exception as e:
            print("JSON ERROR:", e)
            print("MODEL OUTPUT:\n", decoded)
            return None

    def generate(self, chunk: str, diff: float) -> Optional[Quiz]:
        """
        중국어 금지 + JSON 파싱 실패를 고려한 재시도 로직.
        - 최대 3번까지 재시도
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            print(f"[GEN] Attempt {attempt}/{max_retries}")
            quiz = self._generate_once(chunk, diff)
            if quiz is not None:
                return quiz
        # 전부 실패
        return None


# --------------------------------------------------
# 메인 앱
# --------------------------------------------------

class ReadingAssistantApp:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.generator = SPICEQuizGenerator()
        self.user = User("user")
        self.chunks: List[str] = []
        self.cache: List[Quiz] = []
        self.difficulty: float = 0.5  # 일단 고정

    def load_pdf(self, path: str) -> str:
        """
        path 기반으로 PDF 읽고 청크 생성.
        Gradio에서 문자열 경로만 넘어오므로 여기서 처리.
        """
        self.chunks = self.processor.extract_text_from_pdf(path)
        if not self.chunks:
            return "❌ PDF 로드 실패: 텍스트를 추출할 수 없습니다."

        return f"✅ PDF 로드 완료! (총 {len(self.chunks)}개 청크)"

    def generate_quiz(self) -> str:
        if not self.chunks:
            return "❌ 문제 생성 실패: 먼저 PDF를 업로드해주세요."

        chunk = random.choice(self.chunks)
        quiz = self.generator.generate(chunk, self.difficulty)

        if quiz is None:
            return "❌ 문제 생성 실패: 모델 응답을 파싱하지 못했습니다. 다시 시도해주세요."

        self.cache.append(quiz)

        # 사람이 읽기 좋게 문자열 출력
        text = f"📘 문제:\n{quiz.question}\n\n"
        for i, op in enumerate(quiz.options):
            text += f"{i}. {op}\n"
        return text

    def check_answer(self, idx_str: str) -> str:
        if not self.cache:
            return "❌ 먼저 문제를 생성하세요."

        try:
            idx = int(idx_str)
        except Exception:
            return "❌ 정답은 0~3 사이의 숫자로 입력하세요."

        quiz = self.cache[-1]
        correct = (idx == quiz.correct_answer)

        if correct:
            self.user.points += 10
            self.user.correct_answers += 1

        self.user.total_questions += 1

        if correct:
            return f"✅ 정답입니다! 현재 점수: {self.user.points}점"
        else:
            return f"❌ 오답입니다. 정답은 {quiz.correct_answer}번입니다. 현재 점수: {self.user.points}점"
