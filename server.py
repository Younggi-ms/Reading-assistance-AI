# server.py
import uvicorn
from fastapi import FastAPI
import gradio as gr

from engine import ReadingAssistantApp


# --------------------------------------------------
# FastAPI + Gradio 통합 서버
# --------------------------------------------------

app = FastAPI()
engine = ReadingAssistantApp()


# ---------------- Gradio용 래퍼 함수 ----------------

def pdf_ui(pdf_path: str):
    """
    Gradio File 컴포넌트에서 type="filepath"로 받으면
    pdf_path는 그냥 문자열 경로임.
    거기에 맞춰서 engine.load_pdf를 호출.
    """
    if pdf_path is None or pdf_path == "":
        return "❌ 파일을 먼저 업로드해주세요."

    msg = engine.load_pdf(pdf_path)
    return msg


def quiz_ui():
    return engine.generate_quiz()


def answer_ui(idx: str):
    return engine.check_answer(idx)


# ---------------- Gradio Blocks UI 구성 ----------------

def build_gradio_app():
    with gr.Blocks(title="독서 보조 AI - Qwen2.5-7B") as demo:
        gr.Markdown("# 📚 독서 보조 AI - Qwen2.5-7B-Instruct 기반 SPICE 문제 생성")

        # 1) PDF 업로드 탭
        with gr.Tab("📄 PDF 업로드"):
            pdf_input = gr.File(
                label="PDF 업로드",
                type="filepath",  # 문자열 경로로 받기
            )
            pdf_btn = gr.Button("PDF 분석 시작")
            pdf_out = gr.Textbox(label="결과", lines=3)
            pdf_btn.click(pdf_ui, inputs=pdf_input, outputs=pdf_out)

        # 2) 문제 풀기 탭 (문제 생성 + 정답 제출 한 페이지)
        with gr.Tab("📝 문제 풀기"):
            gr.Markdown("### 1단계: 문제 생성 → 2단계: 정답 번호 입력")

            # 문제 생성 영역
            q_btn = gr.Button("문제 생성")
            q_out = gr.Textbox(lines=10, label="생성된 문제")

            # 정답 입력 + 제출 영역 (같은 탭 안에 배치)
            ans_input = gr.Textbox(label="정답 번호 (0~3)", lines=1)
            ans_btn = gr.Button("정답 제출")
            ans_out = gr.Textbox(label="채점 결과", lines=2)

            # 버튼 동작 연결
            q_btn.click(quiz_ui, outputs=q_out)
            ans_btn.click(answer_ui, inputs=ans_input, outputs=ans_out)

    return demo


# Gradio 앱 생성 & FastAPI에 mount
gr_app = build_gradio_app()
app = gr.mount_gradio_app(app, gr_app, path="/")


# --------------------------------------------------
# 실행
# --------------------------------------------------

if __name__ == "__main__":
    # GCP VM에서:
    #   python3 server.py
    # 로 실행하면
    #   http://서버IP:7860/
    # 로 접속 가능
    uvicorn.run(app, host="0.0.0.0", port=7860)
