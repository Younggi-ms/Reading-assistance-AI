# Reading-assistance-AI
## 프로젝트 개요
- 본 프로젝트는 독서가 생활화 되어있지 않은 학생/성인분들을 독서력 향상 및 독서에 대한 거부감 감소를 목표로 하였습니다.

- 독서를 평소에 접하지 않은 학생/성인분들은 책을 뭐부터 읽어야 하는지 몰라, 어떤 책이든 일단 읽어보기 시작하다 어려움에 봉착하는 경우가 많습니다. 이를 극복하기 위해 어떠한 책이든 쉽게 풀어줄 수 있는 독서 보조 AI를 목표로 개발을 진행하였습니다.

## 개발 목표
- 책 내용 요약 기능
- 책 내용 리마인드를 위한 문제 출제 및 정답 체크 기능
- GPU 서버와 연동 후 홈페이지 구현
- 많은 수 유저들의 접속시 발생 트래픽 체크 

## 시스템 구성
### 소프트 웨어
- Vscode Debugging
- 기초적 LLM 모델[Qwen/Qwen2.5-7B-Instruct]
### 하드웨어 
- 구글 클라우드 [GPU 서버]

## 본인 담당 역할 및 구현 역량

### 담당 범위
GPU 서버 연동 및 연동 이후 LLM모델 데이터 이상 여부 테스트하였습니다.
### 주요 구현 내용
- LLM 모델 GPU서버 연동: LLM 모델의 오픈소스 내용을 engine.py, server.py로 코드를 분할한 후, 서버에서 server.py를 항시 구동하면서 필요시 engine.py를 호출하는 식으로 연동하였습니다.
- 모델 출력 데이터 이상 여부 체크: 모델의 출력 데이터가 한국어/중국어/영어 중 한국어로 출력이 되도록 강제한 후, 이후 모델 데이터의 이상 여부를 체크하였습니다.  
### 핵심 기술 판단
- LLM 모델의 코드 분할을 통해 적은 리소스로 실시간 구글 클라우드 GPU 서버 구현
- 서버 구동 및 로그 확인을 통한 LLM 모델 입/출력 데이터 이상 여부 체크

## 구현

### 초기 구현 및 연동 단계
<img width="1137" height="883" alt="image" src="https://github.com/user-attachments/assets/da9873dd-d416-4f13-818f-946ce6824c22" />

- 초기 server.py 업로드 후, server.py에 fast.api를 통한 홈페이지 디자인.

<img width="1455" height="864" alt="image" src="https://github.com/user-attachments/assets/9893479d-8fa6-41d5-ad98-7b59b3e89bcd" />
- PDF 데이터 입력후 LLM 모델에서 중국어가 출력되는 모습, 아직 한국어를 강제하기 이전

<img width="887" height="779" alt="image" src="https://github.com/user-attachments/assets/5e163c66-a7b1-4b07-b9c6-00c721da13fb" />

- 이후 engine.py 업로드 후, 홈페이지 접속 및 PDF 데이터 입력 테스트시 나온 로그

### 중간 테스트 단계
<img width="1010" height="434" alt="image" src="https://github.com/user-attachments/assets/31ada75f-c362-4853-9c30-b9645136f9fa" />
<img width="1008" height="784" alt="image" src="https://github.com/user-attachments/assets/792e122e-555b-4cf5-b640-0c6085aacd93" />
- 적절한 책에 대한 자료가 없어, 출석 지침 PDF를 입력, 해당 PDF에서도 문제가 영어로 출력되는 모습.

### 최종
<img width="979" height="376" alt="image" src="https://github.com/user-attachments/assets/c5ba601f-3ea3-4c0e-8312-9c14af0a92a6" />
<img width="983" height="748" alt="image" src="https://github.com/user-attachments/assets/14a3889e-33b8-4153-9509-b98334e4a151" />
- 책 내용에 대한 요약을 성공적이나, 문제가 영어로 출력되는 모습.

## 프로젝트 결론
### 실험 결과 요약
- LLM 모델 정상 작동 성공 확인
- GPU서버 연동 및 구동 성공
- LLM 모델의 한국어 출력 실패(한국어 강제시 오류율 상승)
### 분석 및 한계
- 해당 모델이 사용하는 언어모델에 대한 조사부족
- GPU서버 구동에 대한 지식 미흡에 따른 개발기간 증가
### 향후 개선 방안
- Qwen 모델에 대한 추가 자료 수집
- GPU 서버 구동 및 네트워크 트래픽 관리에 대한 추가 자료 수집
- LLM 타 모델 조사
### 프로젝트 의의
