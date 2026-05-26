# ToA 파일 구성

이 저장소는 긴 문서를 여러 조각으로 나누고, 여러 LLM 에이전트가 각자 추론한 뒤 의견 교환과 투표를 통해 최종 답을 정하는 실험용 코드입니다.

## 루트 파일

- `README.md`: 프로젝트 파일 구성을 설명하는 문서입니다.
- `requirements.txt`: 실행에 필요한 Python 패키지 목록입니다. `torch`, `transformers`, `openai`, `datasets`, `tiktoken` 등이 포함되어 있습니다.

## `src` 디렉터리

- `src/run.py`: 실험 실행 진입점입니다. 명령행 인자를 파싱하고, 데이터 로드, 에이전트 생성, 문서 분할, 초기 답변, 의견 교환, 답변 보정, 최종 투표, 결과 저장까지 전체 흐름을 제어합니다.
- `src/agent.py`: LLM 에이전트 클래스를 정의합니다. 공통 상태와 초기화/리셋 로직을 가진 `BaseAgent`, DeepSeek API용 `DeepSeek`, OpenAI 호환 로컬 LLaMA 서버용 `LocalLlama`가 있습니다.
- `src/data_utils.py`: 데이터셋 로딩과 전처리를 담당합니다. `DetectiveQA`, `NovelQA` 형식의 pickle 데이터를 읽고, 질문/선택지/정답/문맥 형태로 변환합니다.
- `src/utils.py`: 프롬프트 템플릿, JSON 파싱, 숫자 추출, 순열 생성, 다수결 처리, 동률 해소, 토큰 기준 문서 분할/잘라내기 같은 보조 기능을 모아 둔 파일입니다.

## 생성 파일

- `src/__pycache__/`: Python이 모듈 실행 시 자동 생성하는 바이트코드 캐시 디렉터리입니다. 직접 수정하거나 실행할 필요는 없습니다.
- `src/__pycache__/*.pyc`: `agent.py`, `data_utils.py`, `utils.py` 등을 실행하면서 생성된 캐시 파일입니다. 소스 코드는 아니며 삭제해도 다시 생성될 수 있습니다.

## 실행 시 예상되는 외부 디렉터리

현재 저장소에는 없지만 코드에서 다음 경로를 사용합니다.

- `../datasets/`: `DetectiveQA`, `NovelQA` 데이터 pickle 파일을 찾는 위치입니다.
- `../logs/`: 실행 로그가 저장되는 위치입니다.
- `../results/`: 실험 결과 JSON 파일이 저장되는 위치입니다.
