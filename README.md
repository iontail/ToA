# ToA

긴 문서를 청크로 나눈 뒤, 각 청크에서 Evidence Card를 만들고 전역 메모리를 구성합니다. 이후 Planner가 논리적 순서를 정하고, 그 순서대로 근거를 누적해 답안을 작성한 뒤 검증 루프로 최종 답을 확정합니다.

## 전체 알고리즘

```text
Input (q, D)
  -> Chunking
  -> Evidence Extraction (Ei)
  -> Evidence Memory (global)
  -> Planner (build traversal path P)
  -> Sequential Execution (with retrieval from memory)
  -> Draft Answer
  -> Evidence-based Verification Loop
  -> Final Answer
```

## Evidence Card

각 청크는 답 자체보다 근거 구조를 유지하는 카드로 변환됩니다.

- `Claim`: 질문과 관련된 주장
- `Condition`: 주장을 뒷받침하는 근거
- `Exception`: 주장을 제한하거나 뒤집을 수 있는 예외
- `Join Key`: 다른 청크와 연결되는 단서
- `Source Pointer`: 원문 청크 위치
- `Answer Impact`: 답을 지지, 제한, 반박, 무관 중 하나로 판단한 값

## 파일 역할

- `src/run.py`: 실험 실행 파일입니다. 데이터 로드, 청킹, Evidence Card 추출, planning, 순차 실행, 초안 작성, 검증 루프, 결과 저장을 담당합니다.
- `src/agent.py`: API 없이 로컬 `LLaMA3.1-8B-Instruct`를 `transformers`로 직접 로드하고 응답을 생성합니다.
- `src/data_utils.py`: `DetectiveQA`, `NovelQA` pickle 데이터를 읽어 질문, 선택지, 정답, 문맥 형태로 변환합니다.
- `src/utils.py`: 프롬프트, JSON 파싱, 토큰 청킹, 카드 정규화, 경로 정규화, 메모리 검색 같은 보조 기능을 제공합니다.
- `requirements.txt`: 실행에 필요한 Python 패키지 목록입니다.

## 실행 준비

의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

로컬 LLaMA 모델 경로를 지정합니다.

```powershell
$env:MODEL_PATH="C:\path\to\Meta-Llama-3.1-8B-Instruct"
```

데이터는 기존 코드와 동일하게 저장소 상위의 `datasets` 디렉터리에서 읽습니다.

```text
../datasets/DetectiveQA/human_anno.pkl
../datasets/DetectiveQA/novel_data.pkl
../datasets/NovelQA/data.pkl
```

## 실행 예시

```bash
python src/run.py --dataset DetectiveQA --sample_num 1 --agent_num 5
```

`--agent_num`은 기존 실험 인자명을 유지하지만, 현재 알고리즘에서는 문서를 나눌 청크 수로 사용됩니다. 결과는 `../results`, 로그는 `../logs`에 저장됩니다.
