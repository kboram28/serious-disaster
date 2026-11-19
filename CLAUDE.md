# CLAUDE.md: 이 저장소에서 AI가 지킬 작업 규칙

산업재해 사례·지침·법령을 연결하는 RAG 챗봇. V1(졸업작품) → V2.2(현재 기준선) → V3(FastAPI 백엔드 + React, 브랜치 `v3`).
설계 근거는 [docs/decisions.md](docs/decisions.md), 비교 기준 숫자는 [eval/results/v2.2_baseline.md](eval/results/v2.2_baseline.md).

## 1. 진행 방식

- 사용자는 이 프로젝트를 면접에서 설명할 수 있어야 한다. 작업마다 **무엇을, 어떻게, 왜** 하는지 먼저 설명한다
- 한 번에 한 단계씩 진행하고, 사용자가 확인한 뒤 다음으로 넘어간다
- **git 커밋·푸시는 사용자가 직접 한다.** AI는 커밋 메시지를 보여주고 승인을 받는다
- **측정 스크립트(평가·응답 시간·Recall·비용·부하)는 AI가 실행하지 않는다.** 실행 방법만 안내하고, 결과는 함께 해석한다
- 명령을 안내할 때 코드 블록에는 입력할 명령만 넣는다 (예상 출력은 문장으로 설명)

## 2. 폴더 구조

**그대로 두는 것 (V1·V2.2 비교 기준, 데이터 재현)**

| 폴더 | 내용 |
|---|---|
| `v1_baseline/`, `v2/` | V1 재현, V2~V2.2 RAG. **수정 금지** (새 파일 추가는 가능, 예: `v1_baseline/v1_original.py`) |
| `v1_original/` | 졸업작품 챗봇 원본 코드·실행 캡처 (decisions [24]). 기록 보관용, **수정 금지** |
| `pipeline/` | 데이터 정제·라벨링 |
| `collectors/` | 외부 API 수집. 새 수집기(지침·조문·재해사례)도 여기에 추가 |
| `eval/` | 평가 스크립트·평가셋. V3 평가도 여기에 추가 |
| `data/` | 원본(`raw/`)·가공(`processed/`) 데이터 |
| `app/` | V2.2 Streamlit 화면. V3 완료 후 `legacy/`로 이동 |

**V3에서 새로 만드는 것**

```
backend/app/
  main.py  config.py  errors.py
  api/v1/        엔드포인트: chat, classify, cases, stats, guides, feedback
  schemas/       Pydantic 요청·응답 (= API 명세)
  services/      흐름 조립 (대화 저장 등)
  repositories/  DB 읽기·쓰기만
  db/            테이블 정의, 세션
  rag/           질문 이해, 필터, 검색(vector·bm25·rrf), 답변, 인용 검증, 분류, 프롬프트
backend/tests/   backend/alembic/
frontend/        React + TypeScript + Vite
.github/workflows/
```

- 층을 지킨다: `api` → `services` → `repositories` / `rag`. `api`에서 DB나 OpenAI를 직접 부르지 않는다
- **2순위라 나중에 추가하는 것**: `worker/`(Redis·arq, 업로드 분석과 함께, decisions [19]), 업로드·로그인·관리자 엔드포인트, `infra/`(AWS 배포)
- 인증은 decisions [20]을 따른다: 1순위는 게스트 ID(대화 이력·피드백을 게스트 ID에 저장), JWT 로그인은 2순위

## 3. 코드 규칙

- API 명세서에 없는 엔드포인트·필드를 만들지 않는다. 명세를 바꿔야 하면 명세서부터 고친다
- 기능마다 테스트를 함께 작성하고, 테스트가 통과해야 커밋한다 (목표 커버리지 75%)
- 테스트에서 OpenAI를 실제로 호출하지 않는다 (녹화 응답·가짜 응답 사용)
- 파이썬은 `backend/`에만 ruff를 적용한다. 커밋 전에 `ruff check`, `ruff format`을 통과해야 한다. `v1_baseline/`, `v2/`는 모양도 바꾸지 않는다
- decisions.md의 결정과 다른 방식이 필요하면, 코드를 짜기 전에 먼저 말하고 결정 기록에 남긴다 (다음 번호부터)
- 실행 명령이 바뀌면 같은 커밋에서 이 파일의 5장도 고친다

## 4. 절대 하지 않는 것

| 대상 | 이유 |
|---|---|
| `eval/questions_holdout2.csv`, `eval/retrieval_eval.csv` 수정 | 평가셋 잠금 (decisions [16], 태그 `v3-eval-frozen`) |
| `eval/test_gold.csv`, 블라인드 채점 파일(`eval/*.xlsx`) 수정 | 사람 정답 보존 |
| `v1_baseline/`, `v2/`, `v1_original/`의 기존 파일 수정 | 비교 기준·원본 기록 |
| `.env`의 값을 코드·문서·대화에 쓰거나 출력 | API 키 보호. 스크립트는 값을 읽기만 하고 출력하지 않는다 |
| `docs/planning/` 커밋 | 개인 계획 문서 (gitignore) |
| 사용자 질문 데이터(`data/feedback/` 등) 커밋 | 개인정보 |

**사람이 반드시 읽어야 하는 코드**: 프롬프트, DB 스키마, 인증·업로드 보안, 평가 스크립트의 채점 로직.
AI는 이 부분을 작성하거나 바꾸면 작업을 마칠 때 "꼭 읽어 주세요"라고 따로 알린다.

## 5. 명령

| 구분 | 명령 |
|---|---|
| 가상환경 | `.venv\Scripts\Activate.ps1` |
| V2.2 화면 (비교용) | `streamlit run app/app.py` |
| 평가 (👤 사용자가 실행) | `python eval/run_eval.py v2.2 holdout2` / `python eval/eval_retrieval.py v2.2` / `python eval/latency_cost.py v2.2` |
| 평가셋 잠금 확인 (출력이 비어 있어야 정상) | `git diff v3-eval-frozen -- eval/questions_holdout2.csv eval/retrieval_eval.csv` |
| 백엔드 실행 | (백엔드 생성 후 채움) |
| 테스트 | (백엔드 생성 후 채움) |
| Docker | (생성 후 채움) |

## 6. 커밋 메시지

```
<대상>: <무엇을 했는지 한 줄 요약>
```

- 한국어. 숫자·결과가 있으면 요약에 넣는다 (예: `V2.2 기준선 측정: 답변 30개, 검색 Recall@10 0.475, p95 6.0초`)
- 기능 하나 = 커밋 하나
- 마지막 줄에 `Co-Authored-By:` 표시
