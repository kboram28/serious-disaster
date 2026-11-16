# GPT 기반 중대재해 유형 분류 시스템

산업현장 중대재해 사례를 근거로 **재해유형 분류, 유사 사례 검색, 예방대책, 사고 발생 시 대응**을 안내하는 챗봇.
순천대학교 인공지능공학부 졸업작품(V1)을 다시 구축하며 데이터 수집·구조를 개선한 버전(V2.2).

- 개선 내용과 V1·V2 비교: [docs/final_report.md](docs/final_report.md)
- 무엇을 왜 바꿨는지: [docs/decisions.md](docs/decisions.md)

## 데이터

| 출처 | 건수 | 수집 |
|---|---|---|
| 안전보건공단 재해사례 (엑셀 정리본) | 509 | 직접 정리 → 정제·라벨 재검증 |
| 사망사고 고위험요인(SIF) 아카이브 | 6,032 | 공공데이터포털 오픈 API |

테스트용 100건(엑셀)은 검색 DB에서 제외 → 검색 DB 6,441건.

## 폴더 구조

```
data/raw/          원본 (엑셀, SIF API 응답) — 수정하지 않음
data/processed/    정제·라벨·통합 결과
collectors/        API 수집 (sif_api.py)
pipeline/          정제, 라벨 재검증, SIF 분류, 정보 추출, 통합
v1_baseline/       V1 재현 (사례 전체를 프롬프트에 입력)
v2/                V2 검색 인덱스와 RAG (rag_v22.py = 최종)
eval/              테스트셋, 질문 세트, 평가·채점 스크립트, 결과
app/               웹 화면 (Streamlit)
docs/              보고서, 결정 기록, 단계별 리포트, API 메모
sample_data/       업로드 분석 시험용 예시 파일
```

## 실행 방법

### 1. 환경 준비
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
`.env.example`을 복사해 `.env`를 만들고 키를 넣는다.
```
OPENAI_API_KEY=...
DATA_GO_KR_KEY=...      # 공공데이터포털 Decoding 키 (SIF 수집할 때만 필요)
```

### 2. 검색 인덱스 만들기 (처음 한 번)
`vectordb/`는 git에 포함되지 않으므로 처음에 한 번 만든다 (임베딩 6,441건, 몇 분 소요).
```powershell
python v2/index.py
```

### 3. 웹 화면
```powershell
streamlit run app/app.py
```

### 데이터를 처음부터 다시 만들 때 (선택)
처리 결과(`data/processed/`)가 이미 포함되어 있어 보통은 필요 없다. 아래 순서로 실행하면 재현된다.
GPT 판정 결과는 캐시 파일(`*.jsonl`)에 저장되어 있어 다시 실행해도 API를 재호출하지 않는다.

| 순서 | 명령 | 내용 |
|---|---|---|
| 1 | `python pipeline/clean.py` | 엑셀 정제·진단 |
| 2 | `python pipeline/relabel.py` | 재해유형 GPT 재판정 → 검수표 |
| 3 | `python pipeline/apply_review.py` | 사람 검수 결과 반영 |
| 4 | `python eval/make_testset.py` | 테스트 100건 분리, 블라인드 시트 |
| 5 | `python eval/finalize_testset.py` | 테스트 정답 확정 |
| 6 | `python collectors/sif_api.py` | SIF API 수집 |
| 7 | `python pipeline/label_sif.py` | SIF 재해유형 분류 |
| 8 | `python pipeline/extract_excel.py` | 엑셀 기인물·원인 추출 |
| 9 | `python pipeline/build_dataset.py` | 통합 데이터셋 |
| 10 | `python v2/index.py` | 검색 인덱스 |

### 평가
```powershell
python eval/run_eval.py v1                 # V1: 분류 100건 + 질문 43개
python eval/run_eval.py v2
python eval/score_answers.py auto          # 자동 검사
python eval/score_answers.py judge         # GPT 블라인드 채점
python eval/score_answers.py report        # 종합 리포트
# 새 질문 세트에서 V2 vs V2.1: 위 명령 뒤에 holdout (예: python eval/run_eval.py v2.1 holdout)
```

## 기술 스택

| 구분 | V1 (졸업작품) | V2.2 (재구축) |
|---|---|---|
| 웹 | Django | Streamlit 1.66, 그래프 Altair 6.3 |
| LLM | OpenAI GPT-4 | OpenAI `gpt-5.4-mini` (답변·분류·추출·사진 설명), 평가 채점 `gpt-5.5` |
| 검색 | 사례 전체를 프롬프트에 입력 (재현 기준) | 임베딩 의미 검색: `text-embedding-3-small` + numpy 코사인 유사도 (별도 벡터 DB 없음), 조건 필터는 pandas |
| 데이터 처리 | 엑셀 수작업 정리 | pandas 3.0, openpyxl 3.1, olefile (hwp 가이드 추출) |
| 데이터 수집 | 공단 재해사례 수작업 수집 | 공공데이터포털 오픈 API (requests), python-dotenv |
| 언어·도구 | Python | Python 3.13, git, AI 코딩 도구 Claude Code (Anthropic) |

벡터 DB를 따로 두지 않은 이유: 6,441건 규모에서는 numpy로 전체와 유사도를 계산해도 수 밀리초로 충분하고 설치·운영이 단순함 (결정 기록 [10]).
