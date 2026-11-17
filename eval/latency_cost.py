"""응답 시간·비용 집계 (API를 다시 호출하지 않고, 이미 저장된 평가 결과만 읽어서 계산)

입력  eval/results/<시스템>_holdout2_answers.jsonl   30문항 답변 (문항별 시간·토큰, 순차 실행)
      eval/results/<시스템>_retrieval.jsonl           검색 평가 200개 (단계별 시간)
출력  eval/results/<시스템>_latency_cost.md

단가: OpenAI 공식 가격표 (https://developers.openai.com/api/docs/pricing, 2026-11-17 확인, Standard 등급)
사용법
  python eval/latency_cost.py v2.2
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval/results"
QUESTIONS = ROOT / "eval/questions_holdout2.csv"

PRICE_DATE = "2026-11-17"
PRICE = {"in": 0.75, "cached": 0.075, "out": 4.50}  # gpt-5.4-mini, 달러 / 100만 토큰
EMBED_PRICE = 0.02                                   # text-embedding-3-small, 달러 / 100만 토큰
KRW_PER_USD = 1400                                   # 환율 가정 (원화 환산용 어림값)
V2_BASELINE_IN = 3410                                # V2 질문당 입력 토큰 (decisions [11])


def cost_usd(u: dict) -> float:
    """캐시된 입력은 싼 단가, 나머지 입력은 정가, 출력은 출력 단가"""
    return ((u["in"] - u["cached"]) * PRICE["in"] + u["cached"] * PRICE["cached"] + u["out"] * PRICE["out"]) / 1e6


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    name = sys.argv[1] if len(sys.argv) > 1 else "v2.2"
    ans = read_jsonl(RESULTS / f"{name}_holdout2_answers.jsonl")
    ret = read_jsonl(RESULTS / f"{name}_retrieval.jsonl")
    qs = pd.read_csv(QUESTIONS, dtype=str, keep_default_na=False)[["질문ID", "질문유형"]]
    ans = qs.merge(ans, on="질문ID")

    for k in ("in", "cached", "out"):
        ans[k] = ans["usage"].map(lambda u: u[k])
    ans["usd"] = ans["usage"].map(cost_usd)

    sec = ans["sec"]
    usd = ans["usd"].mean()
    by_type = ans.groupby("질문유형", sort=False).agg(개수=("sec", "size"), 평균초=("sec", "mean"), 최대초=("sec", "max"),
                                                    입력토큰=("in", "mean"), 출력토큰=("out", "mean"))
    type_rows = "\n".join(f"| {t} | {int(r.개수)} | {r.평균초:.1f} | {r.최대초:.1f} | {r.입력토큰:,.0f} | {r.출력토큰:,.0f} |"
                          for t, r in by_type.iterrows())
    parse, search = ret["parse_sec"].mean(), ret["search_sec"].mean()
    slow = ans.nlargest(3, "sec")

    md = f"""# {name.upper()} 응답 시간·비용 (기준선)

측정 조건: holdout2 30문항을 **하나씩 순차 실행** (동시 요청 없음), 내 PC → OpenAI API, 모델 gpt-5.4-mini.
시간은 질문 이해 + 검색 + 답변 생성 전체 (스트리밍 없음 = 답변이 다 만들어진 뒤 한 번에 표시).

## 1. 응답 시간 (질문 1개)

| 지표 | 값 |
|---|---|
| 평균 | {sec.mean():.2f}초 |
| p50 (중간값) | {sec.quantile(0.5):.2f}초 |
| **p95** | **{sec.quantile(0.95):.2f}초** |
| 최대 | {sec.max():.2f}초 ({slow.iloc[0]['질문ID']}) |

### 단계별 (검색 평가 200개의 평균으로 추정)

| 단계 | 평균 | 비중 |
|---|---|---|
| 질문 이해 (gpt-5.4-mini) | {parse:.2f}초 | {parse / sec.mean():.0%} |
| 임베딩 + 검색 (numpy) | {search:.2f}초 | {search / sec.mean():.0%} |
| 답변 생성 (나머지) | {sec.mean() - parse - search:.2f}초 | {(sec.mean() - parse - search) / sec.mean():.0%} |

### 질문 유형별

| 유형 | 개수 | 평균(초) | 최대(초) | 입력 토큰 | 출력 토큰 |
|---|---|---|---|---|---|
{type_rows}

## 2. 토큰·비용 (질문 1개, OpenAI 호출 2회: 질문 이해 + 답변)

| 지표 | 값 |
|---|---|
| 입력 토큰 평균 | {ans['in'].mean():,.0f} (V2 기준 {V2_BASELINE_IN:,}) |
| 캐시 적용 입력 | {ans['cached'].mean():,.0f} |
| 출력 토큰 평균 | {ans['out'].mean():,.0f} |
| **질문당 비용** | **${usd:.4f} ≈ {usd * KRW_PER_USD:.1f}원** |
| 1,000문항 비용 | ${usd * 1000:.2f} ≈ {usd * 1000 * KRW_PER_USD:,.0f}원 |

- 비용 구성: 입력 ${(ans['in'] - ans['cached']).mean() * PRICE['in'] / 1e6:.4f} + 출력 ${ans['out'].mean() * PRICE['out'] / 1e6:.4f}
  → 출력 토큰은 입력의 약 1/5뿐이지만 단가가 6배라 **비용의 절반 이상**. 답변 길이가 비용을 좌우함
- 캐시 적용 0: OpenAI 프롬프트 캐시는 앞부분 1,024토큰 이상이 매번 같아야 적용되는데, 고정된 시스템 프롬프트가 그보다 짧고 뒤는 질문마다 바뀜
- 임베딩(질문 1~2문장)은 질문당 약 $0.000002로 무시할 수 있음

단가 ({PRICE_DATE} 확인, 달러/100만 토큰): gpt-5.4-mini 입력 {PRICE['in']} · 캐시 입력 {PRICE['cached']} · 출력 {PRICE['out']},
text-embedding-3-small {EMBED_PRICE}. 환율 1달러 = {KRW_PER_USD:,}원 가정.
"""
    out = RESULTS / f"{name}_latency_cost.md"
    out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
