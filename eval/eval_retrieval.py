"""검색 평가: eval/retrieval_eval.csv(200개, 수정 금지)로 Recall@5·10, MRR 측정

정답 = 질문을 만든 원래 사례(case_id). 검색 상위 10개 안에서 정답이 몇 위인지 기록한다.
- Recall@k: 정답이 상위 k개 안에 있는 질문의 비율
- MRR: 1/정답순위 의 평균 (10위 밖이면 0). 1위 1.0, 2위 0.5, 5위 0.2

V2.2는 두 가지 방식을 함께 잰다
- embed:    질문 문장을 그대로 임베딩해 전체 DB에서 검색 (검색기 자체의 실력)
- pipeline: 실제 V2.2 흐름. 질문 이해(gpt-5.4-mini) → 조건 필터(업종·재해유형·시도) → '검색문장'으로 검색
            필터가 잘못 걸리면 정답이 후보에서 빠질 수 있어 embed와 차이가 날 수 있음
V3는 같은 파일·같은 지표로 API를 통해 잰다 (추가 예정)

사용법
  python eval/eval_retrieval.py v2.2
출력  eval/results/<시스템>_retrieval.jsonl (질문별 순위), eval/results/<시스템>_retrieval_summary.md
"""
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval/retrieval_eval.csv"
RESULTS = ROOT / "eval/results"
SYSTEMS = {"v2.2": ROOT / "v2/rag_v22.py"}
K = 10


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SYSTEMS[name])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rag  # rag_v22 는 rag 모듈의 설정만 바꾼 것. 검색 함수는 rag 에 있음


def rank_of(gold: str, ids: list[str]):
    return ids.index(gold) + 1 if gold in ids else None


def run_v22(rag, row) -> dict:
    df, _ = rag.db()
    everything = np.ones(len(df), dtype=bool)

    t = time.perf_counter()
    embed_ids = rag.search(row["질문"], everything, K)["case_id"].tolist()
    embed_sec = time.perf_counter() - t

    usage = {"in": 0, "out": 0, "cached": 0}
    t = time.perf_counter()
    q = rag.parse_query(row["질문"], [], usage)
    parse_sec = time.perf_counter() - t
    mask, _ = rag.filter_mask(df, q)
    if mask.sum() == 0:  # V2.2 ask() 와 같은 처리
        mask = everything
    t = time.perf_counter()
    pipe_ids = rag.search(q["검색문장"], mask, K)["case_id"].tolist()
    search_sec = time.perf_counter() - t

    gold_in_filter = bool(mask[df.index[df["case_id"] == row["정답_case_id"]][0]])
    return {"평가ID": row["평가ID"], "정답": row["정답_case_id"],
            "embed_rank": rank_of(row["정답_case_id"], embed_ids),
            "pipeline_rank": rank_of(row["정답_case_id"], pipe_ids),
            "필터": {k: q[k] for k in ("업종", "재해유형", "시도") if q[k]},
            "정답이_필터_안에": gold_in_filter, "검색문장": q["검색문장"],
            "embed_sec": round(embed_sec, 3), "parse_sec": round(parse_sec, 3), "search_sec": round(search_sec, 3),
            "usage": usage, "embed_top": embed_ids, "pipeline_top": pipe_ids}


def scores(ranks: pd.Series) -> dict:
    r = ranks.astype("float")
    return {"Recall@5": (r <= 5).mean(), "Recall@10": (r <= 10).mean(),
            "MRR": r.map(lambda x: 0 if np.isnan(x) else 1 / x).mean()}


def summarize(name: str, res: pd.DataFrame, ev: pd.DataFrame) -> str:
    d = ev.merge(res, on="평가ID")
    rows = []
    for mode, label in (("embed", "질문 그대로 임베딩 검색"), ("pipeline", "V2.2 실제 흐름 (질문 이해 → 필터 → 검색)")):
        s = scores(d[f"{mode}_rank"])
        rows.append(f"| {label} | {s['Recall@5']:.3f} | {s['Recall@10']:.3f} | {s['MRR']:.3f} |")
    by_type = []
    for t, g in d.groupby("재해유형"):
        e, p = scores(g["embed_rank"]), scores(g["pipeline_rank"])
        by_type.append(f"| {t} | {len(g)} | {e['Recall@10']:.2f} | {p['Recall@10']:.2f} | {e['MRR']:.2f} | {p['MRR']:.2f} |")
    filtered = d[d["필터"].map(bool)]
    miss = filtered[~filtered["정답이_필터_안에"]]
    p95 = lambda c: d[c].quantile(0.95)
    return f"""# {name.upper()} 검색 평가 (eval/retrieval_eval.csv, {len(d)}개)

정답은 질문을 만든 원래 사례 1건. 비슷한 다른 사례가 위에 와도 오답으로 처리하므로 절대 점수는 낮게 나온다.

| 방식 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|
{chr(10).join(rows)}

## 재해유형별

| 유형 | 개수 | Recall@10 (embed) | Recall@10 (pipeline) | MRR (embed) | MRR (pipeline) |
|---|---|---|---|---|---|
{chr(10).join(by_type)}

## 조건 필터 영향 (pipeline)

- 필터가 걸린 질문: {len(filtered)}개 / {len(d)}개
- 그중 정답이 필터에 걸러져 **처음부터 찾을 수 없었던** 질문: {len(miss)}개 ({', '.join(miss['평가ID']) or '없음'})

## 검색 시간 (질문 1개)

| 단계 | 평균 | p95 |
|---|---|---|
| 질문 임베딩 + 검색 (embed) | {d['embed_sec'].mean():.2f}초 | {p95('embed_sec'):.2f}초 |
| 질문 이해 (gpt-5.4-mini) | {d['parse_sec'].mean():.2f}초 | {p95('parse_sec'):.2f}초 |
| 검색문장 임베딩 + 검색 | {d['search_sec'].mean():.2f}초 | {p95('search_sec'):.2f}초 |

질문 이해 토큰: 평균 입력 {d['usage'].map(lambda u: u['in']).mean():,.0f} / 출력 {d['usage'].map(lambda u: u['out']).mean():,.0f}
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    name = sys.argv[1] if len(sys.argv) > 1 else "v2.2"
    rag = load(name)
    ev = pd.read_csv(EVAL, dtype=str, keep_default_na=False)
    path = RESULTS / f"{name}_retrieval.jsonl"
    done = {}
    if path.exists():  # 중간에 끊겼으면 이어서
        done = {r["평가ID"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}
    todo = [row for _, row in ev.iterrows() if row["평가ID"] not in done]
    print(f"[검색] 전체 {len(ev)} / 완료 {len(done)} / 실행 {len(todo)}")
    rag.db()  # 데이터·임베딩을 먼저 한 번 읽어 둠 (첫 질문 시간에 섞이지 않게)

    with ThreadPoolExecutor(4) as pool, path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(pool.map(lambda row: run_v22(rag, row), todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done[r["평가ID"]] = r
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")

    md = summarize(name, pd.DataFrame(done.values()), ev)
    out = RESULTS / f"{name}_retrieval_summary.md"
    out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
