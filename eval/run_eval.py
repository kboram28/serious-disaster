"""2-3 / 5단계: 시스템(V1·V2)을 고정된 평가셋으로 실행하고 결과 저장

입력  eval/test_gold.csv      분류 테스트 100건
      eval/questions.csv      질문 세트 43개
출력  eval/results/<시스템>_classify.jsonl
      eval/results/<시스템>_answers.jsonl
      eval/results/<시스템>_summary.md

시스템 모듈은 classify(사고내용) -> {"유형", "usage"} 와
ask(질문, history) -> {"answer", "usage"} 를 제공해야 한다.
이미 결과가 있는 문항은 다시 호출하지 않는다(비용 절약).

사용법
  python eval/run_eval.py v1
  python eval/run_eval.py v2.1 holdout    # 새 질문 세트로 답변만 생성
  python eval/run_eval.py v2.2 holdout2   # V3 평가용 30문항 (V2.2 기준선)
"""
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "eval/test_gold.csv"
QUESTIONS = ROOT / "eval/questions.csv"
RESULTS = ROOT / "eval/results"
SYSTEMS = {"v1": ROOT / "v1_baseline/v1.py", "v2": ROOT / "v2/rag.py", "v2.1": ROOT / "v2/rag_v21.py",
           "v2.2": ROOT / "v2/rag_v22.py"}
QUESTION_SETS = {"main": ROOT / "eval/questions.csv", "holdout": ROOT / "eval/questions_holdout.csv",
                 "holdout2": ROOT / "eval/questions_holdout2.csv"}  # V3 평가용, 수정 금지 (태그 v3-eval-frozen)
TYPES = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘", "기타"]


def load_system(name: str):
    spec = importlib.util.spec_from_file_location(name, SYSTEMS[name])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def retry(fn, *args, tries=4):
    for i in range(tries):
        try:
            return fn(*args)
        except Exception as e:  # 속도 제한 등 일시 오류
            if i == tries - 1:
                raise
            wait = 5 * 2 ** i
            print(f"  재시도 {i + 1} ({type(e).__name__}), {wait}초 대기")
            time.sleep(wait)


def timed(fn, *args):
    t = time.perf_counter()
    out = retry(fn, *args)
    out["sec"] = round(time.perf_counter() - t, 2)
    return out


def load_done(path: Path, key: str) -> dict:
    if not path.exists():
        return {}
    return {r[key]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def run_classify(sys_mod, name: str) -> pd.DataFrame:
    path = RESULTS / f"{name}_classify.jsonl"
    done = load_done(path, "문항ID")
    gold = pd.read_csv(GOLD)
    todo = gold[~gold["문항ID"].isin(done)]
    print(f"[분류] 전체 {len(gold)} / 완료 {len(done)} / 실행 {len(todo)}")

    def work(row):
        r = timed(sys_mod.classify, row["사고내용"])
        return {"문항ID": row["문항ID"], "예측": r["유형"], "usage": r["usage"], "sec": r["sec"]}

    with ThreadPoolExecutor(max_workers=4) as pool, path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(pool.map(work, [row for _, row in todo.iterrows()]), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done[r["문항ID"]] = r
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")

    res = pd.DataFrame(done.values())
    return gold.merge(res, on="문항ID")


def cited_cases(answer: str) -> list[str]:
    """답변에 인용된 사례 ID 추출
    V1: [12], 12번, 사례 12 → 'E012' / V2: [E012], [S1a2b3c4d5e]"""
    ids = set(re.findall(r"\b(E\d{3}|S[0-9a-f]{10})\b", answer))
    if not ids:
        nums = re.findall(r"\[(\d{1,4})\]|(\d{1,4})\s*번|사례\s*(\d{1,4})", answer)
        ids = {f"E{int(n):03d}" for t in nums for n in t if n}
    return sorted(ids)


def run_questions(sys_mod, name: str, qset: str = "main") -> pd.DataFrame:
    suffix = "" if qset == "main" else f"_{qset}"
    path = RESULTS / f"{name}{suffix}_answers.jsonl"
    done = load_done(path, "질문ID")
    qs = pd.read_csv(QUESTION_SETS[qset], dtype=str).fillna("")
    assert (qs["질문"] != "").all(), "빈 질문이 있음"
    print(f"[질문] 전체 {len(qs)} / 완료 {len(done)} / 실행 {len(qs) - len(done)}")

    def work(row):
        history = []
        if row["이전질문ID"]:  # 이어지는 질문: 이전 질문과 그 답변을 함께 전달
            prev = done.get(row["이전질문ID"]) or work(qs[qs["질문ID"] == row["이전질문ID"]].iloc[0])
            history = [(prev["질문"], prev["answer"])]
        r = timed(sys_mod.ask, row["질문"], history)
        return {"질문ID": row["질문ID"], "질문": row["질문"], "answer": r["answer"],
                "인용사례": cited_cases(r["answer"]), "검색사례": r.get("retrieved", []),
                "검색조건": r.get("query"), "usage": r["usage"], "sec": r["sec"]}

    # 이전 질문이 없는 것부터 병렬 실행 → 이어지는 질문은 그 다음에
    first = [row for _, row in qs.iterrows() if row["질문ID"] not in done and not row["이전질문ID"]]
    second = [row for _, row in qs.iterrows() if row["질문ID"] not in done and row["이전질문ID"]]
    # holdout2는 응답 시간(p50·p95)도 기준선으로 쓰므로 동시 호출 없이 하나씩 실행 (동시 실행은 시간이 섞임)
    workers = 1 if qset == "holdout2" else 4
    with path.open("a", encoding="utf-8") as f:
        for batch in (first, second):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for r in pool.map(work, batch):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    done[r["질문ID"]] = r
                    print(f"  {r['질문ID']} {r['sec']}초")
    return qs.merge(pd.DataFrame(done.values()).drop(columns="질문"), on="질문ID")


def f1_table(df: pd.DataFrame) -> tuple[str, float]:
    rows, f1s = [], []
    for t in TYPES:
        tp = ((df["예측"] == t) & (df["정답_유형"] == t)).sum()
        p = tp / max((df["예측"] == t).sum(), 1)
        r = tp / max((df["정답_유형"] == t).sum(), 1)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        rows.append(f"| {t} | {(df['정답_유형'] == t).sum()} | {p:.2f} | {r:.2f} | {f1:.2f} |")
    return "\n".join(rows), sum(f1s) / len(f1s)


def usage_stats(records: pd.DataFrame) -> dict:
    u = pd.DataFrame(list(records["usage"]))
    return {"calls": len(u), "in": int(u["in"].sum()), "cached": int(u["cached"].sum()),
            "out": int(u["out"].sum()), "in_avg": int(u["in"].mean()),
            "sec_avg": round(records["sec"].mean(), 1)}


def write_summary(name: str, cls: pd.DataFrame, ans: pd.DataFrame):
    acc = (cls["예측"] == cls["정답_유형"]).mean()
    acc_blind = (cls["예측"] == cls["블라인드_유형"]).mean()
    table, macro = f1_table(cls)
    wrong = cls[cls["예측"] != cls["정답_유형"]]
    uc, ua = usage_stats(cls), usage_stats(ans)
    ct = pd.crosstab(cls["정답_유형"], cls["예측"], rownames=["정답"], colnames=["예측"])
    cite_rate = (ans["인용사례"].map(len) > 0).mean()

    md = f"""# {name.upper()} 평가 결과

## 1. 재해유형 분류 (테스트 {len(cls)}건)

| 지표 | 값 |
|---|---|
| 정확도 (조정 정답) | **{acc:.1%}** |
| 정확도 (블라인드 원래 정답) | {acc_blind:.1%} |
| Macro-F1 | {macro:.3f} |

| 유형 | 건수 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
{table}

```
{ct.to_string()}
```

틀린 문항:

| 문항 | 정답 | 예측 | 사고 내용 |
|---|---|---|---|
""" + "\n".join(f"| {r['문항ID']} | {r['정답_유형']} | {r['예측']} | {r['사고내용'][:45]}… |"
                for _, r in wrong.iterrows()) + f"""

## 2. 질문 세트 ({len(ans)}개)

- 사례 번호를 인용한 답변: {cite_rate:.0%}
- 답변 품질 채점은 V2 결과가 나온 뒤 같은 기준으로 함께 진행

## 3. 비용·속도

| | 분류 | 질문 |
|---|---|---|
| 호출 수 | {uc['calls']} | {ua['calls']} |
| 호출당 평균 입력 토큰 | {uc['in_avg']:,} | {ua['in_avg']:,} |
| 총 입력 토큰 (캐시 적용분) | {uc['in']:,} ({uc['cached']:,}) | {ua['in']:,} ({ua['cached']:,}) |
| 총 출력 토큰 | {uc['out']:,} | {ua['out']:,} |
| 평균 응답 시간 | {uc['sec_avg']}초 | {ua['sec_avg']}초 |
"""
    (RESULTS / f"{name}_summary.md").write_text(md, encoding="utf-8")
    print(f"\n분류 정확도 {acc:.1%} (블라인드 {acc_blind:.1%}), Macro-F1 {macro:.3f}")
    print(f"결과 요약: {(RESULTS / f'{name}_summary.md').relative_to(ROOT)}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    name = sys.argv[1] if len(sys.argv) > 1 else "v1"
    qset = sys.argv[2] if len(sys.argv) > 2 else "main"
    RESULTS.mkdir(parents=True, exist_ok=True)
    mod = load_system(name)
    if qset != "main":  # 새 질문 세트: 답변만 생성 (분류 테스트는 V2와 동일 로직이라 생략)
        run_questions(mod, name, qset)
        return
    cls = run_classify(mod, name)
    ans = run_questions(mod, name)
    write_summary(name, cls, ans)


if __name__ == "__main__":
    main()
