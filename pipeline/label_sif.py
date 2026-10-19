"""4-1단계: SIF 6,032건에 재해유형(6종) + 연관유형 부여

엑셀과 같은 분류 기준(pipeline/relabel.py 의 GUIDE, v3)과 같은 모델로 분류한다.
건설업은 SIF 원래 재해종류(disasterType)가 있으므로 GPT 판정과 비교해 검증한다.

입력  data/raw/sif/sif_<날짜>.jsonl
출력  data/processed/sif_label_<버전>.jsonl    GPT 판정 캐시 (다시 실행해도 재호출하지 않음)
      data/processed/sif_labeled.csv            SIF + 유형/연관유형
      docs/reports/03_sif_label_report.md

사용법
  python pipeline/label_sif.py --limit 5
  python pipeline/label_sif.py
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel import MODEL, PROMPT_VERSION, judge  # 엑셀과 같은 기준·모델

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/sif/sif_20261020.jsonl"
# 전체 판정은 v3로 1회 실행. 이후 기준이 바뀌면 영향받는 사례만 패치 파일로 재판정해 덮어쓴다.
CACHE = ROOT / "data/processed/sif_label_v3.jsonl"
PATCHES = {
    # 버전: (파일, 재판정 대상 조건)
    "v3.1": (ROOT / "data/processed/sif_label_v3.1_patch.jsonl",
             lambda row, base: any(w in row["disasterOverview"] for w in ("깔려", "깔림")) and base["유형"] != "깔림"),
}
OUT = ROOT / "data/processed/sif_labeled.csv"
REPORT = ROOT / "docs/reports/03_sif_label_report.md"

# SIF 재해종류 → 6종 (뜻이 하나로 정해지는 것만. 붕괴·전도는 상황에 따라 달라 비교에서 제외)
SIF_TO_OURS = {
    "추락": "떨어짐", "낙하": "맞음", "깔림": "깔림", "끼임": "끼임", "부딪힘": "부딪힘",
    "감전": "기타", "익사": "기타", "화상": "기타", "질식": "기타", "이상온도 접촉": "기타",
}


def sif_id(r) -> str:
    """수집할 때마다 순서가 바뀌어도 같은 사례는 같은 ID가 되도록 내용으로 만든 ID"""
    key = f"{r['disasterOverview']}|{r['orgtNm']}|{r['disasterFactor']}"
    return "S" + hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["sif_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def run(client, rows: list, cache: dict, path: Path):
    """rows 를 판정해 path 에 한 줄씩 저장하고 cache 에 반영"""
    def work(row):
        r = judge_retry(client, row["disasterOverview"])
        r["sif_id"] = row["sif_id"]
        return r

    with ThreadPoolExecutor(max_workers=6) as pool, path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(pool.map(work, rows), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            cache[r["sif_id"]] = r
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}")


def judge_retry(client, text):
    for i in range(5):
        try:
            return judge(client, text)
        except Exception as e:
            if i == 4:
                raise
            time.sleep(5 * 2 ** i)
            print(f"  재시도 {i + 1} ({type(e).__name__})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())

    df = pd.read_json(SRC, lines=True)
    df["sif_id"] = df.apply(sif_id, axis=1)
    df = df.drop_duplicates("sif_id")
    if args.limit:
        df = df.head(args.limit)

    # 1) 기본 판정 (v3, 전체)
    base = load_cache(CACHE)
    rows = [row for _, row in df.iterrows() if row["sif_id"] not in base]
    print(f"[기본 v3] 전체 {len(df)} / 캐시 {len(df) - len(rows)} / 새로 판정 {len(rows)} (모델 {MODEL})")
    run(client, rows, base, CACHE)
    final = {s: {**base[s], "판정버전": "v3"} for s in df["sif_id"]}
    base_label = {s: r["유형"] for s, r in final.items()}  # 패치 전 유형 (리포트용)

    # 2) 패치: 기준이 바뀐 부분만 재판정해 덮어쓰기
    for ver, (path, need) in PATCHES.items():
        assert ver == PROMPT_VERSION, "패치는 현재 GUIDE 버전으로만 실행"
        patch = load_cache(path)
        targets = [row for _, row in df.iterrows() if row["sif_id"] in patch or need(row, base[row["sif_id"]])]
        rows = [row for row in targets if row["sif_id"] not in patch]
        print(f"[패치 {ver}] 대상 {len(targets)} / 새로 판정 {len(rows)}")
        run(client, rows, patch, path)
        for row in targets:
            final[row["sif_id"]] = {**patch[row["sif_id"]], "판정버전": ver}

    g = pd.DataFrame([{**final[s], "sif_id": s} for s in df["sif_id"]])
    tok = pd.DataFrame(list(g["_tokens"]))
    print(f"사용 토큰(최종 판정 기준): 입력 {tok['in'].sum():,} / 출력 {tok['out'].sum():,}")

    g = g.rename(columns={"세부유형": "연관유형", "근거": "GPT_근거", "확신도": "GPT_확신도"})
    g["패치전_유형"] = g["sif_id"].map(base_label)
    out = df.merge(g[["sif_id", "유형", "연관유형", "GPT_근거", "GPT_확신도", "판정버전", "패치전_유형"]], on="sif_id")
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    write_report(out)
    print(f"저장: {OUT.relative_to(ROOT)} / 리포트: {REPORT.relative_to(ROOT)}")


def write_report(d: pd.DataFrame):
    con = d[d["cateSeNm"] == "건설업"].copy()
    con["SIF유형"] = con["disasterType"].fillna("(없음)")
    ct = pd.crosstab(con["SIF유형"], con["유형"], rownames=["SIF 원래"], colnames=["GPT"])
    comp = con[con["SIF유형"].isin(SIF_TO_OURS)]
    agree = (comp["SIF유형"].map(SIF_TO_OURS) == comp["유형"])
    by = (comp.assign(일치=agree).groupby("SIF유형")["일치"].agg(["sum", "count"])
          .sort_values("count", ascending=False))
    diff = comp[~agree]

    md = f"""# SIF 재해유형 분류 리포트

- 생성: `pipeline/label_sif.py` / 모델 `{MODEL}` / 분류 기준 `{PROMPT_VERSION}` (엑셀과 동일)
- 입력: SIF 개요(`disasterOverview`)만 사용 (엑셀 사고내용과 같은 조건)

## 분류 결과 ({len(d):,}건)

| 유형 | 전체 | 건설업 | 전업종 |
|---|---|---|---|
""" + "\n".join(
        f"| {t} | {(d['유형'] == t).sum():,} | {((d['유형'] == t) & (d['cateSeNm'] == '건설업')).sum():,} | "
        f"{((d['유형'] == t) & (d['cateSeNm'] == '전업종')).sum():,} |"
        for t in ["떨어짐", "끼임", "깔림", "맞음", "부딪힘", "기타"]
    ) + f"""

패치 재판정 ({PROMPT_VERSION}, '깔려/깔림' 포함인데 깔림이 아니었던 사례): {(d['판정버전'] != 'v3').sum()}건 →
유형 변경 {(d['유형'] != d['패치전_유형']).sum()}건
({', '.join(f'{a}→{b} {n}' for (a, b), n in d[d['유형'] != d['패치전_유형']].groupby(['패치전_유형', '유형']).size().items())})

연관유형 상위: {', '.join(f'{k} {v}' for k, v in d['연관유형'].value_counts().head(10).items())}

확신도: {', '.join(f'{k} {v}' for k, v in d['GPT_확신도'].value_counts().items())}

## 검증: 건설업 SIF 원래 재해종류와 비교

뜻이 하나로 정해지는 SIF 유형({len(comp):,}건)만 6종으로 바꿔 비교. 붕괴·전도 등은 상황에 따라
깔림/기타/떨어짐 등으로 갈려 비교에서 제외하고 분포만 표시.

**일치율: {agree.mean():.1%}** ({agree.sum():,}/{len(comp):,})

| SIF 원래 | 6종 변환 | 일치 | 건수 | 일치율 |
|---|---|---|---|---|
""" + "\n".join(
        f"| {k} | {SIF_TO_OURS[k]} | {int(r['sum'])} | {int(r['count'])} | {r['sum'] / r['count']:.0%} |"
        for k, r in by.iterrows()
    ) + f"""

### 전체 교차표 (건설업)
```
{ct.to_string()}
```

### 불일치 예시 (최대 30건)

| SIF 원래 | GPT | 연관 | 개요 |
|---|---|---|---|
""" + "\n".join(
        f"| {r['SIF유형']} | {r['유형']} | {r['연관유형']} | {str(r['disasterOverview'])[:60]}… |"
        for _, r in diff.head(30).iterrows()
    ) + "\n"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
