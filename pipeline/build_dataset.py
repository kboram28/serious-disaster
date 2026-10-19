"""4-2단계: 엑셀(공단 재해사례) + SIF → V2 통합 데이터셋

입력  data/processed/cases_labeled.csv     엑셀 정제·검증 결과
      data/processed/excel_extract.jsonl   엑셀 기인물·작업·재해유발요인 (GPT 추출)
      data/processed/split.csv             엑셀 db/test 구분
      data/processed/sif_labeled.csv       SIF + 재해유형
출력  data/processed/cases_v2.csv
      docs/reports/04_dataset_report.md

업종은 SIF 대분류 체계(산재보험 업종)로 맞춘다. 원래 값은 업종_원본에 보존.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data/processed"
OUT = P / "cases_v2.csv"
REPORT = ROOT / "docs/reports/04_dataset_report.md"

# 엑셀 업종 → SIF 대분류 체계
INDUSTRY_MAP = {
    "건설업": "건설업", "전기공사업": "건설업",
    "제조업": "제조업", "철강업": "제조업",
    "운수업": "운수창고통신업", "물류업": "운수창고통신업", "해운업": "운수창고통신업",
    "임업": "임업", "산림업": "임업",
    "위생업": "기타의사업", "문화재 발굴 및 보존업": "기타의사업",
}

COLUMNS = ["case_id", "출처", "split", "재해유형", "연관유형", "업종", "업종_세부", "작업분류", "작업",
           "기인물", "사고개요", "재해유발요인", "예방대책", "시도", "시군구", "연도", "업종_원본", "라벨출처"]


def excel() -> pd.DataFrame:
    e = pd.read_csv(P / "cases_labeled.csv").merge(pd.read_csv(P / "split.csv"), on="번호")
    ex = pd.DataFrame(map(json.loads, (P / "excel_extract.jsonl").read_text(encoding="utf-8").splitlines()))
    e = e.merge(ex.drop(columns="_tokens"), on="번호")
    return pd.DataFrame({
        "case_id": "E" + e["번호"].astype(str).str.zfill(3),
        "출처": "공단 재해사례",
        "split": e["split"],
        "재해유형": e["유형_검증"],
        "연관유형": e["연관유형"],
        "업종": e["업종"].map(INDUSTRY_MAP),
        "업종_세부": "",
        "작업분류": "",
        "작업": e["작업"],
        "기인물": e["기인물"],
        "사고개요": e["사고내용_교정"],
        "재해유발요인": e["재해유발요인"],
        "예방대책": e["예방대책"],
        "시도": e["시도"],
        "시군구": e["시군구"].fillna(""),
        "연도": pd.NA,
        "업종_원본": e["업종_원본"],
        "라벨출처": "사람 검수 + GPT",
    })


def sif() -> pd.DataFrame:
    s = pd.read_csv(P / "sif_labeled.csv")
    con = s["cateSeNm"] == "건설업"
    year = s["disasterOverview"].str.extract(r"(20\d\d)\s*년")[0].astype("Int64")
    clean = lambda x: re.sub(r"\s+", "", str(x)) if pd.notna(x) else ""  # '어  업' → '어업'
    return pd.DataFrame({
        "case_id": s["sif_id"],
        "출처": "SIF",
        "split": "db",
        "재해유형": s["유형"],
        "연관유형": s["연관유형"],
        "업종": s["sifLclsfNm"].where(~con, "건설업").map(clean),
        # 전업종: 중·소분류가 세부 업종 / 건설업: 대·중·소분류가 공종 > 작업 > 단위작업
        "업종_세부": (s["sifMclsfNm"] + " > " + s["sifSclsfNm"]).where(~con, ""),
        "작업분류": (s["sifLclsfNm"] + " > " + s["sifMclsfNm"] + " > " + s["sifSclsfNm"]).where(con, s["situation"]),
        "작업": "",
        "기인물": s["orgtNm"],
        "사고개요": s["disasterOverview"],
        "재해유발요인": s["disasterFactor"],
        "예방대책": s["dcrsCntrplnCn"],
        "시도": "",
        "시군구": "",
        "연도": year,
        "업종_원본": s["cateSeNm"] + " / " + s["sifLclsfNm"],
        "라벨출처": "GPT (" + s["판정버전"] + ")",
    })


def main():
    df = pd.concat([excel()[COLUMNS], sif()[COLUMNS]], ignore_index=True)
    assert df["case_id"].is_unique
    assert df["재해유형"].notna().all() and df["업종"].notna().all()
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    write_report(df)
    db = df[df["split"] == "db"]
    print(f"저장: {OUT.relative_to(ROOT)} (전체 {len(df):,} / DB {len(db):,} / test {len(df) - len(db)})")


def fill_rate(df: pd.DataFrame, col: str) -> str:
    v = df[col].fillna("").astype(str).str.strip()
    return f"{(~v.isin(['', '미상', '<NA>'])).mean():.0%}"


def write_report(df: pd.DataFrame):
    v1 = df[(df["출처"] == "공단 재해사례") & (df["split"] == "db")]
    v2 = df[df["split"] == "db"]
    rows = [("사례 수", f"{len(v1):,}", f"{len(v2):,}")]
    for col in ["재해유형", "연관유형", "업종", "작업분류", "작업", "기인물", "재해유발요인", "예방대책", "시도", "연도"]:
        rows.append((col, "-" if col in ("연관유형", "작업분류", "작업", "기인물", "재해유발요인", "연도") else fill_rate(v1, col),
                     fill_rate(v2, col)))
    ind = pd.crosstab(v2["업종"], v2["출처"], margins=True, margins_name="합계").sort_values("합계", ascending=False)
    typ = pd.crosstab(v2["재해유형"], v2["출처"], margins=True, margins_name="합계").sort_values("합계", ascending=False)

    md = f"""# V2 통합 데이터셋 리포트

- 생성: `pipeline/build_dataset.py` → `data/processed/cases_v2.csv`
- 테스트 100건(엑셀)은 `split=test`로 표시, 검색 DB에서 제외

## V1 DB vs V2 DB (테스트 제외)

V1 열은 원본 엑셀에 있던 정보 기준 (V1은 원본 값을 그대로 사용).

| 항목 | V1 | V2 |
|---|---|---|
""" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows) + f"""

- V2의 엑셀 사례 기인물·작업·재해유발요인은 GPT가 사고 내용 문장에서 추출 (`pipeline/extract_excel.py`, 추측 금지)
- 시도는 엑셀 사례에만 있음 (SIF는 장소 익명화)

## 업종 분포 (V2 DB)
```
{ind.to_string()}
```

## 재해유형 분포 (V2 DB)
```
{typ.to_string()}
```
"""
    REPORT.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
