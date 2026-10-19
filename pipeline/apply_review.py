"""1-3단계: 사람 검수 결과를 정제 데이터에 합치고 라벨 정확도를 계산

입력  data/processed/cases_clean.csv
      data/processed/relabel_review.xlsx     사람이 검수한 표 (수정하지 않고 읽기만 함)
      data/processed/relabel_gpt_v1.jsonl    GPT 1차 판정
      data/processed/relabel_gpt_v2.jsonl    GPT 2차 판정
출력  data/processed/cases_labeled.csv
      docs/reports/02_relabel_report.md
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/processed/cases_clean.csv"
REVIEW = ROOT / "data/processed/relabel_review.xlsx"
GPT = {v: ROOT / f"data/processed/relabel_gpt_{v}.jsonl" for v in ("v1", "v2")}
OUT = ROOT / "data/processed/cases_labeled.csv"
REPORT = ROOT / "docs/reports/02_relabel_report.md"

# 검수 후 확정한 기준을 적용한 추가 수정 (엑셀행: (유형, 연관유형, 이유))
# 기준 [5]: 토사·굴착면·건물 붕괴로 '매몰' → 기타(연관: 무너짐) / 물체가 '쓰러져' 깔림 → 깔림
#           구조물 붕괴로 사람이 떨어짐 → 떨어짐(연관: 무너짐)
MAEMOL = "매몰은 깔림과 구분: 기타(연관 무너짐)"
OVERRIDES = {
    223: ("기타", "무너짐", "인근 토사가 무너지며 깔림 → " + MAEMOL),
    254: ("기타", "무너짐", "옹벽이 무너져 매몰 → " + MAEMOL),
    257: ("기타", "무너짐", "굴착사면 붕괴 매몰 → " + MAEMOL),
    258: ("기타", "무너짐", "굴착사면 붕괴 매몰 → " + MAEMOL),
    261: ("기타", "무너짐", "토사 붕괴 매몰 → " + MAEMOL),
    275: ("기타", "무너짐", "거푸집 붕괴 매몰 → " + MAEMOL),
    276: ("기타", "무너짐", "토사·구조물 매몰 → " + MAEMOL),
    277: ("기타", "무너짐", "흙더미 붕괴 매몰 → " + MAEMOL),
    281: ("기타", "무너짐", "토사 붕괴 매몰 → " + MAEMOL),
    283: ("기타", "무너짐", "법면 붕괴, 토사에 깔림 → " + MAEMOL),
    482: ("기타", "무너짐", "굴착지 측면 붕괴, 토사에 깔림 → " + MAEMOL),
    500: ("기타", "무너짐", "건축물 붕괴 매몰 → " + MAEMOL),
    284: ("떨어짐", "무너짐", "슬라브 붕괴로 사람이 떨어짐 → 결과 기준 떨어짐"),
    # 기준 [8] (SIF 검증 중 확정): 충돌 후 깔림 → 깔림 / 떨어진 뒤 중량물에 깔림 → 깔림
    285: ("깔림", "떨어짐", "슬라브와 함께 떨어진 뒤 슬라브에 깔림 → 기준[8] 깔림"),
    196: ("깔림", "부딪힘", "후진 살수차에 부딪히며 깔림 → 기준[8] 충돌 후 깔림 = 깔림"),
    206: ("깔림", "부딪힘", "이동 중인 굴착기에 깔림 → 기준[8] 깔림"),
}


def load_gpt(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["번호"]: r["유형"] for r in rows}


def main():
    clean = pd.read_csv(CLEAN)
    rv = pd.read_excel(REVIEW)

    # 세부유형 → 연관유형 (결과에 함께 관여한 원인. 기타는 구체 유형. 없으면 주 유형과 같음)
    rv = rv.rename(columns={"최종_유형": "유형_검증", "최종_세부유형": "연관유형"})
    rv["수정근거"] = rv["검수메모"].fillna("")
    rv["수정출처"] = "일치(미검수)"                      # 원본·GPT가 같아 검수 대상이 아니었던 행
    rv.loc[rv["검수필요"], "수정출처"] = "검수: GPT 승인"
    rv.loc[rv["검수필요"] & (rv["유형_검증"] != rv["GPT_유형"]), "수정출처"] = "검수: 사람 수정"

    for row, (t, rel, why) in OVERRIDES.items():
        idx = rv.index[rv["엑셀행"] == row]
        assert len(idx) == 1, f"엑셀행 {row}을 찾을 수 없음"
        rv.loc[idx, ["유형_검증", "연관유형", "수정근거", "수정출처"]] = [t, rel, why, "기준[8]" if "기준[8]" in why else "기준[5]"]

    keep = ["번호", "유형_검증", "연관유형", "GPT_유형", "GPT_근거", "수정출처", "수정근거"]
    df = clean.merge(rv[keep], on="번호", how="left")
    assert df["유형_검증"].notna().all()
    df["GPT_v1"] = df["번호"].map(load_gpt(GPT["v1"]))
    df["GPT_v2"] = df["번호"].map(load_gpt(GPT["v2"]))
    df = df.rename(columns={"GPT_근거": "GPT_v2_근거"}).drop(columns="GPT_유형")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    write_report(df)
    print(f"저장: {OUT.relative_to(ROOT)} ({len(df)}행)")
    print(f"리포트: {REPORT.relative_to(ROOT)}")


DESC = {
    "일치(미검수)": "원본과 GPT가 같아 검수 대상에서 제외",
    "검수: GPT 승인": "사람이 확인하고 GPT 판정을 그대로 채택",
    "검수: 사람 수정": "사람이 GPT 판정을 고침",
    "기준[5]": "매몰/붕괴 기준 확정 후 일괄 적용",
    "기준[8]": "깔림 기준 보완(SIF 검증 중) 후 적용",
}


def acc(pred: pd.Series, gold: pd.Series) -> str:
    n = (pred == gold).sum()
    return f"{n}/{len(gold)} ({n / len(gold):.1%})"


def keyword_pred(row) -> str:
    kw = row["유형_키워드"]
    return kw.split(",")[0] if isinstance(kw, str) and kw else "기타"


def write_report(df: pd.DataFrame):
    gold = df["유형_검증"]
    kw = df.apply(keyword_pred, axis=1)
    changed = df[df["유형_원본"] != gold].sort_values("엑셀행")
    ct = pd.crosstab(df["유형_원본"], gold, rownames=["원본"], colnames=["검증"])

    md = f"""# 1-3단계 라벨 재검증 리포트

- 생성: `pipeline/apply_review.py`
- 정답 기준: 사람 검수로 확정한 `유형_검증` (분류 기준은 `docs/decisions.md` [4][5])

## 라벨 정확도 (사람 검수 기준, {len(df)}건)

| 방법 | 정확도 | 비고 |
|---|---|---|
| **원본 엑셀 라벨 (V1)** | **{acc(df['유형_원본'], gold)}** | 구간 밀림 오류 포함 |
| 키워드 규칙 | {acc(kw, gold)} | 첫 번째로 걸린 키워드 유형, 없으면 기타 |
| GPT v1 (최초 기준) | {acc(df['GPT_v1'], gold)} | |
| GPT v2 (보완 기준) | {acc(df['GPT_v2'], gold)} | |

→ 원본 라벨 중 **{len(changed)}건({len(changed) / len(df):.1%})이 수정**되었다.
이 라벨을 정답으로 쓰면 V1 평가에서 그만큼 오차가 생긴다.

> ⚠️ GPT 정확도 해석 주의: 검수표에 GPT 판정을 미리 채워 두고 사람이 확인했기 때문에,
> 정답이 GPT 쪽으로 기울었을 수 있다(앵커링 편향). 또 원본과 GPT가 일치한 행은 검수하지 않았다.
> 따라서 GPT 수치는 "상한값"으로 보고, 보고서에는 **원본 라벨 오류율(12%대)**을 주 근거로 쓴다.
> 이후 SIF 데이터 분류 평가는 사람이 GPT 판정을 보지 않고 먼저 라벨링한 표본으로 측정한다.

## 수정 출처

| 출처 | 건수 | 의미 |
|---|---|---|
""" + "\n".join(
        f"| {k} | {v} | {DESC[k]} |" for k, v in df["수정출처"].value_counts().items()
    ) + f"""

## 원본 → 검증 라벨 변화

```
{ct.to_string()}
```

## 연관유형 분포 (기타 {int((gold == '기타').sum())}건의 구체 유형 포함)

| 연관유형 | 건수 |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in df["연관유형"].value_counts().items()) + f"""

## 라벨이 바뀐 사례 ({len(changed)}건)

| 엑셀행 | 원본 | 검증 | 연관 | 출처 | 사고 내용 |
|---|---|---|---|---|---|
""" + "\n".join(
        f"| {r['엑셀행']} | {r['유형_원본']} | {r['유형_검증']} | {r['연관유형']} | {r['수정출처']} | {r['사고내용'][:40]}… |"
        for _, r in changed.iterrows()
    ) + "\n"
    REPORT.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
