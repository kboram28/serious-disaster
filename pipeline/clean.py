"""1단계: 기존 엑셀(공단 재해사례 정리본) 정제

입력  data/raw/serious disaster accident_1.xlsx
출력  data/processed/cases_clean.csv
      docs/reports/01_clean_report.md

원본 값은 *_원본 열로 남기고, 고친 값은 별도 열에 둔다.
"""
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/serious disaster accident_1.xlsx"
OUT = ROOT / "data/processed/cases_clean.csv"
REPORT = ROOT / "docs/reports/01_clean_report.md"

# 시도 표준명: 짧은 이름으로 통일
# (강원·전북·제주는 특별자치도로 명칭이 바뀌어 긴 이름이 시기마다 다르므로 짧은 이름을 씀)
SIDO_MAP = {
    "서울": "서울", "서울시": "서울", "서울특별시": "서울",
    "부산": "부산", "부산시": "부산", "부산광역시": "부산",
    "대구": "대구", "대구시": "대구", "대구광역시": "대구",
    "인천": "인천", "인천시": "인천", "인천광역시": "인천",
    "광주": "광주", "광주시": "광주", "광주광역시": "광주",
    "대전": "대전", "대전시": "대전", "대전광역시": "대전",
    "울산": "울산", "울산시": "울산", "울산광역시": "울산",
    "세종": "세종", "세종시": "세종",
    "경기": "경기", "경기도": "경기",
    "강원": "강원", "강원도": "강원",
    "충북": "충북", "충청북도": "충북", "청북": "충북",
    "충남": "충남", "충청남도": "충남", "충천남도": "충남",
    "전북": "전북", "전라북도": "전북",
    "전남": "전남", "전라남도": "전남",
    "경북": "경북", "경상북도": "경북",
    "경남": "경남", "경상남도": "경남",
    "제주": "제주", "제주도": "제주",
}

# 지역 문자열 전체를 바로잡는 규칙 (공백 정규화 후 기준)
REGION_FIX = {
    "서울시 중락구": ("서울", "중랑구"),     # 오타
    "경기도 안상시": ("경기", "안성시"),     # 오타 (안산/안성 중 표기상 안성으로 판단)
    "경기도 연천시": ("경기", "연천군"),     # 연천은 군
    "경기도 삼척시": ("강원", "삼척시"),     # 삼척은 강원
    "충남 서상시": ("충남", "서산시"),       # 오타
    "울산군": ("울산", ""),                 # 울산광역시 오기, 구·군 불명
    "제주시": ("제주", "제주시"),
    "세종시": ("세종", ""),
    "논산시": ("충남", "논산시"),
    "김제시": ("전북", "김제시"),
}

# 업종 칸에 재해유형이 들어간 행 → 사고 내용 확인 후 수정
INDUSTRY_FIX = {"떨어짐": "건설업"}

# 재해유형 키워드 (사고 내용에서 라벨을 검증하는 1차 규칙)
TYPE_KEYWORDS = {
    "떨어짐": ["떨어져", "떨어짐", "추락"],
    "끼임": ["끼여", "끼어", "끼임", "말려"],
    "깔림": ["깔려", "깔림", "매몰"],
    "맞음": ["맞아", "맞음", "타격", "가격"],
    "부딪힘": ["부딪", "충돌"],
}

# 사고 내용 오타 (검색용 텍스트에만 적용, 원문은 보존)
TEXT_FIX = {
    "23n ": "23m ", "밟운": "밟은", "무넞지며": "무너지며", "유로품": "유로폼",
    "재해재가": "재해자가", "치료 증": "치료 중", "병워네서": "병원에서",
    "생할시설": "생활시설", "생황주택": "생활주택", "설피 공사": "설치 공사",
    "작억": "작업", "천골기": "천공기", "정빅공사": "정비공사", "첯골": "철골",
    "철자제": "철골자재", "철골자제": "철골자재", "연겨하던": "연결하던",
    "천잘": "천장", "공자 ": "공장 ", "현자에서": "현장에서", "공상 현장": "공사 현장",
}


def norm_space(s):
    return " ".join(str(s).split()) if pd.notna(s) else ""


def split_region(raw: str):
    r = norm_space(raw)
    if r in REGION_FIX:
        return REGION_FIX[r]
    parts = r.split(" ", 1)
    sido = SIDO_MAP.get(parts[0])
    if sido is None:
        raise ValueError(f"알 수 없는 시도: {raw!r}")
    return sido, parts[1] if len(parts) > 1 else ""


def keyword_types(text: str):
    return [t for t, kws in TYPE_KEYWORDS.items() if any(k in text for k in kws)]


def fix_text(text: str):
    for wrong, right in TEXT_FIX.items():
        text = text.replace(wrong, right)
    return text


def main():
    raw = pd.read_excel(RAW)
    df = raw.copy()
    df.columns = [c.replace(" ", "") for c in df.columns]  # '재해 유형' → '재해유형'
    df["엑셀행"] = df.index + 2

    # 번호: 빈 번호는 최대값 다음부터 순서대로
    missing = df["번호"].isna()
    start = int(df["번호"].max()) + 1
    df.loc[missing, "번호"] = range(start, start + missing.sum())
    df["번호"] = df["번호"].astype(int)
    df["번호_부여"] = missing

    # 문자열 공백 정리
    for c in ["재해유형", "업종", "지역", "사고내용", "예방대책"]:
        df[c] = df[c].map(norm_space)

    # 업종
    df["업종_원본"] = df["업종"]
    df["업종"] = df["업종"].replace(INDUSTRY_FIX)

    # 지역 → 시도 / 시군구
    df["지역_원본"] = df["지역"]
    df[["시도", "시군구"]] = df["지역"].apply(lambda r: pd.Series(split_region(r)))
    df["지역"] = (df["시도"] + " " + df["시군구"]).str.strip()

    # 검색용 사고 내용 (오타 교정)
    df["사고내용_교정"] = df["사고내용"].map(fix_text)

    # 재해유형: 원본 보존 + 키워드 검증
    df["유형_원본"] = df["재해유형"]
    df["유형_키워드"] = df["사고내용"].map(lambda t: ",".join(keyword_types(t)))
    df["라벨의심"] = df.apply(
        lambda r: bool(r["유형_키워드"]) and r["유형_원본"] not in r["유형_키워드"].split(","),
        axis=1,
    )

    # 사고 내용 완전 중복 제거 (앞의 것 유지)
    dup = df[df.duplicated("사고내용", keep="first")]
    df = df.drop(dup.index)

    cols = ["번호", "번호_부여", "엑셀행", "유형_원본", "유형_키워드", "라벨의심",
            "업종", "업종_원본", "시도", "시군구", "지역", "지역_원본",
            "사고내용", "사고내용_교정", "예방대책"]
    df = df[cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")  # 엑셀에서 한글이 깨지지 않도록 BOM 포함

    write_report(raw, df, dup)
    print(f"저장: {OUT.relative_to(ROOT)} ({len(df)}행)")
    print(f"리포트: {REPORT.relative_to(ROOT)}")


def table(counter: Counter, head=("값", "건수")):
    lines = [f"| {head[0]} | {head[1]} |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in counter.most_common()]
    return "\n".join(lines)


def write_report(raw: pd.DataFrame, df: pd.DataFrame, dup: pd.DataFrame):
    raw_sido = Counter(norm_space(r).split(" ")[0] for r in raw["지역"])
    raw_ind = Counter(repr(x) for x in raw["업종"])
    region_changed = df[df["지역_원본"].map(norm_space) != df["지역"]]
    fixed_regions = df[df["지역_원본"].map(norm_space).isin(REGION_FIX)]
    suspect = df[df["라벨의심"]]
    text_fixed = (df["사고내용"] != df["사고내용_교정"]).sum()

    md = f"""# 1단계 정제 리포트

- 입력: `{RAW.relative_to(ROOT).as_posix()}`
- 출력: `{OUT.relative_to(ROOT).as_posix()}`
- 생성: `pipeline/clean.py`

## 요약

| 항목 | 정제 전 | 정제 후 |
|---|---|---|
| 행 수 | {len(raw)} | {len(df)} |
| 번호 없는 행 | {raw['번호'].isna().sum()} | 0 |
| 업종 표기 종류 | {len(raw_ind)} | {df['업종'].nunique()} |
| 시도 표기 종류 | {len(raw_sido)} | {df['시도'].nunique()} |
| 사고 내용 완전 중복 | {len(dup)}쌍 | 0 |
| 오타 교정된 사고 내용 | - | {text_fixed}건 (`사고내용_교정` 열) |
| **재해유형 라벨 의심** | - | **{len(suspect)}건** (`라벨의심` 열) |

## 원본 데이터의 한계 (V1 기준)

| 항목 | 상태 |
|---|---|
| 사고 원인 | 없음 (0%) |
| 기인물 | 없음 (사고 내용 문장 안에만 있음) |
| 근거 법규·지침 | 없음 (0%) |
| 사고 내용 길이 | 평균 {df['사고내용'].str.len().mean():.0f}자 (한두 문장) |
| 예방대책 중복 | {df['예방대책'].duplicated().sum()}건이 다른 사례와 같은 문장 |

### 업종 분포 (정제 후)
{table(Counter(df['업종']), ('업종', '건수'))}

### 재해유형 분포 (원본 라벨)
{table(Counter(df['유형_원본']), ('유형', '건수'))}

### 시도 분포 (정제 후)
{table(Counter(df['시도']), ('시도', '건수'))}

## 고친 내용

### 업종
- 끝 공백 제거: {(raw['업종'].astype(str) != raw['업종'].astype(str).str.strip()).sum()}건 (`'건설업 '` → `'건설업'`)
- 업종 칸에 재해유형이 들어간 행: {', '.join(f'{k} → {v}' for k, v in INDUSTRY_FIX.items())}

### 지역
- 시도 표기를 17개 짧은 이름으로 통일 (`경기도`·`경기` → `경기`, `서울시`·`서울` → `서울` 등)
- 지역 문자열이 바뀐 행: {len(region_changed)}건
- 오타·사실 오류 수정:

| 원본 | 수정 |
|---|---|
""" + "\n".join(f"| {k} | {' '.join(x for x in v if x)} |" for k, v in REGION_FIX.items()) + f"""

수정 대상 행 수: {len(fixed_regions)}건

### 중복 제거
""" + "\n".join(f"- 엑셀 {r['엑셀행']}행 (번호 {r['번호']}, 라벨 `{r['유형_원본']}`) 제거: {r['사고내용'][:40]}…" for _, r in dup.iterrows()) + f"""

## ⚠️ 재해유형 라벨 의심 ({len(suspect)}건)

사고 내용의 핵심 단어와 원본 라벨이 맞지 않는 행이다. 특정 구간에 몰려 있어,
엑셀 작성 시 라벨 열(재해유형·업종·지역)과 사고 내용 열이 구간별로 밀린 것으로 보인다.
사고 내용과 예방대책은 서로 일치한다.

→ 다음 단계에서 GPT 재판정 + 사람 검수로 `유형_검증` 열을 만들고,
  업종·지역은 공단 재해사례 API 원문과 대조해 복원한다.

| 엑셀행 | 번호 | 원본 라벨 | 키워드 판정 | 사고 내용 |
|---|---|---|---|---|
""" + "\n".join(
        f"| {r['엑셀행']} | {r['번호']} | {r['유형_원본']} | {r['유형_키워드']} | {r['사고내용'][:45]}… |"
        for _, r in suspect.iterrows()
    ) + "\n"

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
