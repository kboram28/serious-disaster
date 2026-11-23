"""V3 todo 24-④: KOSHA GUIDE 목록에 묶음 라벨 붙이기 (decisions [25])

입력  data/raw/kosha_guide/list_<날짜>.csv     collectors/kosha_guide_list.py 결과
출력  data/processed/kosha_guide_list.csv      + 묶음 열 (①~④)

묶음은 지침을 빼기 위한 것이 아니라 검색할 때 거를 수 있게 붙이는 표시다.
제외 여부는 지침 검색 기능을 만든 뒤 측정으로 정한다 (측정 전 기본값: ①② 제외).

분야 기호는 옛 기술지침(2010~2023, 한 글자)과 새 기술지원규정(2025~, 두 글자) 두 체계가 있고
같은 글자도 뜻이 달라서(옛 C = 건설, 새 C-C = 화학설비) 기호 전체로 매핑한다.
묶음 이름은 지침 이름을 보고 추정한 것이며 공단 공식 분류표로 확인하지 않았다.

사용법
  python pipeline/label_guides.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw/kosha_guide"
OUT = ROOT / "data/processed/kosha_guide_list.csv"

GROUPS = {
    "①측정분석": ["A", "E-H"],               # 작업환경측정·분석 방법 (실험실 절차)
    "②독성시험": ["T", "E-T"],               # 화학물질 독성 시험법
    "③보건의학": ["H", "E-M", "E-G", "W"],   # 건강진단·직무스트레스·MSDS + 응급처치·중독·밀폐공간·고열 (섞여 있음)
}
DEFAULT = "④안전"                            # 건설·기계·전기·일반·공정안전·위험성평가 등 나머지 전부
CODE_TO_GROUP = {code: g for g, codes in GROUPS.items() for code in codes}


def latest_list() -> Path:
    files = sorted(RAW_DIR.glob("list_*.csv"))
    if not files:
        sys.exit("목록이 없음 → 먼저 python collectors/kosha_guide_list.py 실행")
    return files[-1]  # 파일 이름이 list_YYYYMMDD 라 정렬하면 마지막이 최신


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    src = latest_list()
    df = pd.read_csv(src, dtype=str, encoding="utf-8-sig")
    df["묶음"] = df["분야코드"].map(CODE_TO_GROUP).fillna(DEFAULT)
    df["연도"] = df["번호"].str[-4:]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"입력: {src.relative_to(ROOT)} ({len(df)}건)")
    print(df["묶음"].value_counts().sort_index().to_string())
    print("\n묶음 × 분야코드")
    for g, sub in df.groupby("묶음"):
        print(f"  {g}: " + ", ".join(f"{c} {n}" for c, n in sub["분야코드"].value_counts().items()))
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
