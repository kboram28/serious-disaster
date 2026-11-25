"""V3 todo 46-④: 지침 조각 원문 대조용 표본 뽑기 (👤 PDF와 비교)

입력  data/processed/guide_chunks.csv              pipeline/chunk_guides.py 결과
출력  data/processed/guide_chunks_check/대조.md     읽기용: 조각마다 PDF 파일·쪽, 위치, 본문
      data/processed/guide_chunks_check/대조.csv    판정용: 내용·경계·잡음·위치판정 O/X + 메모

- 기본 검색 대상인 ③④ 묶음에서만 뽑는다 (decisions [25] 기본값: ①② 제외)
- 시드 고정 → 다시 실행해도 같은 조각
- 판정을 적어 둔 대조.csv가 있으면 덮어쓰지 않고 멈춘다

사용법
  python pipeline/sample_chunks.py           # 10개
  python pipeline/sample_chunks.py --n 20
  python pipeline/sample_chunks.py --seed 43   # 2차 대조 (조각 규칙을 고친 뒤 새 표본)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/guide_chunks.csv"
OUT_DIR = ROOT / "data/processed/guide_chunks_check"
GROUPS = ["③보건의학", "④안전"]
SEED = 42
JUDGE = ["내용", "경계", "잡음", "위치판정", "메모"]  # 위치판정: 위치(장·절 경로) 열이 실제 제목과 맞는지


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=SEED, help="차수마다 바꿔 이전 표본과 겹치지 않게 (1차 42)")
    args = ap.parse_args()

    sheet = OUT_DIR / "대조.csv"
    if sheet.exists():
        old = pd.read_csv(sheet, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        if any((old[c] != "").any() for c in JUDGE if c in old):
            sys.exit(f"{sheet.relative_to(ROOT)}에 적어 둔 판정이 있음 → 다른 이름으로 옮긴 뒤 다시 실행")

    df = pd.read_csv(SRC, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    pick = df[df["묶음"].isin(GROUPS)].sample(n=args.n, random_state=args.seed).sort_values("조각ID")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = ["# 지침 조각 원문 대조", "",
          "조각마다 `data/raw/kosha_guide/pdf/<번호>.pdf`의 시작 쪽(PDF 쪽 번호, 인쇄된 쪽 번호와 다름)을 열어 비교하고 대조.csv에 적는다.",
          "조각이 다음 쪽으로 이어질 수 있음.", ""]
    for i, r in enumerate(pick.itertuples(index=False), 1):
        fig = f" · 그림 {r.그림}" if r.그림 else ""
        md += [f"## {i}. {r.조각ID}", "",
               f"- PDF: `{r.번호}.pdf` {r.시작쪽}쪽 · {r.글자수}자{fig}",
               f"- 지침: {r.이름} ({r.묶음})",
               f"- 위치: {r.위치}", "", "```text", r.본문, "```", ""]
    (OUT_DIR / "대조.md").write_text("\n".join(md), encoding="utf-8")

    rows = pick[["조각ID", "번호", "시작쪽", "위치"]].copy()
    for c in JUDGE:
        rows[c] = ""
    rows.to_csv(sheet, index=False, encoding="utf-8-sig")
    print(f"표본 {len(pick)}개 ({', '.join(GROUPS)}): {OUT_DIR.relative_to(ROOT)}")
    print("  대조.md 읽기 → 대조.csv에 내용·경계·잡음·위치판정 O/X + 메모")


if __name__ == "__main__":
    main()
