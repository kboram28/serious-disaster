"""V3 todo 46-②: KOSHA GUIDE PDF에서 텍스트 추출 (decisions [25])

입력  data/raw/kosha_guide/pdf/<번호>.pdf             collectors/kosha_guide_pdf.py 결과
      data/processed/kosha_guide_list.csv              번호·이름·묶음
출력  data/processed/kosha_guide_text.jsonl           지침 한 줄 = {번호, 묶음, 이름, pages: [쪽별 원문]}
      data/processed/kosha_guide_text_stats.csv       지침별 쪽수·글자수·상태

- 다듬지 않고 쪽별 원문 그대로 저장한다. 머리글·쪽 번호·목차 제거와 줄 이어 붙이기는
  조각 나누기(46-③)에서 한다 → 추출(수십 분)을 다시 돌리지 않고 다듬는 방식을 여러 번 고칠 수 있음
- 지침 하나가 끝날 때마다 jsonl에 한 줄씩 붙인다 → 다시 실행하면 이미 있는 번호는 건너뜀
- 통계 csv는 마지막에 jsonl 전체에서 다시 계산한다
- 본문(표지·개요 2쪽을 뺀 나머지) 글자가 SCAN_CHARS 미만이면 "스캔의심" (이미지로만 된 PDF)

사용법
  python pipeline/extract_guides.py            # 전체
  python pipeline/extract_guides.py --limit 5  # 앞에서 5개만 (시험용)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "data/processed/kosha_guide_list.csv"
PDF_DIR = ROOT / "data/raw/kosha_guide/pdf"
OUT = ROOT / "data/processed/kosha_guide_text.jsonl"
STATS = ROOT / "data/processed/kosha_guide_text_stats.csv"
FRONT_PAGES = 2      # 표지 + 개요
EMPTY_PAGE = 20      # 이 글자 수 미만인 쪽은 빈 쪽
SCAN_CHARS = 300     # 본문 글자가 이보다 적으면 스캔 의심


def extract(path: Path) -> list[str]:
    with pdfplumber.open(path) as pdf:
        return [p.extract_text() or "" for p in pdf.pages]


def stats_row(rec: dict) -> dict:
    pages = rec["pages"]
    body = pages[FRONT_PAGES:] if len(pages) > FRONT_PAGES else pages
    body_chars = sum(len(p) for p in body)
    if rec.get("오류"):
        status = "오류"
    elif body_chars < SCAN_CHARS:
        status = "스캔의심"
    else:
        status = "ok"
    return {
        "번호": rec["번호"],
        "묶음": rec["묶음"],
        "쪽수": len(pages),
        "글자수": sum(len(p) for p in pages),
        "본문글자수": body_chars,
        "빈쪽수": sum(len(p.strip()) < EMPTY_PAGE for p in body),
        "상태": status,
        "오류": rec.get("오류", ""),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    logging.getLogger("pdfminer").setLevel(logging.ERROR)  # 글꼴·CropBox 경고가 화면을 덮지 않도록
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (0 = 전체)")
    args = ap.parse_args()

    df = pd.read_csv(LIST, dtype=str, encoding="utf-8-sig")
    if args.limit:
        df = df.head(args.limit)

    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {json.loads(line)["번호"] for line in f if line.strip()}

    new, skip, missing = 0, 0, []
    with OUT.open("a", encoding="utf-8") as f:
        for i, r in enumerate(df.itertuples(index=False), 1):
            no = r.번호.strip()
            path = PDF_DIR / f"{no}.pdf"
            if no in done:
                skip += 1
            elif not path.exists():
                missing.append(no)  # 다운로드 실패분 (pdf_log.csv 참고)
            else:
                rec = {"번호": no, "묶음": r.묶음, "이름": r.이름, "pages": []}
                try:
                    rec["pages"] = extract(path)
                except Exception as e:
                    rec["오류"] = f"{type(e).__name__}: {str(e)[:150]}"
                    print(f"  오류 {no}: {rec['오류']}")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                new += 1
            if i % 50 == 0 or i == len(df):
                print(f"  {i}/{len(df)} (새로 추출 {new}, 건너뜀 {skip}, PDF 없음 {len(missing)})")

    # 통계: jsonl 전체에서 다시 계산 (이어서 실행한 경우도 전체 기준)
    with OUT.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    st = pd.DataFrame([stats_row(x) for x in recs])
    st.to_csv(STATS, index=False, encoding="utf-8-sig")

    print(f"\nPDF 없음 {len(missing)}: {', '.join(missing) or '-'}")
    print(f"지침 {len(st)}개, 쪽 {st['쪽수'].sum():,}, 글자 {st['글자수'].sum():,}")
    print("\n상태")
    print(st["상태"].value_counts().to_string())
    print("\n묶음별 (지침 수 / 본문 글자 / 평균 쪽수)")
    g = st.groupby("묶음").agg(지침=("번호", "size"), 본문글자=("본문글자수", "sum"), 평균쪽수=("쪽수", "mean"))
    print(g.round(1).to_string())
    bad = st[st["상태"] != "ok"]
    if len(bad):
        print("\n확인 필요")
        print(bad[["번호", "묶음", "쪽수", "본문글자수", "상태", "오류"]].to_string(index=False))
    print(f"\n저장: {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024**2:.0f}MB), {STATS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
