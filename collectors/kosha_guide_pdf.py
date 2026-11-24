"""V3 todo 46-①: KOSHA GUIDE PDF 다운로드 (decisions [25]: 1,039건 전부)

입력  data/processed/kosha_guide_list.csv         번호·PDF링크 (pipeline/label_guides.py 결과)
출력  data/raw/kosha_guide/pdf/<지침번호>.pdf       원본 PDF (응답 헤더의 한글 파일 이름이 깨져 번호로 저장)
      data/raw/kosha_guide/pdf_log.csv            번호별 결과: ok / fail, 크기, 실패 이유

- 다운로드 링크는 인증이 필요 없다 (공공데이터포털 키를 쓰지 않음)
- 받는 중에는 .part 로 저장하고 다 받은 뒤 이름을 바꾼다 → 중간에 끊긴 파일을 완료로 착각하지 않음
- 파일 앞부분이 %PDF 가 아니면 실패로 기록 (오류 페이지가 HTML 로 오는 경우 대비)
- 다시 실행하면 이미 받은 파일은 건너뛰고 실패한 것만 다시 받는다

사용법
  python collectors/kosha_guide_pdf.py            # 전체
  python collectors/kosha_guide_pdf.py --limit 5  # 앞에서 5개만 (시험용)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "data/processed/kosha_guide_list.csv"
PDF_DIR = ROOT / "data/raw/kosha_guide/pdf"
LOG = ROOT / "data/raw/kosha_guide/pdf_log.csv"
PAUSE = 0.5    # 파일 사이 간격(초). 공단 서버에 부담을 주지 않도록
TIMEOUT = 60


def download(url: str, path: Path) -> tuple[bool, int, str]:
    """(성공 여부, 바이트 수, 실패 이유)"""
    part = path.with_suffix(".pdf.part")
    for i in range(3):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with part.open("wb") as f:
                    for chunk in r.iter_content(64 * 1024):
                        f.write(chunk)
            head = part.read_bytes()[:5]
            if head != b"%PDF-":
                part.unlink(missing_ok=True)
                return False, 0, f"PDF가 아님 (앞부분 {head!r})"
            part.replace(path)
            return True, path.stat().st_size, ""
        except Exception as e:
            part.unlink(missing_ok=True)
            if i == 2:
                return False, 0, f"{type(e).__name__}: {str(e)[:150]}"
            time.sleep(3 * 2**i)
    return False, 0, "알 수 없음"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (0 = 전체)")
    args = ap.parse_args()

    df = pd.read_csv(LIST, dtype=str, encoding="utf-8-sig")
    if args.limit:
        df = df.head(args.limit)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    log, new, skip = [], 0, 0
    for i, r in enumerate(df.itertuples(index=False), 1):
        no = r.번호.strip()
        path = PDF_DIR / f"{no}.pdf"
        if path.exists():
            ok, size, why = True, path.stat().st_size, ""
            skip += 1
        else:
            ok, size, why = download(r.PDF링크, path)
            new += 1
            time.sleep(PAUSE)
        log.append({"번호": no, "결과": "ok" if ok else "fail", "바이트": size, "실패이유": why})
        if not ok:
            print(f"  실패 {no}: {why}")
        if i % 50 == 0 or i == len(df):
            print(f"  {i}/{len(df)} (새로 받음 {new}, 건너뜀 {skip})")

    with LOG.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["번호", "결과", "바이트", "실패이유"])
        w.writeheader()
        w.writerows(log)

    ok_rows = [x for x in log if x["결과"] == "ok"]
    total_mb = sum(x["바이트"] for x in ok_rows) / 1024**2
    print(f"\n성공 {len(ok_rows)} / 실패 {len(log) - len(ok_rows)}, 전체 크기 {total_mb:.0f}MB")
    print(f"기록: {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
