"""V3 todo 24-①: KOSHA GUIDE(기술지원규정) 전체 목록 수집

API: 기술지원규정(코샤가이드) 조회 서비스 (공공데이터포털, 확인 결과 docs/api/v3_api_check.md 2장)
이름 검색은 '시작 일치'로 추정되어 검색에 의존하지 않고 전체 목록을 받는다. 본문(PDF)은 받지 않는다.

출력
  data/raw/kosha_guide/<날짜>/p<페이지>.json   API 응답 원본 (가공하지 않음)
  data/raw/kosha_guide/list_<날짜>.csv        전체 목록: 번호, 이름, 공표일, PDF 링크, 분야코드

확인
  - 받은 건수 = API가 알려준 전체 건수(totalCount)
  - 지침 번호가 겹치지 않음

다시 실행하면 이미 받은 페이지는 건너뛴다. 새 목록을 받으려면 --date 를 바꿔 실행.
인증키는 .env 의 DATA_GO_KR_KEY 에서 읽기만 하고, 오류 메시지에 섞이면 *** 로 가린다.

사용법
  python collectors/kosha_guide_list.py
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
URL = "https://apis.data.go.kr/B552468/koshaguide/getKoshaGuide"
CALL_API_ID = "1050"
ROWS = 100     # 페이지당 건수 → 1,039건이면 11회 호출
PAUSE = 0.3    # 호출 간격(초)
FIELDS = ["techGdlnNo", "techGdlnNm", "techGdlnOfancYmd", "fileDownloadUrl"]


def service_key() -> str:
    load_dotenv(ROOT / ".env")
    return os.environ["DATA_GO_KR_KEY"].strip()


def fetch(key: str, page: int) -> dict:
    params = {"serviceKey": key, "callApiId": CALL_API_ID, "pageNo": str(page), "numOfRows": str(ROWS)}
    for i in range(4):
        try:
            r = requests.get(URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            code = data.get("header", {}).get("resultCode", "00")
            if code != "00":
                raise RuntimeError(f"API 오류 {code}: {data['header'].get('resultMsg')}")
            return data
        except Exception as e:
            msg = str(e).replace(key, "***")  # 오류 메시지에 URL(=키)이 들어갈 수 있음
            if i == 3:
                raise RuntimeError(msg) from None
            print(f"  재시도 {i + 1}: {msg[:120]}")
            time.sleep(3 * 2**i)


def items_of(data: dict) -> list[dict]:
    items = (data.get("body", {}).get("items") or {}).get("item") or []
    return [items] if isinstance(items, dict) else items  # 1건이면 dict 로 오는 경우 대비


def field_code(no: str) -> str:
    """지침 번호에서 분야 코드: 'A-1-2018' → 'A', 'A-G-1-2025' → 'A-G' (숫자 일련번호 앞까지)"""
    m = re.match(r"^(.*?)-\d+-\d{4}$", no.strip())
    return m.group(1) if m else ""


def get_page(key: str, day_dir: Path, page: int) -> dict:
    path = day_dir / f"p{page}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch(key, page)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    time.sleep(PAUSE)
    return data


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"), help="저장 폴더 이름 (수집일)")
    args = ap.parse_args()

    key = service_key()
    out_dir = ROOT / "data/raw/kosha_guide"
    day_dir = out_dir / args.date
    day_dir.mkdir(parents=True, exist_ok=True)

    first = get_page(key, day_dir, 1)
    total = int(first["body"]["totalCount"])
    pages = math.ceil(total / ROWS)
    print(f"전체 {total}건 → {pages}페이지")

    rows = []
    for page in range(1, pages + 1):
        items = items_of(get_page(key, day_dir, page))
        rows += items
        print(f"  p{page}: {len(items)}건 (누적 {len(rows)})")

    # 확인 1: 건수
    assert len(rows) == total, f"받은 건수 {len(rows)} ≠ 전체 {total} → 페이지 파일을 지우고 다시 실행"
    # 확인 2: 번호 중복
    nos = [str(r.get("techGdlnNo", "")).strip() for r in rows]
    dup = sorted({n for n in nos if nos.count(n) > 1})
    if dup:
        print(f"⚠️ 겹치는 지침 번호 {len(dup)}개: {dup[:10]}")

    csv_path = out_dir / f"list_{args.date}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: 엑셀에서 한글이 깨지지 않게
        w = csv.writer(f)
        w.writerow(["번호", "이름", "공표일", "PDF링크", "분야코드"])
        for r in rows:
            vals = [str(r.get(k, "")).strip() for k in FIELDS]
            w.writerow(vals + [field_code(vals[0])])

    no_code = sum(1 for n in nos if not field_code(n))
    print(f"\n저장: {csv_path.relative_to(ROOT)} ({len(rows)}건)")
    print(f"번호 중복 {len(dup)}개, 분야코드를 못 뽑은 번호 {no_code}개")


if __name__ == "__main__":
    main()
