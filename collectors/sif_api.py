"""3단계: SIF(사망사고 고위험요인) 아카이브 API 수집

API 명세: docs/api/sif_api.md
출력
  data/raw/sif/<날짜>/cate<코드>_p<페이지>.json   API 응답 원본 (가공하지 않음)
  data/raw/sif/sif_<날짜>.jsonl                   전체 사례를 한 줄에 하나씩 모은 파일

다시 실행하면 이미 받은 페이지는 건너뛴다. 새로 갱신된 데이터를 받으려면 --date 를 바꿔 실행.

사용법
  python collectors/sif_api.py
"""
import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
URL = "https://apis.data.go.kr/B552468/sifArchive/view"
CALL_API_ID = "1170"
CATEGORIES = {"0": "건설업", "1": "전업종"}
KEYWORD = "%"          # 검색어 필수 → '%' 로 전체 조회
ROWS = 100             # 페이지당 건수
PAUSE = 0.3            # 호출 간격(초). 명세상 최대 30tps보다 훨씬 느리게


def service_key() -> str:
    load_dotenv(ROOT / ".env")
    return os.environ["DATA_GO_KR_KEY"].strip()


def fetch(key: str, cate: str, page: int) -> dict:
    params = {"serviceKey": key, "callApiId": CALL_API_ID, "cateSeCd": cate,
              "disasterOverview": KEYWORD, "pageNo": str(page), "numOfRows": str(ROWS)}
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
            # 오류 메시지에 URL(=키)이 들어갈 수 있으므로 키를 가린다
            msg = str(e).replace(key, "***")
            if i == 3:
                raise RuntimeError(msg) from None
            print(f"  재시도 {i + 1}: {msg[:120]}")
            time.sleep(3 * 2 ** i)


def items_of(data: dict) -> list[dict]:
    items = (data.get("body", {}).get("items") or {}).get("item") or []
    return items if isinstance(items, list) else [items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"), help="수집 폴더 이름(날짜)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    key = service_key()
    raw_dir = ROOT / "data/raw/sif" / args.date
    raw_dir.mkdir(parents=True, exist_ok=True)
    calls = 0

    for cate, name in CATEGORIES.items():
        page, total = 1, None
        while total is None or (page - 1) * ROWS < total:
            path = raw_dir / f"cate{cate}_p{page:03d}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                data = fetch(key, cate, page)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                calls += 1
                time.sleep(PAUSE)
            total = int(data["body"]["totalCount"])
            if page == 1:
                print(f"[{name}] 전체 {total}건, {-(-total // ROWS)}페이지")
            page += 1

    # 원본 페이지 → 한 파일로 합치기 (수집 메타정보 추가)
    out = ROOT / f"data/raw/sif/sif_{args.date}.jsonl"
    n, per = 0, {}
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(raw_dir.glob("cate*_p*.json")):
            cate = p.stem.split("_")[0][4:]
            for it in items_of(json.loads(p.read_text(encoding="utf-8"))):
                it["_cateSeCd"] = cate
                it["_source_file"] = p.name
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
                n += 1
                per[CATEGORIES[cate]] = per.get(CATEGORIES[cate], 0) + 1

    print(f"이번 실행 API 호출 {calls}회")
    print(f"합계 {n}건 {per} → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
