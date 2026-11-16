"""V3 준비: 외부 API 시험 호출 (응답이 오는지, 어떤 필드가 있는지 확인)

수집 코드가 아니라 '확인용'이다. 결과는 화면에만 출력하고 파일로 저장하지 않는다.
인증키는 .env 에서 읽고, 오류 메시지에 키가 섞이면 *** 로 가린다.

사용법
  python collectors/probe_apis.py case      # 국내재해사례 게시판 (공공데이터포털)
  python collectors/probe_apis.py guide     # 기술지원규정(코샤가이드) (공공데이터포털)
  python collectors/probe_apis.py law       # 산업안전보건기준에 관한 규칙 (국가법령정보센터)
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# 활용가이드: docs/api/18. 한국산업안전보건공단_국내재해사례 게시판 조회서비스_오픈API활용가이드.docx
# 가이드에는 http:// 로 적혀 있지만, SIF 처럼 https:// 가 되는지 먼저 시험한다
CASE_URL = "https://apis.data.go.kr/B552468/disaster_api02/getdisaster_api02"
CASE_CALL_API_ID = "1060"

# 활용가이드: docs/api/11. 한국산업안전보건공단_기술지원규정(코샤가이드) 조회서비스_오픈API활용가이드.docx
GUIDE_URL = "https://apis.data.go.kr/B552468/koshaguide/getKoshaGuide"
GUIDE_CALL_API_ID = "1050"

# 국가법령정보센터 OPEN API (공공데이터포털과 별개. 인증키 = 직접 지정한 OC 값, 등록한 IP에서만 호출 가능)
LAW_SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"    # 목록(검색) → 법령 일련번호 찾기
LAW_SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"  # 본문 → 조문 전체
LAW_NAME = "산업안전보건기준에 관한 규칙"


def data_go_kr_key() -> str:
    load_dotenv(ROOT / ".env")
    return os.environ["DATA_GO_KR_KEY"].strip()


def get_json(url: str, params: dict, key: str) -> dict:
    try:
        r = requests.get(url, params=params, timeout=30)
        print(f"HTTP {r.status_code}")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # 요청 URL(=인증키 포함)이 오류 메시지에 들어갈 수 있으므로 가린다
        sys.exit(f"호출 실패: {str(e).replace(key, '***')[:300]}")


def show(data: dict) -> None:
    header = data.get("header", {})
    body = data.get("body", {})
    items = (body.get("items") or {}).get("item") or []
    if isinstance(items, dict):  # 결과가 1건이면 리스트가 아니라 dict 로 오는 API 가 있음
        items = [items]
    print(f"resultCode={header.get('resultCode')} resultMsg={header.get('resultMsg')}")
    print(f"totalCount={body.get('totalCount')}  이번 페이지 {len(items)}건")
    if items:
        print(f"필드: {list(items[0].keys())}")
    for i, it in enumerate(items, 1):
        print(f"\n--- {i} ---")
        for k, v in it.items():
            v = str(v).strip()
            print(f"{k}: {v[:300]}{' …(' + str(len(v)) + '자)' if len(v) > 300 else ''}")


def probe_case() -> None:
    key = data_go_kr_key()
    params = {"serviceKey": key, "callApiId": CASE_CALL_API_ID, "pageNo": "1", "numOfRows": "3"}
    print("== 국내재해사례: 조건 없이 최신 3건 ==")
    show(get_json(CASE_URL, params, key))


def check_download(url: str) -> None:
    """다운로드 링크가 어떤 파일을 주는지 앞부분만 받아서 확인 (전체를 받지 않음)"""
    print(f"\n== 다운로드 링크 확인: {url} ==")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            head = next(r.iter_content(8), b"")
            print(f"HTTP {r.status_code}")
            for h in ("Content-Type", "Content-Length", "Content-Disposition"):
                print(f"{h}: {r.headers.get(h)}")
            print(f"파일 앞 8바이트: {head!r}  (PDF면 b'%PDF-' 로 시작)")
    except Exception as e:
        print(f"다운로드 확인 실패: {str(e)[:300]}")


def probe_guide() -> None:
    key = data_go_kr_key()
    base = {"serviceKey": key, "callApiId": GUIDE_CALL_API_ID, "pageNo": "1", "numOfRows": "3"}

    print("== 코샤가이드: 조건 없이 3건 ==")
    data = get_json(GUIDE_URL, base, key)
    show(data)

    print("\n== 코샤가이드: 이름에 '추락'이 들어간 지침 3건 (검색 조건이 동작하는지) ==")
    show(get_json(GUIDE_URL, {**base, "techGdlnNm": "추락"}, key))

    items = (data.get("body", {}).get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    if items and items[0].get("fileDownloadUrl"):
        check_download(items[0]["fileDownloadUrl"])


def law_get(url: str, params: dict, oc: str) -> dict:
    """법령 API 호출. IP 미등록 등 오류일 때는 JSON 대신 HTML/XML 이 올 수 있어 원문 앞부분을 보여준다"""
    try:
        r = requests.get(url, params=params, timeout=30)
    except Exception as e:
        sys.exit(f"호출 실패: {str(e).replace(oc, '***')[:300]}")
    print(f"HTTP {r.status_code}  Content-Type: {r.headers.get('Content-Type')}")
    try:
        return r.json()
    except ValueError:
        sys.exit(f"JSON 이 아닌 응답 (IP 미등록·OC 오류 가능성):\n{r.text.replace(oc, '***')[:500]}")


def as_list(v) -> list:
    """결과가 1건이면 리스트가 아니라 dict 로 오는 경우가 있어 항상 리스트로 맞춘다"""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def probe_law() -> None:
    load_dotenv(ROOT / ".env")
    oc = os.environ["LAW_OC"].strip()

    print(f"== 법령 목록: '{LAW_NAME}' 검색 ==")
    data = law_get(LAW_SEARCH_URL, {"OC": oc, "target": "law", "type": "JSON", "query": LAW_NAME}, oc)
    search = data.get("LawSearch", data)
    print(f"최상위 키: {list(data.keys())}  /  totalCnt={search.get('totalCnt')}")
    laws = as_list(search.get("law"))
    for law in laws:
        print({k: law.get(k) for k in ("법령명한글", "법령구분명", "법령ID", "법령일련번호", "시행일자", "소관부처명")})

    target = next((x for x in laws if x.get("법령명한글") == LAW_NAME), None)
    if not target:
        sys.exit("정확히 같은 이름의 법령을 찾지 못함 → 위 목록의 필드 이름을 확인")

    print(f"\n== 법령 본문: 법령일련번호(MST)={target.get('법령일련번호')} ==")
    body = law_get(LAW_SERVICE_URL, {"OC": oc, "target": "law", "type": "JSON",
                                     "MST": target.get("법령일련번호")}, oc)
    law_body = body.get("법령", body)
    print(f"최상위 키: {list(body.keys())}  /  법령 아래 키: {list(law_body.keys())}")
    info = law_body.get("기본정보", {})
    print({k: info.get(k) for k in ("법령명_한글", "시행일자", "공포일자", "소관부처")})

    units = as_list((law_body.get("조문") or {}).get("조문단위"))
    articles = [u for u in units if u.get("조문여부") == "조문"]  # '전문'(장·절 제목)과 실제 조문 구분
    print(f"조문단위 {len(units)}개 중 실제 조문 {len(articles)}개")
    if units:
        print(f"조문단위 필드: {list(units[0].keys())}")
    for u in articles[:2]:
        print(f"\n--- 제{u.get('조문번호')}조 {u.get('조문제목')} ---")
        print(str(u.get("조문내용", "")).strip()[:300])
        hang = as_list(u.get("항"))
        print(f"항 {len(hang)}개" + (f", 첫 항 필드: {list(hang[0].keys())}" if hang and isinstance(hang[0], dict) else ""))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    probes = {"case": probe_case, "guide": probe_guide, "law": probe_law}
    if len(sys.argv) != 2 or sys.argv[1] not in probes:
        sys.exit(f"사용법: python collectors/probe_apis.py [{'|'.join(probes)}]")
    probes[sys.argv[1]]()
