"""V3 todo 46-③: KOSHA GUIDE 그림 찾기와 자를 영역 계산 (decisions [26])

입력  data/processed/kosha_guide_text.jsonl      그림 제목 줄이 있는 쪽 찾기 (PDF를 전부 다시 열지 않으려고)
      data/raw/kosha_guide/pdf/<번호>.pdf         그 쪽만 열어 좌표 계산
출력  data/processed/guide_figures.csv            그림 한 줄 = 번호·묶음·그림·제목·쪽·방식·좌표·이유
      data/processed/guide_figures_sample/        --sample N: 무작위 N개를 PNG로 + 판정.csv (👤 눈으로 확인)

자르는 방법 (좌표 단위: PDF 포인트, 원점은 쪽 왼쪽 위 = pdfplumber 기준)
1. 그림 제목 줄 "<그림 N> 제목"의 위치를 찾는다 ("<그림 4>의 …", "<그림 1> 과 같이"처럼 언급만 한 줄은 제외)
2. 쪽의 선·도형·이미지를 모으고(쪽 테두리, 머리글·바닥글 가로줄 제외) 가까운 것끼리 덩어리로 묶는다
3. 제목 줄에 가장 가까운 덩어리 = 그림 (제목이 그림 위든 아래든)
4. 덩어리~제목 줄의 세로 범위 안에 있는 글자는 모두 가로 폭에 넣고(그림 옆·아래 설명 글) + 여백
5. 덩어리가 없거나 멀거나, 영역이 너무 작거나 거의 쪽 전체면 "쪽전체" (쪽 이미지를 그대로 보여 줌)

그릴 해상도 (해상도 열, 백엔드도 같은 값을 씀)
- 기본 BASE_DPI: 고해상도 화면(1pt = 1.33px의 2배 ≈ 190dpi)에서도 선명
- 영역 안 원본 이미지가 더 선명하면 그 해상도로(최대 MAX_DPI). 원본보다 작게 줄여 그리면 가는 획이 사라짐
  (130dpi로 그렸을 때 H-57-2023 그림 1의 작은 글씨가 지워진 듯 보였던 원인)
- 원본dpi 열: 영역 안 원본 이미지 중 가장 낮은 해상도. LOW_DPI 미만이면 원본부터 흐린 그림 (공단 PDF의 한계, 고칠 수 없음)

사용법
  python pipeline/guide_figures.py               # 전체
  python pipeline/guide_figures.py --sample 30   # 전체 + 무작위 30개 PNG (확인용)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/kosha_guide_text.jsonl"
PDF_DIR = ROOT / "data/raw/kosha_guide/pdf"
OUT = ROOT / "data/processed/guide_figures.csv"
SAMPLE_DIR = ROOT / "data/processed/guide_figures_sample"

CAPTION_RE = re.compile(r"^[<〈\[]\s*그림\s*(\d+(?:[-.]\d+)*)\s*[>〉\]]\s+(\S.{0,80})$")
NOT_TITLE_RE = re.compile(r"^(과|와|의|을|를|은|는|에|이|가|참조|참고|같이)(\s|$)")
GAP = 20             # 이 간격(pt) 안의 도형은 같은 덩어리
MIN_AREA = 2000      # 이보다 작은 덩어리는 무시 (점·짧은 밑줄 등)
MAX_DIST = 0.25      # 제목 줄과 덩어리 거리가 쪽 높이의 이 비율보다 멀면 그림이 아님
FULL_RATIO = 0.85    # 잘린 영역이 쪽 넓이의 이 비율보다 크면 자를 의미가 없음 → 쪽전체
PAD = 8
BASE_DPI = 200
MAX_DPI = 300
LOW_DPI = 150
SEED = 42


def find_captions(rec: dict) -> list[dict]:
    """jsonl 텍스트에서 그림 제목 줄 찾기. 같은 그림 번호는 처음 나온 것만"""
    seen, out = set(), []
    for i, page in enumerate(rec["pages"]):
        if "···" in page:  # 목차 쪽
            continue
        for line in page.split("\n"):
            m = CAPTION_RE.match(line.strip())
            if not m or NOT_TITLE_RE.match(m.group(2)) or m.group(2).endswith("다."):
                continue
            if m.group(1) not in seen:
                seen.add(m.group(1))
                out.append({"그림": m.group(1), "제목": m.group(2).strip(), "쪽": i + 1})
    return out


def graphics(page) -> list[tuple]:
    W, H = page.width, page.height
    objs = []
    for o in page.images + page.curves + page.lines + page.rects:
        w, h = o["x1"] - o["x0"], o["bottom"] - o["top"]
        if w > W * 0.9 and h > H * 0.8:
            continue  # 쪽 테두리
        if h < 3 and w > W * 0.7 and (o["top"] < H * 0.12 or o["top"] > H * 0.9):
            continue  # 머리글·바닥글 가로줄
        objs.append((o["x0"], o["top"], o["x1"], o["bottom"]))
    return objs


def blocks(objs: list[tuple]) -> list[tuple]:
    out = []
    for b in sorted(objs, key=lambda b: b[1]):
        if out and b[1] <= out[-1][3] + GAP:
            a = out[-1]
            out[-1] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        else:
            out.append(b)
    return [b for b in out if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_AREA]


def crop_box(page, number: str) -> tuple[str, tuple, str]:
    """(방식, 좌표, 이유). 방식은 "자름" 또는 "쪽전체" """
    W, H = page.width, page.height
    full = (0, 0, W, H)
    pat = re.compile(rf"^[<〈\[]\s*그림\s*{re.escape(number)}\s*[>〉\]]\s+\S")
    lines = page.extract_text_lines()
    cap = next((ln for ln in lines if pat.match(ln["text"].strip())), None)
    if cap is None:
        return "쪽전체", full, "제목 줄 좌표 못 찾음"

    def dist(b):
        return min(abs(b[1] - cap["bottom"]), abs(cap["top"] - b[3]))

    bl = blocks(graphics(page))
    if not bl:
        return "쪽전체", full, "도형 없음"
    best = min(bl, key=dist)
    if dist(best) > H * MAX_DIST:
        return "쪽전체", full, "가까운 도형 없음"

    top = min(best[1], cap["top"]) - PAD
    bottom = max(best[3], cap["bottom"]) + PAD
    # 세로 범위(그림 + 그림과 제목 사이) 안의 글자는 모두 가로 폭에 넣음: 그림 옆 설명(E-G-18 그림 4),
    # 그림 아래 설명(E-G-21 그림 6, 👤 확인에서 오른쪽이 잘려 발견)
    words = [w for w in page.extract_words() if w["top"] >= top + PAD - 2 and w["bottom"] <= bottom - PAD + 2]
    x0 = min([best[0], cap["x0"]] + [w["x0"] for w in words]) - PAD
    x1 = max([best[2], cap["x1"]] + [w["x1"] for w in words]) + PAD
    # 여백 때문에 위·아래 경계에 반쯤 걸친 글자 줄(바로 위 본문, 이웃 그림 제목)은 빼고 그 줄 바깥에서 끊음.
    # 그림에 속한 글자는 이미 영역 안에 다 들어 있어 경계에 걸치지 않음
    for ln in lines:
        if ln is cap:
            continue
        if ln["top"] < top < ln["bottom"]:
            top = ln["bottom"] + 1
        if ln["top"] < bottom < ln["bottom"]:
            bottom = ln["top"] - 1
    box = (max(0, x0), max(0, top), min(W, x1), min(H, bottom))
    if (box[2] - box[0]) * (box[3] - box[1]) > W * H * FULL_RATIO:
        return "쪽전체", full, "영역이 거의 쪽 전체"
    return "자름", box, ""


def render_dpi(page, box: tuple) -> tuple[int, int | None]:
    """(그릴 해상도, 영역 안 원본 이미지의 가장 낮은 해상도 / 이미지가 없으면 None)"""
    native = [
        im["srcsize"][0] / (im["width"] / 72)
        for im in page.images
        if im["width"] > 0 and im["x0"] < box[2] and im["x1"] > box[0] and im["top"] < box[3] and im["bottom"] > box[1]
    ]
    if not native:
        return BASE_DPI, None
    return int(min(MAX_DPI, max(BASE_DPI, max(native)))), int(min(native))


def save_sample(df: pd.DataFrame, n: int):
    """이전 확인용 파일을 지우고 새로 저장. 단, 판정을 적어 둔 판정.csv가 있으면 지우지 않고 멈춤"""
    sheet = SAMPLE_DIR / "판정.csv"
    if sheet.exists():
        old = pd.read_csv(sheet, dtype=str, encoding="utf-8-sig").fillna("")
        filled = [c for c in ("판정", "자르기", "선명", "메모") if c in old and (old[c] != "").any()]
        if filled:
            sys.exit(f"{sheet.relative_to(ROOT)}에 적어 둔 판정이 있음({', '.join(filled)}) → 다른 곳에 옮긴 뒤 다시 실행")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for f in SAMPLE_DIR.glob("*.png"):
        f.unlink()
    pick = df.sample(n=min(n, len(df)), random_state=SEED)
    rows = []
    for r in pick.itertuples(index=False):
        name = f"{r.번호}_그림{r.그림}.png"
        with pdfplumber.open(PDF_DIR / f"{r.번호}.pdf") as pdf:
            page = pdf.pages[r.쪽 - 1]
            page.crop((r.x0, r.top, r.x1, r.bottom)).to_image(resolution=int(r.해상도)).save(SAMPLE_DIR / name)
        rows.append({"파일": name, "번호": r.번호, "그림": r.그림, "제목": r.제목, "쪽": r.쪽,
                     "방식": r.방식, "원본dpi": r.원본dpi, "자르기": "", "선명": "", "메모": ""})
    pd.DataFrame(rows).to_csv(SAMPLE_DIR / "판정.csv", index=False, encoding="utf-8-sig")
    print(f"확인용 {len(rows)}개: {SAMPLE_DIR.relative_to(ROOT)}")
    print("  판정.csv → 자르기: O/X (영역이 맞는지), 선명: O/흐림 (흐리면 원본dpi 열과 비교)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="무작위 N개를 PNG로 저장")
    args = ap.parse_args()

    rows = []
    with SRC.open(encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    for i, rec in enumerate(recs, 1):
        caps = find_captions(rec)
        if caps:
            try:
                with pdfplumber.open(PDF_DIR / f"{rec['번호']}.pdf") as pdf:
                    for c in caps:
                        page = pdf.pages[c["쪽"] - 1]
                        how, box, why = crop_box(page, c["그림"])
                        dpi, native = render_dpi(page, box)
                        rows.append({"번호": rec["번호"], "묶음": rec["묶음"], **c, "방식": how,
                                     "x0": round(box[0], 1), "top": round(box[1], 1),
                                     "x1": round(box[2], 1), "bottom": round(box[3], 1),
                                     "해상도": dpi, "원본dpi": native, "이유": why})
            except Exception as e:
                print(f"  오류 {rec['번호']}: {type(e).__name__}: {str(e)[:150]}")
        if i % 100 == 0 or i == len(recs):
            print(f"  {i}/{len(recs)} (그림 {len(rows)})")

    df = pd.DataFrame(rows)
    df["원본dpi"] = df["원본dpi"].astype("Int64")  # 이미지가 없는 그림은 빈칸 (소수점 없이)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n그림 {len(df):,}개 (지침 {df['번호'].nunique()}개)")
    print(df["방식"].value_counts().to_string())
    print("\n쪽전체 이유")
    print(df.loc[df["방식"] == "쪽전체", "이유"].value_counts().to_string())
    print("\n그릴 해상도")
    print(df["해상도"].describe()[["min", "50%", "max"]].to_string())
    low = df["원본dpi"].notna() & (df["원본dpi"] < LOW_DPI)
    print(f"원본 이미지가 {LOW_DPI}dpi 미만 (원본부터 흐림): {low.sum():,} / 이미지가 든 그림 {df['원본dpi'].notna().sum():,}")
    print("\n묶음별")
    print(df.groupby("묶음")["방식"].value_counts().unstack(fill_value=0).to_string())
    print(f"\n저장: {OUT.relative_to(ROOT)}")
    if args.sample:
        save_sample(df, args.sample)


if __name__ == "__main__":
    main()
