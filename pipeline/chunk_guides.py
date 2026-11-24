"""V3 todo 46-③: KOSHA GUIDE 텍스트를 검색용 조각으로 나누기 (decisions [25])

입력  data/processed/kosha_guide_text.jsonl    pipeline/extract_guides.py 결과 (쪽별 원문)
      data/processed/guide_figures.csv         pipeline/guide_figures.py 결과 (있으면 그림 열을 채움, [26])
출력  data/processed/guide_chunks.csv          조각 한 줄 = 조각ID·번호·이름·묶음·연도·위치·본문·글자수·시작쪽·그림

규칙
1. 앞부분 빼기: "1. 목적"이 처음 나오는 줄부터 본문 (표지·개요·목차 제거)
2. 머리글·쪽 번호: 쪽마다 앞 3줄·뒤 2줄 중 본문 쪽 절반 이상에 똑같이 나오는 줄을 지움
   (머리글 모양이 10가지 넘게 달라서 모양 대신 반복 횟수로 찾음) + 숫자·대시만 있는 줄
3. 제목: 장은 번호가 1씩 늘 때만 인정(본문 속 번호 목록과 구별), 절 N.N·항 N.N.N은 앞 번호가 지금 장과 같을 때만.
   <부록>은 장 하나로 보고 그 안에서는 번호 제목을 찾지 않음 (부록 안 번호가 다시 1부터 시작하므로)
4. 크기: MIN_CHARS 미만은 같은 장 안의 다음(없으면 이전) 조각과 합침, MAX_CHARS 초과는
   (1)·(가) 같은 항목이 시작하는 줄에서 TARGET_CHARS 안팎으로 자름
5. 줄 이어 붙이기: 쪽 너비 때문에 꺾인 줄은 띄어쓰기 한 칸, 항목 기호로 시작하는 줄은 줄바꿈 유지
6. 표·그림 조각 빼기: 한글 비율 HANGUL_MIN 미만
7. 그림 열: 본문에 "<그림 N>"이 나오고 guide_figures.csv에 그 그림이 있으면 번호를 ";"로 이어 기록 (예: "2;5")
   → guide_figures.py를 먼저 실행해야 채워짐
- 검색할 때 넣는 글은 "이름 > 위치\n본문" 으로 만든다 (위치 머리가 있어야 "(1) 작업 시작 전" 같은 조각이 무슨 내용인지 알 수 있음)

사용법
  python pipeline/chunk_guides.py                      # 전체
  python pipeline/chunk_guides.py --show E-G-18-2026   # 전체 저장 후 이 지침의 조각을 화면에 출력 (원문 대조용)
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/kosha_guide_text.jsonl"
FIGURES = ROOT / "data/processed/guide_figures.csv"
OUT = ROOT / "data/processed/guide_chunks.csv"

MIN_CHARS = 100
MAX_CHARS = 800
TARGET_CHARS = 500
HARD_CUT = 1000      # 항목 기호가 오래 안 나오면 이 길이에서 아무 줄에서나 자름
HANGUL_MIN = 0.3

START_RE = re.compile(r"^1\s*\.\s*목\s*적")
PAGE_NO_RE = re.compile(r"^[\s\-－–]*\d+[\s\-－–]*$")
H1_RE = re.compile(r"^(\d{1,2})\s*\.\s*([가-힣].{0,40})$")
H2_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.?\s+([가-힣].{0,50})$")
H3_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{1,2})\.?\s+([가-힣].{0,50})$")
APPENDIX_RE = re.compile(r"^[<〈\[]\s*부\s*록")
ITEM_RE = re.compile(r"^(\(\d+\)|\([가-힣]\)|\d+\)|[가-힣]\.\s|[①-⑳]|[-•○◦·ㆍ])")
HANGUL_RE = re.compile(r"[가-힣]")
FIG_REF_RE = re.compile(r"[<〈\[]\s*그림\s*(\d+(?:[-.]\d+)*)\s*[>〉\]]")


@dataclass
class Section:
    path: list[str]            # [장, 절, 항] 제목
    chapter: int               # 같은 장끼리만 합치기 위한 순번
    lines: list[tuple[str, int]] = field(default_factory=list)  # (줄, PDF 쪽 1부터)

    def chars(self) -> int:
        return sum(len(t) for t, _ in self.lines)


def body_lines(pages: list[str]) -> list[tuple[str, int]]:
    """본문 시작부터, 머리글·쪽 번호를 지운 (줄, 쪽) 목록. 시작을 못 찾으면 빈 목록"""
    split = [[x.strip() for x in p.split("\n") if x.strip()] for p in pages]
    start = next(
        ((i, j) for i, ls in enumerate(split) if "···" not in "".join(ls)
         for j, x in enumerate(ls) if START_RE.match(x)),
        None,
    )
    if start is None:
        return []
    si, sj = start
    body = split[si:]

    cnt = Counter()
    for ls in body:
        cnt.update(set(ls[:3] + ls[-2:]))
    repeated = {x for x, n in cnt.items() if n >= max(2, len(body) * 0.5)}

    out = []
    for k, ls in enumerate(body):
        for j, x in enumerate(ls):
            if k == 0 and j < sj:
                continue
            if x in repeated or PAGE_NO_RE.match(x):
                continue
            out.append((x, si + k + 1))
    return out


def to_sections(lines: list[tuple[str, int]]) -> list[Section]:
    secs: list[Section] = []
    cur = 0            # 지금 장 번호 (부록이면 -1 → 번호 제목을 더 찾지 않음)
    chapter = 0        # 부록까지 포함한 장 순번
    h1 = h2 = ""
    for text, page in lines:
        m1, m2, m3 = H1_RE.match(text), H2_RE.match(text), H3_RE.match(text)
        if APPENDIX_RE.match(text):
            cur, chapter, h1, h2 = -1, chapter + 1, text, ""
            secs.append(Section([h1], chapter))
        elif m1 and int(m1.group(1)) == cur + 1 and not text.endswith("다."):
            cur, chapter, h1, h2 = cur + 1, chapter + 1, text, ""
            secs.append(Section([h1], chapter))
        elif m3 and int(m3.group(1)) == cur and not text.endswith("다."):
            path = [h1, h2, text] if h2.startswith(f"{m3.group(1)}.{m3.group(2)}") else [h1, text]
            secs.append(Section(path, chapter))
        elif m2 and int(m2.group(1)) == cur and not text.endswith("다."):
            h2 = text
            secs.append(Section([h1, h2], chapter))
        elif secs:
            secs[-1].lines.append((text, page))
    return secs


def first_page(*secs: Section) -> int:
    return next((s.lines[0][1] for s in secs if s.lines), 0)


def merge_small(secs: list[Section]) -> list[Section]:
    """MIN_CHARS 미만 조각을 같은 장의 다음 조각 앞에(없으면 이전 조각 뒤에) 붙인다.
    다음 조각이 자식 절이면 제목이 이미 위치에 들어 있으므로 본문만, 형제 절이면 제목 줄도 함께 붙인다"""
    out: list[Section] = []
    carry: Section | None = None
    for s in secs:
        if carry is not None:
            if carry.chapter == s.chapter:
                is_parent = s.path[: len(carry.path)] == carry.path
                head = [] if is_parent else [(carry.path[-1], first_page(carry, s))]
                s.lines = head + carry.lines + s.lines
            else:
                out.append(carry)  # 같은 장에 다음 조각이 없음 → 아래에서 이전 조각과 합치기 시도
            carry = None
        if s.chars() < MIN_CHARS:
            carry = s
        else:
            out.append(s)
    if carry is not None:
        out.append(carry)

    final: list[Section] = []
    for s in out:
        if s.chars() < MIN_CHARS and final and final[-1].chapter == s.chapter:
            prev = final[-1]
            prev.lines += [(s.path[-1], first_page(s, prev))] + s.lines
        elif s.lines:
            final.append(s)
    return final


def split_large(s: Section) -> list[list[tuple[str, int]]]:
    if s.chars() <= MAX_CHARS:
        return [s.lines]
    pieces, buf, n = [], [], 0
    for text, page in s.lines:
        cut = buf and ((n >= TARGET_CHARS and ITEM_RE.match(text)) or n >= HARD_CUT)
        if cut:
            pieces.append(buf)
            buf, n = [], 0
        buf.append((text, page))
        n += len(text)
    if buf:
        if pieces and n < MIN_CHARS:
            pieces[-1] += buf
        else:
            pieces.append(buf)
    return pieces


def join(lines: list[tuple[str, int]]) -> str:
    out = ""
    for text, _ in lines:
        if not out:
            out = text
        elif ITEM_RE.match(text):
            out += "\n" + text
        else:
            out += " " + text
    return out


def hangul_ratio(text: str) -> float:
    chars = re.sub(r"\s", "", text)
    return len(HANGUL_RE.findall(chars)) / len(chars) if chars else 0.0


def figure_refs(text: str, known: set[str]) -> str:
    """본문에 나온 그림 중 guide_figures.csv에 있는 것만, 나온 순서대로 "2;5" """
    refs = dict.fromkeys(n for n in FIG_REF_RE.findall(text) if n in known)
    return ";".join(refs)


def chunk(rec: dict, figures: set[str] = frozenset()) -> tuple[list[dict], int]:
    """(조각 목록, 표·그림이라 버린 수). figures = 이 지침에서 찾은 그림 번호"""
    secs = merge_small(to_sections(body_lines(rec["pages"])))
    rows, dropped = [], 0
    for s in secs:
        for piece in split_large(s):
            text = join(piece)
            if hangul_ratio(text) < HANGUL_MIN:
                dropped += 1
                continue
            rows.append({
                "번호": rec["번호"],
                "이름": rec["이름"],
                "묶음": rec["묶음"],
                "연도": rec["번호"][-4:],
                "위치": " > ".join(p for p in s.path if p),
                "본문": text,
                "글자수": len(text),
                "시작쪽": piece[0][1],
                "그림": figure_refs(text, figures),
            })
    for i, r in enumerate(rows, 1):
        r["조각ID"] = f"{rec['번호']}#{i:03d}"
    return rows, dropped


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default="", help="이 번호의 조각을 화면에 출력")
    args = ap.parse_args()

    figs: dict[str, set[str]] = {}
    if FIGURES.exists():
        fd = pd.read_csv(FIGURES, dtype=str, encoding="utf-8-sig")
        for no, g in fd.groupby("번호"):
            figs[no] = set(g["그림"])
    else:
        print(f"{FIGURES.relative_to(ROOT)} 없음 → 그림 열을 비워 둠 (먼저 python pipeline/guide_figures.py)")

    rows, dropped, empty = [], 0, []
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            r, d = chunk(rec, figs.get(rec["번호"], set()))
            if not r:
                empty.append(rec["번호"])
            rows += r
            dropped += d

    cols = ["조각ID", "번호", "이름", "묶음", "연도", "위치", "본문", "글자수", "시작쪽", "그림"]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    n = df["글자수"]
    q = statistics.quantiles(n, n=10)
    print(f"지침 {df['번호'].nunique()}개 → 조각 {len(df):,}개 (표·그림이라 뺀 조각 {dropped:,})")
    print(f"조각 없음 {len(empty)}: {', '.join(empty) or '-'}")
    print(f"글자수: 중앙값 {n.median():.0f}, 10% {q[0]:.0f}, 90% {q[-1]:.0f}, 최대 {n.max():,}")
    print(f"  {MIN_CHARS}자 미만 {(n < MIN_CHARS).sum():,}, {HARD_CUT}자 초과 {(n > HARD_CUT).sum():,}")
    print(f"그림이 연결된 조각 {(df['그림'] != '').sum():,}")
    print("\n묶음별 (조각 / 글자)")
    print(df.groupby("묶음").agg(조각=("조각ID", "size"), 글자=("글자수", "sum")).to_string())
    print(f"\n저장: {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024**2:.0f}MB)")

    if args.show:
        for r in df[df["번호"] == args.show].itertuples(index=False):
            fig = f" · 그림 {r.그림}" if r.그림 else ""
            print(f"\n[{r.조각ID}] {r.시작쪽}쪽 · {r.글자수}자{fig} · {r.위치}\n{r.본문}")


if __name__ == "__main__":
    main()
