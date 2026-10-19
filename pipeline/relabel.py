"""1-2단계: 재해유형 라벨 재검증 (GPT 판정 → 사람 검수)

입력  data/processed/cases_clean.csv
출력  data/processed/relabel_gpt.jsonl      GPT 판정 결과 캐시 (다시 실행해도 재호출하지 않음)
      data/processed/relabel_review.csv     사람이 검수할 표 (최종_유형 열을 확인·수정)

사용법
  python pipeline/relabel.py --limit 3     # 3건만 시험
  python pipeline/relabel.py               # 전체
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/cases_clean.csv"
REVIEW = ROOT / "data/processed/relabel_review.xlsx"

MODEL = "gpt-5.4-mini"
# 분류 기준(GUIDE)을 바꾸면 버전을 올린다 → 새 캐시 파일에 다시 판정 (이전 결과는 기록으로 보존)
# v1: 최초 기준 / v2: 쓰러짐→깔림, 복합 시 가장 큰 원인, 감전·익사 등 추측 금지
# v3: 매몰↔깔림 구분, 붕괴+추락→떨어짐, 세부유형=연관유형 (사람 검수로 확정한 기준. 엑셀 509건에는 미실행,
#     이후 SIF 등 새 데이터 판정에 사용)
PROMPT_VERSION = "v3"
CACHE = ROOT / f"data/processed/relabel_gpt_{PROMPT_VERSION}.jsonl"
TYPES = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘", "기타"]
SUBTYPES = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘", "무너짐", "넘어짐",
            "감전", "폭발·파열", "화재", "화상·이상온도", "질식·중독", "익사", "찔림", "기타"]

GUIDE = """너는 산업재해 사례를 재해유형으로 분류하는 안전보건 전문가다.
사고 내용을 읽고 '사람이 무엇에 의해 다치거나 사망했는가(결과)'를 기준으로 분류한다.
물체에 일어난 일(쓰러짐, 붕괴 등)이 아니라 사람이 입은 피해의 방식으로 판단한다.

[재해유형 6종]
- 떨어짐: 사람이 높은 곳(비계, 지붕, 사다리, 개구부, 고소작업대 등)에서 떨어짐
- 끼임: 기계·설비·장비의 움직이는 부분이나 물체 사이에 신체가 끼이거나 말려 들어감
- 깔림: 물체·구조물·장비가 쓰러지거나 넘어져 사람이 그 아래에 깔림, 차량 바퀴에 깔림
- 맞음: 날아오거나 위에서 떨어지는 물체(인양물, 낙하물, 파편 등)에 사람이 맞음
- 부딪힘: 움직이는 차량·장비에 사람이 부딪히거나, 이동 중인 사람이 물체에 부딪힘
- 기타: 위 5종이 아닌 경우 (토사·굴착면·건물 붕괴로 인한 매몰, 감전, 폭발, 화재, 화상, 질식, 익사, 넘어짐 등)

[판단 규칙]
- 사람이 떨어진 경우는 떨어짐. 물체가 위에서 떨어져 사람에게 닿은 경우는 맞음.
- 벽체·옹벽·적재물·철골·장비 등이 쓰러져(넘어져) 사람이 깔린 경우는 깔림이다.
  쓰러진 것은 물체이고, 사람은 그 결과 깔려서 피해를 입었기 때문이다.
- 토사·굴착면·법면이 무너지거나 건물·거푸집 전체가 붕괴하여 사람이 매몰된 경우는 기타(세부유형: 무너짐).
  매몰은 깔림과 다르다(질식 위험, 흙막이 등 예방대책이 다름). 토사에 '깔려'라고 적혀 있어도 매몰로 본다.
- 구조물이 붕괴하면서 사람이 함께 떨어진 경우는 떨어짐(세부유형: 무너짐).
- 세부유형은 결과에 함께 관여한 원인(계기·후속)을 적는다. 기타일 때는 구체 유형(감전 등)을 적는다.
  따로 없으면 재해유형과 같은 값을 적는다.
- 두 가지 이상이 함께 나오면, 사람이 다치거나 사망한 가장 큰 원인을 고른다.
  예) 감전된 뒤 사다리에서 떨어졌다고 명시 → 사망의 주원인을 판단하고, 근거에 두 가지를 모두 적는다.
- 문장에 없는 내용은 추측하지 않는다.
  예) "사다리에 올라 작업 중 감전되어 사망" → 떨어졌다는 말이 없으므로 기타(세부유형: 감전)

세부유형은 재해유형을 더 구체적으로 적는다(기타일 때 특히 중요).
근거에는 판단에 쓴 사고 내용의 핵심 구절을 짧게 인용한다."""

SCHEMA = {
    "type": "object",
    "properties": {
        "유형": {"type": "string", "enum": TYPES},
        "세부유형": {"type": "string", "enum": SUBTYPES},
        "근거": {"type": "string"},
        "확신도": {"type": "string", "enum": ["높음", "보통", "낮음"]},
    },
    "required": ["유형", "세부유형", "근거", "확신도"],
    "additionalProperties": False,
}


def judge(client: OpenAI, text: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": GUIDE},
            {"role": "user", "content": f"사고 내용: {text}"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "label", "schema": SCHEMA, "strict": True},
        },
    )
    out = json.loads(resp.choices[0].message.content)
    out["_tokens"] = {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}
    return out


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["번호"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="앞에서부터 N건만 처리 (시험용)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())

    df = pd.read_csv(SRC)
    if args.limit:
        df = df.head(args.limit)

    cache = load_cache()
    todo = df[~df["번호"].isin(cache)]
    print(f"전체 {len(df)}건 / 캐시 {len(df) - len(todo)}건 / 새로 판정 {len(todo)}건 (모델 {MODEL})")

    def work(row):
        r = judge(client, row["사고내용_교정"])
        r["번호"] = int(row["번호"])
        return r

    with ThreadPoolExecutor(max_workers=5) as pool, CACHE.open("a", encoding="utf-8") as f:
        for i, r in enumerate(pool.map(work, [row for _, row in todo.iterrows()]), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            cache[r["번호"]] = r
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")

    tok_in = sum(cache[n]["_tokens"]["in"] for n in df["번호"])
    tok_out = sum(cache[n]["_tokens"]["out"] for n in df["번호"])
    print(f"사용 토큰: 입력 {tok_in:,} / 출력 {tok_out:,}")

    write_review(df, cache)


def write_review(df: pd.DataFrame, cache: dict):
    g = pd.DataFrame([cache[n] for n in df["번호"]]).drop(columns="_tokens")
    g = g.rename(columns={"유형": "GPT_유형", "세부유형": "GPT_세부유형", "근거": "GPT_근거", "확신도": "GPT_확신도"})
    m = df.merge(g, on="번호")

    m["불일치"] = m["유형_원본"] != m["GPT_유형"]
    m["검수필요"] = m["불일치"] | m["라벨의심"] | (m["GPT_확신도"] != "높음")
    m["최종_유형"] = m["GPT_유형"]          # 사람이 검수하며 고칠 열
    m["최종_세부유형"] = m["GPT_세부유형"]
    m["검수메모"] = ""

    cols = ["번호", "엑셀행", "검수필요", "불일치", "유형_원본", "유형_키워드",
            "GPT_유형", "GPT_세부유형", "GPT_확신도", "GPT_근거",
            "최종_유형", "최종_세부유형", "검수메모", "사고내용"]
    m = m[cols].sort_values(["검수필요", "번호"], ascending=[False, True])

    # 사람이 이미 검수 중인 파일은 덮어쓰지 않는다
    out = REVIEW
    if out.exists():
        out = REVIEW.with_name(f"{REVIEW.stem}_{PROMPT_VERSION}_new.xlsx")
        print(f"\n[주의] {REVIEW.name}이 이미 있어 덮어쓰지 않고 {out.name}로 저장합니다.")
    save_review_xlsx(m, out)

    print(f"\n원본 라벨과 GPT 불일치: {m['불일치'].sum()}건")
    print(f"검수 필요(불일치·키워드 의심·확신도 낮음): {m['검수필요'].sum()}건")
    print(f"검수표: {out.relative_to(ROOT)}")


def save_review_xlsx(m: pd.DataFrame, path: Path):
    """검수하기 편하게: 머리글 고정, 필터, 열 너비, 검수필요 행 강조, 최종 열 색 구분"""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        m.to_excel(xw, index=False, sheet_name="검수")
        ws = xw.sheets["검수"]
        ws.freeze_panes = "C2"
        ws.auto_filter.ref = ws.dimensions
        widths = {"번호": 6, "엑셀행": 7, "검수필요": 9, "불일치": 7, "유형_원본": 9, "유형_키워드": 11,
                  "GPT_유형": 9, "GPT_세부유형": 11, "GPT_확신도": 9, "GPT_근거": 45,
                  "최종_유형": 10, "최종_세부유형": 12, "검수메모": 25, "사고내용": 80}
        col_idx = {c: i + 1 for i, c in enumerate(m.columns)}
        for c, w in widths.items():
            ws.column_dimensions[ws.cell(1, col_idx[c]).column_letter].width = w
        for cell in ws[1]:
            cell.font = Font(bold=True)

        need = PatternFill("solid", fgColor="FFF2CC")   # 검수 필요 행: 연노랑
        edit = PatternFill("solid", fgColor="DDEBF7")   # 사람이 고칠 열: 연파랑
        wrap = Alignment(wrap_text=True, vertical="top")
        for r in range(2, ws.max_row + 1):
            if ws.cell(r, col_idx["검수필요"]).value:
                for c in range(1, ws.max_column + 1):
                    ws.cell(r, c).fill = need
            for c in ("최종_유형", "최종_세부유형", "검수메모"):
                ws.cell(r, col_idx[c]).fill = edit
            for c in ("GPT_근거", "사고내용"):
                ws.cell(r, col_idx[c]).alignment = wrap

        # 최종 열은 목록에서 고르도록 (오타 방지)
        for col, values in (("최종_유형", TYPES), ("최종_세부유형", SUBTYPES)):
            dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=False)
            letter = ws.cell(1, col_idx[col]).column_letter
            dv.add(f"{letter}2:{letter}{ws.max_row}")
            ws.add_data_validation(dv)


if __name__ == "__main__":
    main()
