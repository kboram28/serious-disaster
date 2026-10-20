"""5단계: V1·V2 답변 품질 채점

  python eval/score_answers.py auto     ① 자동 검사: 인용 유효성, 조건 준수
  python eval/score_answers.py judge    ② GPT 채점: 블라인드 쌍 비교 (상위 모델)
  python eval/score_answers.py sheet    ③ 사람 채점 시트 생성 (15문항, 블라인드)
  python eval/score_answers.py report   ①②③ 결과 종합 → eval/results/answer_quality.md

  뒤에 holdout 을 붙이면 새 질문 세트에서 V2 vs V2.1 비교 (예: python eval/score_answers.py judge holdout)

블라인드 원칙: 채점자(GPT·사람)는 어느 답변이 V1/V2인지 모르고, A/B 순서는 문항마다 무작위.
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval/results"
CASES = ROOT / "data/processed/cases_v2.csv"
# 비교 설정: main = 기존 43문항 V1 vs V2 / holdout = 새 15문항 V2 vs V2.1
SETS = {
    "main": {"pair": ("v1", "v2"), "questions": ROOT / "eval/questions.csv", "suffix": ""},
    "holdout": {"pair": ("v2", "v2.1"), "questions": ROOT / "eval/questions_holdout.csv", "suffix": "_holdout"},
}
QSET = sys.argv[2] if len(sys.argv) > 2 else "main"
PAIR = SETS[QSET]["pair"]
SFX = SETS[QSET]["suffix"]
QUESTIONS = SETS[QSET]["questions"]
JUDGE = RES / f"judge_pairwise{SFX}.jsonl"
SHEET = ROOT / f"eval/human_pairwise{SFX}.xlsx"
REPORT = RES / f"answer_quality{SFX}.md"
JUDGE_MODEL = "gpt-5.5"
SEED = 42

# 질문에 명시된 조건 (자동 검사용). 인용된 사례의 검증된 속성으로 판정
CONDITIONS = {
    "Q04": {"기인물포함": "지게차"},
    "Q29": {"업종": "제조업"},
    "Q30": {"업종": "건설업", "재해유형": "떨어짐"},
    "Q31": {"시도": "경기", "재해유형": "끼임"},
    "Q32": {"시도": "서울"},
    "Q33": {"재해유형": "기타"},
    "Q41": {"업종": "건설업"},
    "Q42": {"업종": "건설업"},
    "Q43": {"업종": "제조업", "재해유형": "끼임"},
    "H13": {"업종": "운수창고통신업", "재해유형": "부딪힘"},
    "H14": {"시도": "경북", "재해유형": "떨어짐"},
    "H15": {"업종": "건설업"},
}
STAT_QUESTIONS = ("Q29", "Q30", "Q41", "H13", "H15")

CRITERIA = ["정확성", "관련성", "구체성", "근거제시", "안전성"]


def load_answers(name: str) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "eval"))
    from run_eval import cited_cases  # V1·V2 공통 형식(E012 / S…)으로 답변 원문에서 다시 추출
    rows = [json.loads(l) for l in (RES / f"{name}{SFX}_answers.jsonl").read_text(encoding="utf-8").splitlines() if l]
    df = pd.DataFrame(rows).set_index("질문ID")
    df["인용사례"] = df["answer"].map(cited_cases)
    return df


def cases() -> pd.DataFrame:
    return pd.read_csv(CASES, dtype=str, keep_default_na=False).set_index("case_id")


# ---------- ① 자동 검사 ----------

def meets(c: pd.Series, cond: dict) -> bool:
    for k, v in cond.items():
        if k == "기인물포함":
            if v not in c["기인물"] + c["사고개요"]:
                return False
        elif c[k] != v:
            return False
    return True


def auto() -> pd.DataFrame:
    cs = cases()
    v1_db = set(cs.index[(cs["출처"] == "공단 재해사례") & (cs["split"] == "db")])
    v2_db = set(cs.index[cs["split"] == "db"])
    rows = []
    for name in PAIR:
        db = v1_db if name == "v1" else v2_db
        ans = load_answers(name)
        for qid, a in ans.iterrows():
            cited = list(a["인용사례"])
            valid = [c for c in cited if c in db]
            row = {"시스템": name, "질문ID": qid, "인용수": len(cited), "유효인용": len(valid)}
            if qid in CONDITIONS:
                ok = [meets(cs.loc[c], CONDITIONS[qid]) for c in valid]
                row["조건준수"] = sum(ok)
                row["조건검사대상"] = len(ok)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"auto_checks{SFX}.csv", index=False, encoding="utf-8-sig")
    return df


# ---------- ② GPT 블라인드 쌍 비교 ----------

JUDGE_GUIDE = """너는 산업안전 전문가로서 중대재해 예방 챗봇의 두 답변(A, B)을 비교 평가한다.
어느 답변이 어떤 시스템인지 모른다. 길이나 형식이 아니라 내용으로 판단한다.

각 답변이 인용한 사례 ID의 '실제 데이터'를 함께 준다. 답변이 사례 내용·유형·업종·지역을 실제와 다르게 말하면 정확성 감점.
질문의 조건(업종·지역·유형 등)과 맞지 않는 사례를 근거로 쓰면 관련성 감점.
건수·순위 같은 숫자는 [참고 집계]가 있으면 그것과 대조한다.

평가 기준 (각 1~5점)
- 정확성: 사실·숫자·사례 설명이 실제 데이터와 맞는가
- 관련성: 질문 의도와 조건에 맞는 내용인가
- 구체성: 현장에서 바로 쓸 수 있을 만큼 구체적인가
- 근거제시: 주장에 실제 사례 근거를 제시하는가
- 안전성: 잘못되거나 위험한 조언이 없는가 (없으면 5)
마지막으로 전체적으로 더 나은 답변을 A, B, 비슷 중에서 고른다."""


def judge_schema():
    score = {"type": "object", "properties": {c: {"type": "integer", "minimum": 1, "maximum": 5} for c in CRITERIA},
             "required": CRITERIA, "additionalProperties": False}
    return {"type": "object",
            "properties": {"A": score, "B": score, "승자": {"type": "string", "enum": ["A", "B", "비슷"]},
                           "이유": {"type": "string"}},
            "required": ["A", "B", "승자", "이유"], "additionalProperties": False}


def ab_order(qid: str) -> dict:
    """문항마다 고정된 무작위 A/B 배정 (사람 시트와 GPT 채점이 같은 배정을 씀)"""
    rnd = random.Random(f"{SEED}-{qid}")
    a, b = PAIR
    return {"A": a, "B": b} if rnd.random() < 0.5 else {"A": b, "B": a}


def cited_facts(ids: list[str], cs: pd.DataFrame) -> str:
    out = []
    for c in ids[:12]:
        if c not in cs.index:
            out.append(f"[{c}] (데이터에 없는 ID)")
            continue
        r = cs.loc[c]
        where = f", 지역 {r['시도']} {r['시군구']}" if r["시도"] else ""
        out.append(f"[{c}] 유형 {r['재해유형']}, 업종 {r['업종']}{where}, 기인물 {r['기인물']} / {r['사고개요'][:120]}")
    return "\n".join(out) or "(인용 없음)"


def ref_stats(qid: str, cs: pd.DataFrame) -> str:
    """통계 질문의 참고 집계: V1은 원본 데이터, V2는 통합 데이터를 쓰므로 둘 다 제공"""
    cond = CONDITIONS.get(qid, {})
    if qid not in STAT_QUESTIONS:
        return ""
    lines = []
    for label, d in (("원본 엑셀(V1 데이터, 원본 라벨)", cs[(cs["출처"] == "공단 재해사례") & (cs["split"] == "db")]),
                     ("통합 데이터(V2 데이터)", cs[cs["split"] == "db"])):
        if "업종" in cond:
            d = d[d["업종"] == cond["업종"]]
        if "재해유형" in cond:
            d = d[d["재해유형"] == cond["재해유형"]]
        vc = d["재해유형"].value_counts()
        lines.append(f"{label}: {len(d)}건 / " + ", ".join(f"{k} {v}" for k, v in vc.items()))
    return "[참고 집계] 각 답변은 자기 데이터 기준으로 맞으면 정확한 것으로 본다\n" + "\n".join(lines)


def judge():
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
    qs = pd.read_csv(QUESTIONS, dtype=str, keep_default_na=False).set_index("질문ID")
    ans = {n: load_answers(n) for n in PAIR}
    cs = cases()
    done = {}
    if JUDGE.exists():
        done = {r["질문ID"]: r for r in map(json.loads, JUDGE.read_text(encoding="utf-8").splitlines()) if r}

    def work(qid):
        order = ab_order(qid)
        q = qs.loc[qid]
        prev = ""
        if q["이전질문ID"]:
            prev = f"(이전 질문: {qs.loc[q['이전질문ID'], '질문']})\n"
        parts = [f"[질문] {prev}{q['질문']}", ref_stats(qid, cs)]
        for side in ("A", "B"):
            a = ans[order[side]].loc[qid]
            parts.append(f"[답변 {side}]\n{a['answer']}\n\n[답변 {side}가 인용한 사례의 실제 데이터]\n"
                         f"{cited_facts(list(a['인용사례']), cs)}")
        for i in range(4):
            try:
                r = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "system", "content": JUDGE_GUIDE},
                              {"role": "user", "content": "\n\n".join(p for p in parts if p)}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "judge", "schema": judge_schema(), "strict": True}})
                break
            except Exception as e:
                if i == 3:
                    raise
                print(f"  재시도 {qid} ({type(e).__name__})")
        out = json.loads(r.choices[0].message.content)
        win = {"A": order["A"], "B": order["B"], "비슷": "비슷"}[out["승자"]]
        side = {v: k for k, v in order.items()}
        return {"질문ID": qid, "배정": order, "승자": win, PAIR[0]: out[side[PAIR[0]]],
                PAIR[1]: out[side[PAIR[1]]], "이유": out["이유"],
                "usage": {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}}

    todo = [q for q in qs.index if q not in done]
    print(f"GPT 채점 ({JUDGE_MODEL}): 전체 {len(qs)} / 완료 {len(done)} / 실행 {len(todo)}")
    with ThreadPoolExecutor(max_workers=4) as pool, JUDGE.open("a", encoding="utf-8") as f:
        for r in pool.map(work, todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()


# ---------- ③ 사람 채점 시트 ----------

HUMAN_PICK = {"키워드": 2, "상황설명": 3, "사고대응": 3, "통계·조건": 3, "데이터 밖": 2, "실사용(V1 시연)": 2}


def sheet():
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    qs = pd.read_csv(QUESTIONS, dtype=str, keep_default_na=False)
    rnd = random.Random(SEED)
    picked = []
    for kind, n in (HUMAN_PICK if QSET == "main" else {}).items():
        ids = qs[qs["질문유형"] == kind]["질문ID"].tolist()
        picked += ids if kind.startswith("실사용") and "Q41" in ids else rnd.sample(ids, n)
    if QSET == "main":
        picked = [p for p in picked if p != "Q41"] + ["Q41"]  # Q42를 볼 때 Q41 맥락 필요 → 함께
    else:
        picked = qs["질문ID"].tolist()  # 새 질문 세트는 전부 사람 채점
    picked = list(dict.fromkeys(picked))
    rnd.shuffle(picked)

    ans = {n: load_answers(n) for n in PAIR}
    q = qs.set_index("질문ID")
    rows, key = [], []
    for i, qid in enumerate(picked, 1):
        order = ab_order(qid)
        prev = f"(이전 질문: {q.loc[q.loc[qid, '이전질문ID'], '질문']}) " if q.loc[qid, "이전질문ID"] else ""
        rows.append({"문항": f"H{i:02d}", "질문": prev + q.loc[qid, "질문"],
                     "답변A": ans[order["A"]].loc[qid, "answer"], "답변B": ans[order["B"]].loc[qid, "answer"],
                     "더 좋은 답변": "", "이유": ""})
        key.append({"문항": f"H{i:02d}", "질문ID": qid, "A": order["A"], "B": order["B"]})

    guide = [
        ("목적", "두 답변 중 현장 안전 담당자에게 더 도움이 되는 답변을 고른다."),
        ("블라인드", "어느 쪽이 어느 시스템인지 알 수 없게 순서를 섞었다. 답변 형식·말투로 추측하지 않는다."),
        ("기준", "정확성(사실·숫자가 맞는가), 관련성(질문 조건에 맞는가), 구체성, 근거 사례 제시, 잘못된 조언이 없는가"),
        ("선택", "더 좋은 답변 칸에서 A / B / 비슷 중 선택. 이유는 짧게."),
        ("주의", "길이가 길다고 더 좋은 것은 아니다. 질문과 맞지 않는 사례를 길게 나열하면 감점."),
    ]
    with pd.ExcelWriter(SHEET, engine="openpyxl") as xw:
        pd.DataFrame(rows).to_excel(xw, index=False, sheet_name="비교")
        pd.DataFrame(guide, columns=["항목", "설명"]).to_excel(xw, index=False, sheet_name="채점기준")
        pd.DataFrame(key).to_excel(xw, index=False, sheet_name="key")
        ws = xw.sheets["비교"]
        for col, w in zip("ABCDEF", (6, 30, 70, 70, 12, 30)):
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "C2"
        for cell in ws[1]:
            cell.font = Font(bold=True)
        edit = PatternFill("solid", fgColor="DDEBF7")
        for r in range(2, ws.max_row + 1):
            ws.row_dimensions[r].height = 400
            for c in range(1, 7):
                ws.cell(r, c).alignment = Alignment(wrap_text=True, vertical="top")
            for c in (5, 6):
                ws.cell(r, c).fill = edit
        dv = DataValidation(type="list", formula1='"A,B,비슷"')
        dv.add(f"E2:E{ws.max_row}")
        ws.add_data_validation(dv)
        xw.sheets["채점기준"].column_dimensions["B"].width = 100
        xw.book["key"].sheet_state = "hidden"
    print(f"사람 채점 시트: {SHEET.relative_to(ROOT)} ({len(rows)}문항)")


# ---------- 종합 리포트 ----------

def report():
    a = pd.read_csv(RES / f"auto_checks{SFX}.csv")
    P0, P1 = PAIR
    j = pd.DataFrame(map(json.loads, JUDGE.read_text(encoding="utf-8").splitlines()))
    qs = pd.read_csv(QUESTIONS, dtype=str, keep_default_na=False).set_index("질문ID")
    j["질문유형"] = j["질문ID"].map(qs["질문유형"])

    def pct(x, y):
        return f"{x / y:.0%} ({x}/{y})" if y else "-"

    lines = [f"# 답변 품질 비교 ({P0.upper()} vs {P1.upper()}, {QSET})", "", "## ① 자동 검사", "",
             f"| 지표 | {P0} | {P1} |", "|---|---|---|"]
    g = {"v1": a[a["시스템"] == P0], "v2": a[a["시스템"] == P1]}
    lines.append(f"| 인용한 사례 ID가 실제 DB에 있음 | {pct(g['v1']['유효인용'].sum(), g['v1']['인용수'].sum())} | "
                 f"{pct(g['v2']['유효인용'].sum(), g['v2']['인용수'].sum())} |")
    lines.append(f"| 조건 있는 질문({int(g["v1"]["조건검사대상"].notna().sum())}개)에서 인용 사례가 조건에 맞음 | "
                 f"{pct(int(g['v1']['조건준수'].sum()), int(g['v1']['조건검사대상'].sum()))} | "
                 f"{pct(int(g['v2']['조건준수'].sum()), int(g['v2']['조건검사대상'].sum()))} |")
    lines += ["", "질문별 조건 준수:", "", f"| 질문 | 조건 | {P0} | {P1} |", "|---|---|---|---|"]
    for qid, cond in CONDITIONS.items():
        if qid not in set(g["v1"]["질문ID"]):
            continue
        r = {n: g[n][g[n]["질문ID"] == qid].iloc[0] for n in ("v1", "v2")}
        lines.append(f"| {qid} {qs.loc[qid, '질문'][:25]} | {cond} | "
                     f"{pct(int(r['v1']['조건준수']), int(r['v1']['조건검사대상']))} | "
                     f"{pct(int(r['v2']['조건준수']), int(r['v2']['조건검사대상']))} |")

    win = j["승자"].value_counts()
    lines += ["", f"## ② GPT 블라인드 채점 ({JUDGE_MODEL}, {len(j)}문항)", "",
              f"**승: {P1} {win.get(P1, 0)} / {P0} {win.get(P0, 0)} / 비슷 {win.get('비슷', 0)}**", "",
              f"| 기준 (1~5점 평균) | {P0} | {P1} |", "|---|---|---|"]
    for c in CRITERIA:
        lines.append(f"| {c} | {j[P0].map(lambda s: s[c]).mean():.2f} | {j[P1].map(lambda s: s[c]).mean():.2f} |")
    lines += ["", "질문유형별 승패:", "", f"| 질문유형 | {P1} 승 | {P0} 승 | 비슷 |", "|---|---|---|---|"]
    for t, d in j.groupby("질문유형"):
        vc = d["승자"].value_counts()
        lines.append(f"| {t} | {vc.get(P1, 0)} | {vc.get(P0, 0)} | {vc.get('비슷', 0)} |")
    lines += ["", f"{P0}이 이기거나 비슷한 문항:", ""]
    for _, r in j[j["승자"] != P1].iterrows():
        lines.append(f"- {r['질문ID']} ({r['승자']}) {qs.loc[r['질문ID'], '질문'][:40]} — {r['이유'][:150]}")

    if SHEET.exists():
        s = pd.read_excel(SHEET, sheet_name=None)
        h = s["비교"].merge(s["key"], on="문항")
        if h["더 좋은 답변"].notna().all():
            h["사람"] = h.apply(lambda r: r[r["더 좋은 답변"]] if r["더 좋은 답변"] in ("A", "B") else "비슷", axis=1)
            h = h.merge(j[["질문ID", "승자"]], on="질문ID")
            vc = h["사람"].value_counts()
            agree = (h["사람"] == h["승자"]).mean()
            lines += ["", f"## ③ 사람 블라인드 채점 ({len(h)}문항)", "",
                      f"**승: {P1} {vc.get(P1, 0)} / {P0} {vc.get(P0, 0)} / 비슷 {vc.get('비슷', 0)}**", "",
                      f"사람과 GPT 채점 일치율: **{agree:.0%}**", "",
                      "| 문항 | 질문 | 사람 | GPT | 사람 이유 |", "|---|---|---|---|---|"]
            for _, r in h.iterrows():
                lines.append(f"| {r['질문ID']} | {r['질문'][:30]} | {r['사람']} | {r['승자']} | {str(r['이유'])[:60]} |")
        else:
            lines += ["", "## ③ 사람 블라인드 채점", "", "(아직 채점 전)"]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"리포트: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"auto": lambda: print(auto().groupby("시스템")[["인용수", "유효인용", "조건준수", "조건검사대상"]].sum()),
     "judge": judge, "sheet": sheet, "report": report}[sys.argv[1]]()
