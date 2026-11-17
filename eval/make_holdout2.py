"""V3 평가용 새 질문 30개 중 23개 생성 (V3 구현 전에 고정, 이후 수정 금지)

구성 (계획 00_1주일_일정.md Day 1)
- 사고대응 8 / 상황설명(작업 전 위험) 6 / 키워드 4
    테스트 100건(검색 DB에 없음)에서 시드 고정으로 무작위 추출한 사례를 gpt-5.5가 질문으로 바꿈.
    V2.1 holdout(H01~H15)에 이미 쓴 테스트 사례는 제외한다.
- 통계·조건 5: 업종·재해유형·시도를 무작위로 골라 문장 틀에 넣음
- 지침·법령 5 / 이어지는질문 2: 빈 행으로 남겨 두고 사람이 직접 작성 (N24~N30)

V3의 개선 내용(지침·법령 근거, 하이브리드 검색 등)은 생성 프롬프트에 알려주지 않는다.
출력  eval/questions_holdout2.csv — 이미 있으면 덮어쓰지 않고 멈춘다 (고정된 평가셋 보호)
"""
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval/questions_holdout2.csv"
PREV = ROOT / "eval/questions_holdout.csv"
SEED = 2027  # V2.1 holdout은 2026
MODEL = "gpt-5.5"

# 기존 make_holdout.py 와 같은 문장 (V2.1 holdout과 같은 방식으로 만들어야 결과를 비교할 수 있음)
PROMPTS = {
    "사고대응": "아래 사고가 방금 현장에서 일어났다고 가정하고, 옆에 있던 동료 근로자가 챗봇에게 다급하게 물어볼 법한 질문 한 문장을 써라. "
              "사망했다는 표현은 쓰지 말고, 사고 직후 상황(쓰러짐, 끼임, 깔림 등)으로 표현한다.",
    "상황설명": "아래 사고와 비슷한 작업을 앞으로 하게 될 근로자가, 작업 전에 위험과 주의사항을 챗봇에게 물어볼 법한 질문 한 문장을 써라. "
              "사고가 났다는 말은 쓰지 않는다.",
    "키워드": "아래 사고의 핵심 작업이나 기인물을 바탕으로, 사고 사례를 검색할 때 쓸 법한 짧은 검색어(10자 안팎)를 써라.",
}
COUNTS = [("사고대응", 8), ("상황설명", 6), ("키워드", 4)]

STAT_TEMPLATES = [
    "{업종}에서 {유형} 사고는 주로 어떤 기인물 때문에 발생해?",
    "{시도} 지역에서 발생한 {유형} 사고 사례를 알려줘",
    "{업종}에서 가장 많이 발생하는 재해유형과 예방대책은?",
    "{업종} {유형} 사고는 어떤 작업을 하다가 많이 일어나?",
    "{시도}에서 일어난 {업종} 사고 사례를 보여줘",
]
INDUSTRIES = ["건설업", "제조업", "기타의사업", "운수창고통신업"]
TYPES = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘"]

MANUAL = [("지침·법령", 5), ("이어지는질문", 2)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if OUT.exists():
        sys.exit(f"{OUT.name} 이(가) 이미 있음. 고정된 평가셋이라 덮어쓰지 않는다.")
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
    rnd = random.Random(SEED)

    # V2.1 holdout에 쓴 테스트 사례 제외 ("테스트 T016" → T016)
    prev = pd.read_csv(PREV, dtype=str, keep_default_na=False)
    used = {m.group(0) for s in prev["출처"] for m in [re.search(r"T\d{3}", s)] if m}
    gold = pd.read_csv(ROOT / "eval/test_gold.csv")
    pool = [i for i in gold.index if gold.loc[i, "문항ID"] not in used]
    print(f"테스트 100건 중 V2.1 holdout 사용 {len(used)}건 제외 → 후보 {len(pool)}건")

    need = sum(n for _, n in COUNTS)
    picks = iter(rnd.sample(pool, need))
    plan = [(kind, next(picks)) for kind, n in COUNTS for _ in range(n)]

    rows = []
    for kind, i in plan:
        g = gold.loc[i]
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": PROMPTS[kind] + " 질문 문장만 출력한다."},
                      {"role": "user", "content": f"사고: {g['사고내용']}"}])
        rows.append({"질문유형": kind, "이전질문ID": "",
                     "질문": r.choices[0].message.content.strip().strip('"'), "출처": f"테스트 {g['문항ID']}"})

    cases = pd.read_csv(ROOT / "data/processed/cases_v2.csv", dtype=str, keep_default_na=False)
    db = cases[cases["split"] == "db"]
    sidos = db[db["시도"] != ""]["시도"].value_counts().head(8).index.tolist()  # 사례가 어느 정도 있는 지역
    prev_q = set(prev["질문"])
    for t in STAT_TEMPLATES:
        while True:  # V2.1 holdout과 똑같은 문장이 나오면 다시 뽑음
            q = t.format(업종=rnd.choice(INDUSTRIES), 유형=rnd.choice(TYPES), 시도=rnd.choice(sidos))
            if q not in prev_q:
                break
        rows.append({"질문유형": "통계·조건", "이전질문ID": "", "질문": q, "출처": "무작위 조합"})

    for kind, n in MANUAL:  # 사람이 직접 채울 빈 행
        rows += [{"질문유형": kind, "이전질문ID": "", "질문": "", "출처": "직접 작성"} for _ in range(n)]

    df = pd.DataFrame(rows)
    df.insert(0, "질문ID", [f"N{k:02d}" for k in range(1, len(df) + 1)])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    for r in df.itertuples():
        print(r.질문ID, r.질문유형, "|", r.질문 or "(직접 작성)", "|", r.출처)


if __name__ == "__main__":
    main()
