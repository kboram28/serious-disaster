"""V2.1 검증용 새 질문 15개 생성 (작성자 선택 편향을 줄이기 위해 무작위 + 고정 절차)

- 사고대응 5 / 상황설명 4 / 키워드 3: 테스트 100건(어느 DB에도 없음)에서 시드 고정으로 무작위 추출한
  사례를 바탕으로, gpt-5.5가 '현장 근로자가 물어볼 법한 질문'으로 바꿈. V2.1 개선 내용은 알려주지 않음
- 통계·조건 3: 업종·재해유형·시도 조합을 무작위로 뽑아 정해진 문장 틀에 넣음
출력  eval/questions_holdout.csv (기존 파일을 덮어씀), 근거 사례 번호는 '출처' 열에 기록
"""
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval/questions_holdout.csv"
SEED = 2026
MODEL = "gpt-5.5"

PROMPTS = {
    "사고대응": "아래 사고가 방금 현장에서 일어났다고 가정하고, 옆에 있던 동료 근로자가 챗봇에게 다급하게 물어볼 법한 질문 한 문장을 써라. "
              "사망했다는 표현은 쓰지 말고, 사고 직후 상황(쓰러짐, 끼임, 깔림 등)으로 표현한다.",
    "상황설명": "아래 사고와 비슷한 작업을 앞으로 하게 될 근로자가, 작업 전에 위험과 주의사항을 챗봇에게 물어볼 법한 질문 한 문장을 써라. "
              "사고가 났다는 말은 쓰지 않는다.",
    "키워드": "아래 사고의 핵심 작업이나 기인물을 바탕으로, 사고 사례를 검색할 때 쓸 법한 짧은 검색어(10자 안팎)를 써라.",
}
STAT_TEMPLATES = [
    "{업종}에서 {유형} 사고는 주로 어떤 기인물 때문에 발생해?",
    "{시도} 지역에서 발생한 {유형} 사고 사례를 알려줘",
    "{업종}에서 가장 많이 발생하는 재해유형과 예방대책은?",
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
    rnd = random.Random(SEED)

    gold = pd.read_csv(ROOT / "eval/test_gold.csv")
    picks = rnd.sample(list(gold.index), 12)
    plan = [("사고대응", i) for i in picks[:5]] + [("상황설명", i) for i in picks[5:9]] + [("키워드", i) for i in picks[9:]]

    rows = []
    for n, (kind, i) in enumerate(plan, 1):
        g = gold.loc[i]
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": PROMPTS[kind] + " 질문 문장만 출력한다."},
                      {"role": "user", "content": f"사고: {g['사고내용']}"}])
        rows.append({"질문ID": f"H{n:02d}", "질문유형": kind, "이전질문ID": "",
                     "질문": r.choices[0].message.content.strip().strip('"'), "출처": f"테스트 {g['문항ID']}"})

    cases = pd.read_csv(ROOT / "data/processed/cases_v2.csv", dtype=str, keep_default_na=False)
    db = cases[cases["split"] == "db"]
    inds = ["건설업", "제조업", "기타의사업", "운수창고통신업"]
    types = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘"]
    sidos = db[db["시도"] != ""]["시도"].value_counts().head(8).index.tolist()  # 사례가 어느 정도 있는 지역
    for k, t in enumerate(STAT_TEMPLATES):
        q = t.format(업종=rnd.choice(inds), 유형=rnd.choice(types), 시도=rnd.choice(sidos))
        rows.append({"질문ID": f"H{13 + k}", "질문유형": "통계·조건", "이전질문ID": "", "질문": q, "출처": "무작위 조합"})

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    for r in rows:
        print(r["질문ID"], r["질문유형"], "|", r["질문"])


if __name__ == "__main__":
    main()
