"""questions_holdout2.csv 의 빈 행(N24~N30)을 gpt-5.5로 채움

원래 계획은 사람이 직접 작성이었지만, 작성자가 현장 경험이 없어 '질문 주제'만 사람이 미리 고정하고
문장은 gpt-5.5가 쓰게 바꿨다. V3를 구현하는 AI가 평가 질문을 직접 쓰지 않도록 하기 위함이다.

- 지침·법령 5 (N24~N28): 사고 유형 × (기준 / 절차)를 아래 TOPICS로 고정
- 이어지는질문 2 (N29, N30): 앞 질문을 시드로 뽑음 (사고대응 N01~N08 중 1, 통계·조건 N19~N23 중 1)
지침 목록·법 조문은 프롬프트에 주지 않는다. 이미 채워진 행이 있으면 멈춘다.
"""
import os
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "eval/questions_holdout2.csv"
SEED = 2027
MODEL = "gpt-5.5"

KINDS = {
    "기준": "법이나 안전 규정에서 정한 기준(높이·농도·거리 같은 수치, 반드시 해야 하는 의무)",
    "절차": "법이나 안전 지침에서 정한 작업 방법·순서·점검 사항",
}
TOPICS = [  # (사고 유형, 묻는 종류) — 유형을 고루 섞고 기준 2 / 절차 3
    ("떨어짐(높은 곳 작업)", "기준"),
    ("감전(전기 작업)", "기준"),
    ("끼임(기계 정비·청소)", "절차"),
    ("깔림·맞음(크레인·중량물 인양)", "절차"),
    ("질식(밀폐공간 작업)", "절차"),
]
GUIDE_PROMPT = ("너는 건설·제조 현장의 근로자다. '{topic}' 위험과 관련해 {kind}이 궁금해서 안전 챗봇에게 물어볼 법한 질문 한 문장을 써라. "
                "법 조항 번호나 지침 이름은 쓰지 말고, 현장에서 쓰는 편한 말투로 쓴다. 질문 문장만 출력한다.")
FOLLOW_PROMPT = ("근로자가 안전 챗봇에게 아래 질문을 하고 답을 받았다. 이어서 할 법한 짧은 후속 질문 한 문장을 써라. "
                 "앞 질문의 업종·사고 유형·상황을 다시 쓰지 말고 '그럼', '그런', '이런' 같은 말로 앞 대화를 가리켜서, "
                 "앞 대화를 봐야만 뜻이 통하게 쓴다. 질문 문장만 출력한다.")


def ask(client: OpenAI, system: str, user: str) -> str:
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": system},
                                                              {"role": "user", "content": user}])
    return r.choices[0].message.content.strip().strip('"')


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    guide = df.index[df["질문유형"] == "지침·법령"].tolist()
    follow = df.index[df["질문유형"] == "이어지는질문"].tolist()
    if len(guide) != len(TOPICS) or len(follow) != 2:
        sys.exit("빈 행 구성이 예상과 다름")
    if (df.loc[guide + follow, "질문"] != "").any():
        sys.exit("이미 채워진 행이 있음. 고정된 평가셋이라 덮어쓰지 않는다.")
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
    rnd = random.Random(SEED)

    for i, (topic, kind) in zip(guide, TOPICS):
        df.loc[i, "질문"] = ask(client, GUIDE_PROMPT.format(topic=topic, kind=KINDS[kind]), "질문을 써라.")
        df.loc[i, "출처"] = f"gpt-5.5 생성 (주제 고정: {topic} / {kind})"

    pools = [df.index[df["질문유형"] == "사고대응"].tolist(), df.index[df["질문유형"] == "통계·조건"].tolist()]
    for i, pool in zip(follow, pools):
        p = rnd.choice(pool)
        df.loc[i, "이전질문ID"] = df.loc[p, "질문ID"]
        df.loc[i, "질문"] = ask(client, FOLLOW_PROMPT, f"앞 질문: {df.loc[p, '질문']}")
        df.loc[i, "출처"] = "gpt-5.5 생성 (앞 질문 시드 추출)"

    df.to_csv(CSV, index=False, encoding="utf-8-sig")
    for i in guide + follow:
        r = df.loc[i]
        print(r["질문ID"], r["질문유형"], f"(이전: {r['이전질문ID']})" if r["이전질문ID"] else "", "|", r["질문"], "|", r["출처"])


if __name__ == "__main__":
    main()
