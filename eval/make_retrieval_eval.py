"""검색 평가셋 200개 생성 (V3 구현 전에 고정, 이후 수정 금지)

- 검색 DB 사례(split=db) 중 200건을 재해유형 비율대로 층화 추출 (시드 2027)
- gpt-5.5가 각 사례를 '이 사고와 비슷한 사례를 찾으려는 사람의 검색 질문'으로 바꿈. 정답 = 원래 사례(case_id)
- 사고개요 문장을 그대로 베끼면 검색이 너무 쉬워지므로, 질문과 사고개요의 글자 겹침 비율을 함께 저장해
  사람이 표본을 확인할 때 참고한다 (4단계)
출력  eval/retrieval_eval.csv — 이미 있으면 덮어쓰지 않고 멈춘다

평가 지표(나중에 eval_retrieval.py): 정답 case_id가 검색 상위 5·10개 안에 있는 비율(Recall@5·10), MRR
한계: 비슷한 사고가 여러 건 있어도 정답은 1건이라 실제 성능보다 낮게 나올 수 있다. V2.2와 V3에 똑같이 적용되므로 비교에는 문제없다.
"""
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval/retrieval_eval.csv"
SEED = 2027
MODEL = "gpt-5.5"
N = 200

PROMPT = ("아래는 실제 산업재해 사고 개요다. 이 사고를 보거나 전해 들은 현장 근로자가, 비슷한 사고 사례를 찾으려고 "
          "안전 챗봇에게 물어볼 법한 질문 한 문장을 써라.\n"
          "- 사고 개요의 문장이나 표현을 그대로 옮기지 말고 자기 말로 바꿔 쓴다\n"
          "- 작업 내용, 기인물(사고를 일으킨 기계·물체), 상황 중 핵심 2~3개는 담는다\n"
          "- 날짜, 회사·지역 이름, 정확한 수치(높이 m, 나이 등)는 쓰지 않는다\n"
          "질문 문장만 출력한다.")


def allocate(counts: pd.Series, n: int) -> dict:
    """재해유형별 비율대로 n개를 나눔 (최대 나머지 방식: 내림 후 나머지가 큰 유형부터 1개씩)"""
    raw = counts / counts.sum() * n
    alloc = raw.astype(int)
    for t in (raw - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[t] += 1
    return alloc.to_dict()


def overlap(q: str, src: str) -> float:
    """질문의 글자 2-gram 중 사고개요에도 있는 비율 (높을수록 베낀 것)"""
    grams = lambda s: {s[i:i + 2] for i in range(len(s) - 1)}
    q, src = q.replace(" ", ""), src.replace(" ", "")
    gq = grams(q)
    return round(len(gq & grams(src)) / len(gq), 2) if gq else 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if OUT.exists():
        sys.exit(f"{OUT.name} 이(가) 이미 있음. 고정된 평가셋이라 덮어쓰지 않는다.")
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())

    cases = pd.read_csv(ROOT / "data/processed/cases_v2.csv", dtype=str, keep_default_na=False)
    db = cases[cases["split"] == "db"]
    db = db[~db["사고개요"].duplicated(keep=False)]  # 개요가 똑같은 사례는 정답이 하나로 정해지지 않으므로 제외
    alloc = allocate(db["재해유형"].value_counts(), N)
    print(f"후보 {len(db)}건, 재해유형별 추출: {alloc}")

    rnd = random.Random(SEED)
    picks = []
    for t, k in sorted(alloc.items()):
        picks += rnd.sample(sorted(db.index[db["재해유형"] == t]), k)
    rnd.shuffle(picks)  # 유형별로 몰려 있지 않게 섞음
    sample = db.loc[picks]

    def make(src: str) -> str:
        for attempt in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": PROMPT},
                                                                          {"role": "user", "content": src}])
                return r.choices[0].message.content.strip().strip('"')
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"재시도: {str(e)[:100]}")

    with ThreadPoolExecutor(8) as ex:
        questions = list(ex.map(make, sample["사고개요"]))

    out = pd.DataFrame({
        "평가ID": [f"R{k:03d}" for k in range(1, N + 1)],
        "질문": questions,
        "정답_case_id": sample["case_id"].values,
        "재해유형": sample["재해유형"].values,
        "업종": sample["업종"].values,
        "출처": sample["출처"].values,
        "겹침비율": [overlap(q, s) for q, s in zip(questions, sample["사고개요"])],
        "사고개요": sample["사고개요"].values,  # 표본 확인용
    })
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT.relative_to(ROOT)} ({len(out)}개)")
    print(f"겹침비율 평균 {out['겹침비율'].mean():.2f}, 0.6 이상 {int((out['겹침비율'] >= 0.6).sum())}개")
    for r in out.head(5).itertuples():
        print(f"\n{r.평가ID} [{r.재해유형}] 겹침 {r.겹침비율}\n  질문: {r.질문}\n  원문: {r.사고개요[:120]}")


if __name__ == "__main__":
    main()
