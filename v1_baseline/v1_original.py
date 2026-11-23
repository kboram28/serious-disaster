"""실제 V1 재현: 졸업작품 원본 로직 (키워드 사전 → 업종 AND 유형 필터 → 최대 5건 요약)

원본 코드는 v1_original/views.py (docs/decisions.md [24]).
v1.py(추측한 V1, '통째로 넣기')와 별도로, 원본에서 찾은 실제 방식을 같은 조건에서 평가한다.

원본 그대로 둔 것
- 동의어 사전 (업종 8개, 재해유형 6개), map_synonyms (사전에 없으면 문장 전체를 반환하는 버그 포함)
- 업종 AND 재해유형 str.contains 필터, 앞에서 최대 5건, 프롬프트 문구
- 키워드는 현재 질문에서만 찾고, 대화 이력은 메시지로 함께 전달

평가를 위해 바꾼 것 (모두 다른 시스템과 조건을 맞추기 위함)
1. 데이터: 원본 엑셀과 같은 값(유형_원본·업종_원본·사고내용·예방대책)을 쓰되 테스트셋 100건 제외 (v1.py와 같은 DB 409건)
2. 모델: gpt-4-turbo(서비스 종료) → V2와 같은 gpt-5.4-mini, temperature·max_tokens 없이 기본값 (V2·v1.py와 동일)
3. 호출 방식: openai 0.28 ChatCompletion → 현재 SDK (호출 내용은 같음)
4. 분류 기능은 원본에 없으므로 classify를 제공하지 않음 → run_eval은 질문 세트만 실행
5. str.contains에 regex=False: 버그로 질문 문장 전체가 검색어가 될 때 괄호 등이 있으면 정규식 오류로 멈추는 것을 막음
   (어차피 문장 전체와 일치하는 사례는 없어 0건이라는 결과는 같음)

사용법
  python v1_baseline/v1_original.py "건설업 추락 사고"
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/cases_labeled.csv"
SPLIT = ROOT / "data/processed/split.csv"

MODEL = "gpt-5.4-mini"

# ---- 원본 그대로 (v1_original/views.py) ----
industry_synonyms = {
    "건설업": ["건설", "공사", "건축", "시공"],
    "제조업": ["제작", "제조", "가공", "생산"],
    "운송업": ["운송", "운반", "교통", "물류", "배송"],
    "농업": ["농사", "농업", "농장", "경작"],
    "어업": ["어업", "수산", "양식"],
    "광업": ["광업", "채굴", "광산"],
    "서비스업": ["서비스", "서비스업", "접객업", "요식업", "레저업"],
    "의료업": ["의료", "병원", "보건", "치료"],
}

disaster_synonyms = {
    "떨어짐": ["떨어짐", "추락", "낙하", "낙상"],
    "끼임": ["끼임", "압착", "기계 끼임", "눌림"],
    "부딪힘": ["부딪힘", "충돌", "접촉 사고"],
    "깔림": ["깔림", "매몰", "덮임"],
    "맞음": ["맞음", "타격", "충격", "부상"],
    "기타": ["기타", "알 수 없는 사고", "기타 사고"],
}


def map_synonyms(text, synonyms_dict):
    for key, synonyms in synonyms_dict.items():
        for synonym in synonyms:
            if synonym in text:
                return key
    return text  # 원본 버그: 못 찾으면 문장 전체를 키워드로 반환


def extract_keywords(text):
    industry = map_synonyms(text, industry_synonyms)
    disaster = map_synonyms(text, disaster_synonyms)
    return industry, disaster


def get_disaster_info(industry, disaster):
    df = data()
    # 원본은 str.contains 기본값(정규식). 질문 문장에 괄호 등이 있으면 정규식 오류가 나므로 regex=False
    filtered_data = df[
        (df["업종"].str.contains(industry, na=False, regex=False))
        & (df["재해 유형"].str.contains(disaster, na=False, regex=False))
    ]

    if filtered_data.empty:
        disaster_info = f"{industry} 업종에서 '{disaster}' 사고에 대한 구체적인 정보는 찾을 수 없습니다."
        prevention_info = "따라서 예방 대책에 대한 정보도 제공되지 않았습니다."
        return disaster_info, prevention_info, []

    case_count = min(len(filtered_data), 5)
    limited_data = filtered_data.head(case_count)

    disaster_info = f"{industry}에서 발생한 '{disaster}' 사고 사례는 다음과 같습니다:\n\n"
    prevention_info = f"'{disaster}' 사고에 대한 예방 대책은 다음과 같습니다:\n\n"

    for idx, row in limited_data.iterrows():
        disaster_info += f"{idx + 1}. 사고 내용: {row['사고 내용']}\n"
        prevention_info += f"{idx + 1}. 예방 대책: {row['예방 대책']}\n\n"

    ids = [f"E{idx + 1:03d}" for idx in limited_data.index]
    return disaster_info.strip(), prevention_info.strip(), ids


def build_prompt(industry, disaster, disaster_info, prevention_info):
    prompt = f"다음은 '{industry}' 업종에서 발생한 '{disaster}' 사고 사례 및 예방 대책입니다:\n\n"
    prompt += f"{disaster_info}\n예방 대책: {prevention_info}\n\n이 정보를 간단히 요약해 주세요."
    return prompt


# ---- 평가용 연결 ----
@lru_cache
def client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())


@lru_cache
def data() -> pd.DataFrame:
    """원본 엑셀 열 이름으로 맞춘 DB 사례. 인덱스 = 번호 - 1 (원본의 idx + 1 = 사례 번호가 되도록)"""
    df = pd.read_csv(DATA).merge(pd.read_csv(SPLIT), on="번호")
    db = df[df["split"] == "db"].sort_values("번호")
    # 원본 값 열만 먼저 고른 뒤 이름 변경 (정제된 '업종' 열이 따로 있어 그대로 바꾸면 이름이 겹침)
    cols = {"유형_원본": "재해 유형", "업종_원본": "업종", "사고내용": "사고 내용", "예방대책": "예방 대책"}
    out = db[list(cols)].rename(columns=cols)
    out.index = db["번호"] - 1
    return out


def ask(question: str, history: list[tuple[str, str]] = ()) -> dict:
    """원본 chatbot_api의 텍스트 처리 부분. history = [(이전 질문, 이전 답변), ...]"""
    chat_history = []
    for q, a in history:  # 원본도 사용자 원문 질문과 챗봇 답변을 이력으로 쌓음
        chat_history += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

    industry, disaster = extract_keywords(question)
    if not (industry and disaster):  # 원본: 빈 입력일 때만 해당 (버그로 사실상 항상 통과)
        return {"answer": "질문에서 업종이나 사고 유형을 확인할 수 없습니다.", "usage": {"in": 0, "cached": 0, "out": 0},
                "retrieved": [], "query": {"업종": industry, "재해유형": disaster}}

    disaster_info, prevention_info, ids = get_disaster_info(industry, disaster)
    prompt = build_prompt(industry, disaster, disaster_info, prevention_info)
    r = client().chat.completions.create(model=MODEL, messages=chat_history + [{"role": "user", "content": prompt}])
    return {"answer": r.choices[0].message.content.strip(), "usage": usage(r),
            "retrieved": ids, "query": {"업종": industry, "재해유형": disaster}}


def usage(r) -> dict:
    cached = getattr(getattr(r.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return {"in": r.usage.prompt_tokens, "cached": cached, "out": r.usage.completion_tokens}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) or "건설업 추락 사고"
    res = ask(q)
    print(res["query"], res["retrieved"])
    print(res["answer"])
    print("\n---", res["usage"])
