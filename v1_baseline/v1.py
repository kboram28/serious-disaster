"""V1 재현: 기존 방식 (엑셀 데이터를 프롬프트에 통째로 넣는 GPT 챗봇)

원래 코드는 유실되었고 데이터 전달 방식도 기억나지 않아,
가장 단순한 '프롬프트에 통째로' 방식으로 재현한다. (docs/decisions.md [6])

- 데이터: 원본 엑셀의 값 그대로 (원본 라벨·지역·업종, 정제/검증 결과 사용 안 함)
- 테스트셋(split=test) 100건은 DB에서 제외
- 모델: V2와 같은 gpt-5.4-mini (차이를 데이터·구조로만 비교하기 위해)

사용법
  python v1_baseline/v1.py "채광창 작업 시 예방대책은?"
"""
import json
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
TYPES = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘", "기타"]

INSTRUCTION = """너는 산업현장 중대재해 예방을 돕는 챗봇이다.
아래 [중대재해 사례 데이터]를 참고해 사용자의 질문에 답한다.
관련 사례가 있으면 사례 번호를 함께 알려주고, 재해 유형과 예방대책을 정리해 제시한다."""


@lru_cache
def client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())


@lru_cache
def context() -> str:
    """DB 사례를 원본 값 그대로 한 줄씩 나열한 텍스트"""
    df = pd.read_csv(DATA).merge(pd.read_csv(SPLIT), on="번호")
    db = df[df["split"] == "db"].sort_values("번호")
    lines = [
        f"[{r['번호']}] 재해유형: {r['유형_원본']} | 업종: {r['업종_원본']} | 지역: {r['지역_원본']} | "
        f"사고내용: {r['사고내용']} | 예방대책: {r['예방대책']}"
        for _, r in db.iterrows()
    ]
    return "[중대재해 사례 데이터]\n" + "\n".join(lines)


def system_prompt() -> str:
    # 데이터를 앞에 두어 매번 같은 접두부가 되도록 함 → OpenAI 자동 프롬프트 캐싱으로 비용 절감
    return context() + "\n\n" + INSTRUCTION


def ask(question: str, history: list[tuple[str, str]] = ()) -> dict:
    """챗봇 답변. history = [(이전 질문, 이전 답변), ...] (이어지는 대화용)"""
    messages = [{"role": "system", "content": system_prompt()}]
    for q, a in history:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": question})
    r = client().chat.completions.create(model=MODEL, messages=messages)
    return {"answer": r.choices[0].message.content, "usage": usage(r)}


def classify(accident: str) -> dict:
    """사고 내용 → 재해유형 (평가용). 같은 데이터·지시문에 분류 요청만 추가"""
    schema = {
        "type": "object",
        "properties": {"유형": {"type": "string", "enum": TYPES}},
        "required": ["유형"],
        "additionalProperties": False,
    }
    r = client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt()},
                  {"role": "user", "content": f"다음 사고의 재해유형을 분류해줘.\n사고 내용: {accident}"}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "type", "schema": schema, "strict": True}},
    )
    return {"유형": json.loads(r.choices[0].message.content)["유형"], "usage": usage(r)}


def usage(r) -> dict:
    cached = getattr(getattr(r.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return {"in": r.usage.prompt_tokens, "cached": cached, "out": r.usage.completion_tokens}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) or "지붕 위에서 작업할 때 주의할 점과 예방대책을 알려줘"
    res = ask(q)
    print(res["answer"])
    print("\n---", res["usage"])
