"""V2: 질문 이해 → 조건 필터(코드) → 의미 검색 → 근거 기반 답변 (RAG)

V1과 같은 인터페이스: ask(질문, history), classify(사고내용)
모델도 V1과 같은 gpt-5.4-mini. 차이는 데이터(통합 6,441건)와 구조(검색·필터·집계)뿐.

사용법
  python v2/rag.py "경기도에서 발생한 끼임 사고 사례를 알려줘"
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from relabel import GUIDE as LABEL_GUIDE  # 엑셀·SIF 라벨링과 같은 분류 기준

DATA = ROOT / "data/processed/cases_v2.csv"
EMB = ROOT / "vectordb/v2_emb.npy"
IDS = ROOT / "vectordb/v2_ids.csv"
MODEL = "gpt-5.4-mini"
EMBED_MODEL = "text-embedding-3-small"
TOP_K = 8          # 답변에 넣을 사례 수
CLASSIFY_K = 10    # 분류할 때 참고할 유사 사례 수

TYPES = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘", "기타"]
SIDO = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원",
        "충북", "충남", "전북", "전남", "경북", "경남", "제주"]


# ---------- 데이터·클라이언트 ----------

@lru_cache
def client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())


@lru_cache
def db() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    ids = pd.read_csv(IDS, dtype=str, keep_default_na=False)["case_id"]
    df = df.set_index("case_id").loc[ids].reset_index()   # 임베딩 행 순서와 맞춤 (split=db 만)
    return df, np.load(EMB)


@lru_cache
def industries() -> list[str]:
    return sorted(db()[0]["업종"].unique())


def embed_query(text: str) -> np.ndarray:
    v = np.array(client().embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding, dtype=np.float32)
    return v / np.linalg.norm(v)


def usage_add(total: dict, r) -> None:
    u = r.usage
    total["in"] += u.prompt_tokens
    total["out"] += u.completion_tokens
    total["cached"] += getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0


def chat_json(messages, schema: dict, name: str, usage: dict) -> dict:
    r = client().chat.completions.create(
        model=MODEL, messages=messages,
        response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
    )
    usage_add(usage, r)
    return json.loads(r.choices[0].message.content)


# ---------- 1) 질문 이해: 조건과 의도를 구조화 ----------

def parse_query(question: str, history, usage: dict) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "의도": {"type": "string", "enum": ["사례·예방", "통계", "사고대응"]},
            "업종": {"type": "array", "items": {"type": "string", "enum": industries()}},
            "재해유형": {"type": "array", "items": {"type": "string", "enum": TYPES}},
            "시도": {"type": "array", "items": {"type": "string", "enum": SIDO}},
            "검색문장": {"type": "string"},
        },
        "required": ["의도", "업종", "재해유형", "시도", "검색문장"],
        "additionalProperties": False,
    }
    guide = f"""사용자 질문을 분석해 검색 조건을 만든다.
- 의도: 사고 사례나 예방대책을 묻는 경우 '사례·예방', 빈도·순위·가장 많은 유형 등을 묻는 경우 '통계',
  사고가 이미 일어나 지금 무엇을 해야 하는지 묻는 경우 '사고대응'
- 업종·재해유형·시도: 질문에 명시된 조건만 넣는다. 명시되지 않으면 빈 배열.
  업종 목록: {', '.join(industries())} (전기공사·인테리어·철거 등은 건설업, 공장은 제조업, 서비스·시설관리·청소 등은 기타의사업)
  재해유형 분류 기준: 떨어짐/끼임/깔림/맞음/부딪힘/기타(매몰·감전·폭발·화재·질식·익사 등)
- 검색문장: 이전 대화를 반영해 질문만 읽어도 뜻이 통하도록 다시 쓴 검색용 문장
  (예: 이전 질문이 건설업이었고 지금 '그럼 예방은?'이면 '건설업 주요 사고 예방대책')
- 이전 대화의 조건(업종 등)을 이어받는 질문이면 그 조건도 넣는다."""
    msgs = [{"role": "system", "content": guide}]
    for q, a in history:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a[:1500]}]
    msgs.append({"role": "user", "content": question})
    return chat_json(msgs, schema, "query", usage)


# ---------- 2) 조건 필터 + 3) 의미 검색 (코드) ----------

def filter_mask(df: pd.DataFrame, q: dict) -> tuple[np.ndarray, list[str]]:
    mask = np.ones(len(df), dtype=bool)
    notes = []
    if q["업종"]:
        mask &= df["업종"].isin(q["업종"]).to_numpy()
    if q["재해유형"]:
        mask &= df["재해유형"].isin(q["재해유형"]).to_numpy()
    if q["시도"]:
        mask &= df["시도"].isin(q["시도"]).to_numpy()
        notes.append("지역 조건은 지역 정보가 있는 공단 재해사례에만 적용됨 (SIF는 장소 익명화)")
    return mask, notes


def search(text: str, mask: np.ndarray, k: int) -> pd.DataFrame:
    df, emb = db()
    scores = emb @ embed_query(text)
    scores = np.where(mask, scores, -np.inf)
    top = np.argsort(-scores)[:min(k, int(mask.sum()))]
    out = df.iloc[top].copy()
    out["유사도"] = scores[top].round(3)
    return out


def case_block(r) -> str:
    where = f" | 지역: {r['시도']} {r['시군구']}".rstrip() if r["시도"] else ""
    year = f" | 연도: {r['연도']}" if r["연도"] else ""
    work = r["작업분류"] or r["작업"]
    return (f"[{r['case_id']}] 출처: {r['출처']} | 재해유형: {r['재해유형']} (연관: {r['연관유형']}) | "
            f"업종: {r['업종']}{where}{year}\n"
            f"  작업: {work} | 기인물: {r['기인물']}\n"
            f"  사고개요: {r['사고개요']}\n"
            f"  재해유발요인: {r['재해유발요인']}\n"
            f"  예방대책: {r['예방대책']}")


def stats_block(df: pd.DataFrame, mask: np.ndarray) -> str:
    """통계 질문용 집계 (GPT가 세지 않고 코드가 센다)"""
    d = df[mask]
    if d.empty:
        return "조건에 맞는 사례 0건"
    lines = [f"조건에 맞는 사례: {len(d):,}건 (공단 재해사례 {(d['출처'] == '공단 재해사례').sum()}, SIF {(d['출처'] == 'SIF').sum()})"]
    for col, n in (("재해유형", 6), ("업종", 8), ("기인물", 10), ("연관유형", 8)):
        vc = d[col].replace("", pd.NA).dropna().value_counts().head(n)
        lines.append(f"{col} 상위: " + ", ".join(f"{k} {v}건({v / len(d):.0%})" for k, v in vc.items()))
    work = d["작업분류"].where(d["작업분류"] != "", d["작업"]).str.split(" > ").str[-1]
    vc = work.replace("", pd.NA).dropna().value_counts().head(10)
    lines.append("작업 상위: " + ", ".join(f"{k} {v}건" for k, v in vc.items()))
    return "\n".join(lines)


# ---------- 4) 답변 ----------

ANSWER_GUIDE = """너는 산업현장 중대재해 예방을 돕는 챗봇이다. 아래 [검색된 사례]와 [집계]만 근거로 답한다.

규칙
- 사례를 언급할 때는 반드시 사례 ID를 [E123] 또는 [S1a2b3c4d5e] 형식으로 인용한다.
- 숫자(건수·비율·순위)는 [집계]에 있는 값만 쓴다. 직접 세거나 추정하지 않는다.
- 검색된 사례가 질문 조건(업종·지역·유형)과 맞지 않으면 쓰지 않는다.
- 데이터에 근거가 없는 일반 안전 지식(예: 응급조치 절차)을 쓸 때는 '일반 안전수칙'이라고 구분해서 표시한다.
- 관련 사례가 없으면 없다고 말하고, 가장 가까운 사례를 '유사 사례'로만 제시한다.
- 구성: 핵심 답변 → 관련 사례(ID, 무슨 일이 있었는지, 원인) → 예방대책(사례 근거) → 필요 시 즉시 대응
"""


STYLE = "v2"   # v2.1: 의도별 답변 형식 + 인용 ID 검증 (v2/rag_v21.py 에서 설정)

COMMON_RULES = """너는 산업현장 중대재해 예방을 돕는 챗봇이다. 아래 [검색된 사례]와 [집계]를 근거로 답한다.

공통 규칙
- 사례를 언급할 때는 [검색된 사례]에 있는 사례 ID만 [E123] 또는 [S1a2b3c4d5e] 형식으로 인용한다. ID를 지어내지 않는다.
- 숫자(건수·비율·순위)는 [집계]에 있는 값만 쓴다. 직접 세거나 추정하지 않는다.
- 질문 조건(업종·지역·유형)과 맞지 않는 사례는 쓰지 않는다.
- 첫 문장은 반드시 질문에 대한 직접적인 답으로 시작한다. "직접 일치하는 사례는 없지만" 같은 말로 시작하지 않는다.
  직접 맞는 사례가 없으면 사례 목록에서만 '(유사 사례)'라고 짧게 표시한다.
- 짧고 명확하게 쓴다. 같은 내용을 반복하지 않는다."""

STYLE_GUIDES = {
    "사고대응": """
이 질문은 사고가 이미 일어난 긴급 상황이다. 현장에서 휴대폰으로 바로 읽는다고 생각하고 아래 형식을 지킨다.

## 지금 바로 할 일
1~6단계 번호 목록. 한 줄에 하나씩, 동사로 시작 (예: "기계 전원을 즉시 차단한다").
2차 사고 방지 → 119 신고 → 응급조치 → 현장 통제 순서를 우선한다.
이 부분은 일반 응급조치 지식으로 쓰고, 제목 옆에 '(일반 응급조치)'라고 한 번만 표시한다.
## 하면 안 되는 것
1~3개, 한 줄씩 (예: 끼인 신체를 억지로 빼내지 않는다)
## 비슷한 사고 사례
2~3개, 사례마다 한 줄 (ID + 무슨 일이 있었는지)
## 재발 방지
2~4개, 사례 근거 ID 포함
전체 600자 안팎.""",
    "사례·예방": """
형식
## 핵심
질문에 대한 답 1~2문장 (어떤 위험이 있고 무엇이 가장 중요한지)
## 관련 사례
3~5개, 사례마다 1~2줄 (ID, 무슨 일이 있었는지, 원인)
## 예방대책
반드시 포함. 4~6개, 현장에서 바로 할 수 있는 조치 위주, 근거 사례 ID 포함""",
    "통계": """
형식
## 결과
[집계] 숫자로 질문에 바로 답한다 (순위·건수·비율)
## 주요 사례
2~3개, 한 줄씩
## 예방 포인트
3~4개, 근거 사례 ID 포함""",
}


def answer_guide(intent: str) -> str:
    if STYLE == "v2":
        return ANSWER_GUIDE
    if STYLE == "v2.2" and intent != "사고대응":   # v2.2: 검증된 사고대응 형식만 채택, 나머지는 V2 형식
        return ANSWER_GUIDE
    return COMMON_RULES + "\n" + STYLE_GUIDES[intent]


def check_citations(answer: str, allowed: set[str]) -> tuple[str, list[str]]:
    """검색된 사례가 아닌 ID 인용을 찾아 표시 (지어낸 ID 방지)"""
    import re
    bad = sorted({c for c in re.findall(r"\b(E\d{3}|S[0-9a-f]{10})\b", answer) if c not in allowed})
    for c in bad:
        answer = re.sub(rf"\[?{c}\]?", "(확인되지 않은 사례)", answer)
    return answer, bad


def ask(question: str, history: list[tuple[str, str]] = ()) -> dict:
    usage = {"in": 0, "out": 0, "cached": 0}
    df, _ = db()
    q = parse_query(question, history, usage)
    mask, notes = filter_mask(df, q)
    if mask.sum() == 0:
        notes.append("조건에 맞는 사례가 없어 조건 없이 검색함")
        mask = np.ones(len(df), dtype=bool)

    hits = search(q["검색문장"], mask, TOP_K)
    context = "[검색된 사례]\n" + "\n".join(case_block(r) for _, r in hits.iterrows())
    cond = {k: q[k] for k in ("업종", "재해유형", "시도") if q[k]}
    # v2.2: '가장 많은 유형 + 예방대책'처럼 통계와 예방이 섞인 질문에서 의도가 '사례·예방'으로 잡혀
    #       집계가 빠지는 문제 → 조건이 있으면 의도와 상관없이 집계를 넣고, 빈도를 물을 때만 쓰게 함
    if q["의도"] == "통계" or (STYLE == "v2.2" and cond):
        note = "" if q["의도"] == "통계" else " (질문이 건수·순위·비율을 물을 때만 사용)"
        context = f"[집계]{note}\n" + stats_block(df, mask) + "\n\n" + context
    context += f"\n\n[검색 조건] 의도: {q['의도']} / 조건: {cond or '없음'}" + "".join(f"\n- {n}" for n in notes)

    msgs = [{"role": "system", "content": answer_guide(q["의도"])}]
    for pq, pa in history:
        msgs += [{"role": "user", "content": pq}, {"role": "assistant", "content": pa}]
    msgs.append({"role": "user", "content": f"{context}\n\n[질문] {question}"})
    r = client().chat.completions.create(model=MODEL, messages=msgs)
    usage_add(usage, r)
    answer, bad = r.choices[0].message.content, []
    if STYLE != "v2":
        answer, bad = check_citations(answer, set(hits["case_id"]))
    return {"answer": answer, "usage": usage, "query": q,
            "retrieved": hits["case_id"].tolist(), "invalid_citations": bad}


IMAGE_GUIDE = """너는 산업안전 점검자다. 현장 사진을 보고 사고 사례 검색에 쓸 수 있도록 객관적으로 설명한다.
- 장소·시설 (예: 슬레이트 지붕, 철골 구조물, 공장 내부 컨베이어)
- 작업 내용과 사람의 위치·자세 (보이는 경우)
- 설비·장비·기인물 (예: 고소작업대, 지게차, 이동식 비계, 사다리)
- 눈에 보이는 위험 요소 (예: 안전난간 없음, 개구부, 채광창, 안전대 미착용, 적재물 불안정)
- 사고가 난 상황이면 그 상태 (예: 사람이 쓰러져 있음, 장비가 넘어져 있음)
보이지 않는 것은 추측하지 않는다. 5~8줄의 짧은 문장으로 쓴다. 사람의 신원은 언급하지 않는다."""


def describe_image(image: bytes, mime: str = "image/jpeg") -> dict:
    """현장 사진 → 검색용 텍스트 설명 (사진 DB 없이 텍스트 사례 DB로 검색하기 위함)"""
    import base64
    usage = {"in": 0, "out": 0, "cached": 0}
    url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    r = client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": IMAGE_GUIDE},
                  {"role": "user", "content": [{"type": "text", "text": "이 현장 사진을 설명해줘."},
                                               {"type": "image_url", "image_url": {"url": url}}]}],
    )
    usage_add(usage, r)
    return {"description": r.choices[0].message.content.strip(), "usage": usage}


def classify(accident: str) -> dict:
    """사고 내용 → 재해유형. 분류 기준(GUIDE) + 유사 사례 10건을 참고"""
    usage = {"in": 0, "out": 0, "cached": 0}
    df, _ = db()
    hits = search(accident, np.ones(len(df), dtype=bool), CLASSIFY_K)
    refs = "\n".join(f"- ({r['재해유형']}, 연관 {r['연관유형']}) {r['사고개요'][:150]}" for _, r in hits.iterrows())
    schema = {"type": "object", "properties": {"유형": {"type": "string", "enum": TYPES}},
              "required": ["유형"], "additionalProperties": False}
    out = chat_json(
        [{"role": "system", "content": LABEL_GUIDE + "\n\n참고로 비슷한 과거 사례와 그 분류를 함께 준다. "
                                                     "사례는 참고만 하고, 분류 기준에 따라 판단한다."},
         {"role": "user", "content": f"[유사 사례]\n{refs}\n\n사고 내용: {accident}"}],
        schema, "type", usage)
    return {"유형": out["유형"], "usage": usage, "retrieved": hits["case_id"].tolist()}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]) or "지붕 위에서 작업할 때 주의할 점과 예방대책을 알려줘"
    res = ask(question)
    print(res["answer"])
    print("\n--- 검색 조건:", res["query"])
    print("--- 사용량:", res["usage"])
