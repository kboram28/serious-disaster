"""6단계: GPT 기반 중대재해 유형 분류 시스템 — 웹 화면 (Streamlit)

실행
  streamlit run app/app.py

화면 구성 (포스터의 시스템 구성 4가지)
  💬 챗봇           대화형 질의응답 + 현장 사진 입력 + 답변 피드백
  📁 사고 데이터 분석  사용자가 올린 사고 데이터의 재해유형 분류·경향 분석
  ❓ 자주 묻는 질문   FAQ / 가이드라인 (누르면 챗봇으로 질문)
  📊 통계·사례 검색   업종·유형·지역 조건별 통계와 사례 목록
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v2"))
import rag_v22 as system  # 최종 버전 V2.2
from rag import TYPES, db, describe_image, search

FEEDBACK = ROOT / "data/feedback/feedback.jsonl"
BAR_COLOR = "#2a78d6"          # 단일 계열 막대 색 (한 가지 색만 사용)
MAX_UPLOAD_ROWS = 50           # 업로드 분석 최대 행 수 (비용 제한)

st.set_page_config(page_title="중대재해 예방 챗봇", page_icon="🦺", layout="wide")


# ---------------- 공통 ----------------

@st.cache_data
def cases() -> pd.DataFrame:
    return db()[0]


def case_card(r: pd.Series):
    where = f" · {r['시도']} {r['시군구']}" if r["시도"] else ""
    year = f" · {r['연도']}년" if r["연도"] else ""
    st.markdown(f"**[{r['case_id']}] {r['재해유형']}** (연관: {r['연관유형']}) · {r['업종']}{where}{year} · {r['출처']}")
    st.write(r["사고개요"])
    st.caption(f"기인물: {r['기인물']}  |  작업: {r['작업분류'] or r['작업'] or '-'}")
    st.caption(f"재해유발요인: {r['재해유발요인']}")
    with st.expander("예방대책"):
        st.write(r["예방대책"])


def save_feedback(entry: dict):
    FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
    entry["time"] = datetime.now().isoformat(timespec="seconds")
    with FEEDBACK.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def hbar(data: pd.DataFrame, label: str, value: str = "건수", height: int | None = None):
    """가로 막대 (단일 계열: 한 색, 범례 없음, 끝 둥글게, 마우스를 올리면 값 표시)"""
    chart = (alt.Chart(data)
             .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, height={"band": 0.7})
             .encode(y=alt.Y(f"{label}:N", sort="-x", title=None),
                     x=alt.X(f"{value}:Q", title=value, axis=alt.Axis(grid=True, gridOpacity=0.3)),
                     tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip(f"{value}:Q", format=",")])
             .properties(height=height or max(140, 28 * len(data))))
    text = chart.mark_text(align="left", dx=4).encode(text=alt.Text(f"{value}:Q", format=","), color=alt.value("gray"))
    st.altair_chart(chart + text, width="stretch")


# ---------------- 💬 챗봇 ----------------

def page_chat():
    st.header("💬 중대재해 예방 챗봇")
    st.caption("사고 사례·예방대책·사고 발생 시 대응을 물어보세요. 공단 재해사례 + SIF 아카이브 6,441건을 근거로 답합니다.")

    ss = st.session_state
    ss.setdefault("messages", [])

    with st.sidebar:
        st.subheader("📷 현장 사진 (선택)")
        photo = st.file_uploader("사진을 올리면 다음 질문에 함께 반영됩니다", type=["jpg", "jpeg", "png"], key="photo")
        if photo:
            st.image(photo, width="stretch")
        if st.button("대화 지우기"):
            ss.messages = []
            st.rerun()

    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_answer_extras(i, m)

    question = st.chat_input("예: 슬레이트 지붕에서 태양광 패널을 설치하려는데 주의할 점은?")
    if ss.pop("pending_question", None) and not question:
        question = ss.get("_pending_text")
    if not question:
        return

    photo_desc = None
    if photo:
        with st.spinner("사진을 분석하는 중..."):
            photo_desc = describe_image(photo.getvalue(), photo.type)["description"]
    full_q = question + (f"\n\n[현장 사진 설명]\n{photo_desc}" if photo_desc else "")

    ss.messages.append({"role": "user", "content": question + ("  📷" if photo_desc else "")})
    with st.chat_message("user"):
        st.markdown(ss.messages[-1]["content"])

    history = [(ss.messages[k]["question"], ss.messages[k]["content"])
               for k in range(len(ss.messages)) if ss.messages[k]["role"] == "assistant"][-3:]
    with st.chat_message("assistant"):
        with st.spinner("관련 사례를 찾는 중..."):
            res = system.ask(full_q, history)
        msg = {"role": "assistant", "content": res["answer"], "question": full_q,
               "retrieved": res["retrieved"], "query": res["query"], "photo_desc": photo_desc}
        ss.messages.append(msg)
        st.markdown(res["answer"])
        render_answer_extras(len(ss.messages) - 1, msg)


def render_answer_extras(i: int, m: dict):
    df = cases().set_index("case_id")
    if m.get("photo_desc"):
        with st.expander("📷 사진 분석 결과 (검색에 사용)"):
            st.write(m["photo_desc"])
    with st.expander(f"🔎 참고한 사례 {len(m['retrieved'])}건 · 검색 조건"):
        q = m["query"]
        st.caption(f"의도: {q['의도']} / 업종: {q['업종'] or '-'} / 재해유형: {q['재해유형'] or '-'} / "
                   f"지역: {q['시도'] or '-'} / 검색문장: {q['검색문장']}")
        for cid in m["retrieved"]:
            r = df.loc[cid].copy()
            r["case_id"] = cid
            case_card(r)
            st.divider()

    key = f"fb_{i}"
    if st.session_state.get(key):
        st.caption(f"피드백 감사합니다 ({st.session_state[key]})")
        return
    c1, c2, c3 = st.columns([1, 1, 6])
    comment = c3.text_input("의견 (선택)", key=f"fbc_{i}", label_visibility="collapsed", placeholder="의견 (선택)")
    for col, label, val in ((c1, "👍 도움됨", "good"), (c2, "👎 아쉬움", "bad")):
        if col.button(label, key=f"{key}_{val}"):
            save_feedback({"question": m["question"], "answer": m["content"], "rating": val,
                           "comment": comment, "retrieved": m["retrieved"]})
            st.session_state[key] = label
            st.rerun()


# ---------------- 📁 사고 데이터 분석 ----------------

def page_upload():
    st.header("📁 사고 데이터 분석")
    st.caption(f"우리 현장의 사고·아차사고 기록(엑셀/CSV)을 올리면 재해유형을 분류하고 경향을 분석합니다. "
               f"(최대 {MAX_UPLOAD_ROWS}건)")
    f = st.file_uploader("파일 선택", type=["xlsx", "csv"])
    if not f:
        st.info("사고 내용이 적힌 열이 하나 이상 있으면 됩니다. 예: '사고내용' 열에 '사다리에서 작업 중 떨어짐'")
        return
    data = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
    text_cols = [c for c in data.columns if data[c].dtype == object]
    if not text_cols:
        st.error("글자로 된 열이 없습니다.")
        return
    col = st.selectbox("사고 내용이 적힌 열", text_cols)
    rows = data[col].dropna().astype(str).head(MAX_UPLOAD_ROWS).tolist()
    st.write(f"분석 대상: {len(rows)}건")
    if not st.button("분석 시작", type="primary"):
        return

    with st.spinner("재해유형을 분류하는 중..."):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(system.classify, rows))
    out = pd.DataFrame({"사고내용": rows, "재해유형": [r["유형"] for r in results],
                        "유사사례": [", ".join(r["retrieved"][:3]) for r in results]})
    vc = out["재해유형"].value_counts().reindex(TYPES, fill_value=0).rename_axis("재해유형").reset_index(name="건수")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("재해유형 분포")
        hbar(vc[vc["건수"] > 0], "재해유형")
    with c2:
        st.subheader("분류 결과")
        st.dataframe(out, width="stretch", hide_index=True)
        st.download_button("결과 내려받기 (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                           "분류결과.csv", "text/csv")

    top = vc.iloc[0]["재해유형"] if vc["건수"].max() > 0 else None
    with st.spinner("경향과 예방대책을 정리하는 중..."):
        summary = system.ask(
            f"우리 현장 사고 {len(rows)}건을 분류한 결과 재해유형 분포는 "
            + ", ".join(f"{r['재해유형']} {r['건수']}건" for _, r in vc.iterrows() if r["건수"])
            + f"입니다. 가장 많은 {top} 사고의 공통 위험과 우선 예방대책을 알려줘. 사고 예시: "
            + " / ".join(out[out["재해유형"] == top]["사고내용"].head(5)))
    st.subheader("경향 분석과 우선 예방대책")
    st.markdown(summary["answer"])


# ---------------- ❓ 자주 묻는 질문 ----------------

FAQ = {
    "사고가 났을 때": ["작업자가 높은 곳에서 떨어져 의식이 없어. 지금 뭘 해야 해?",
                   "동료가 기계에 손이 끼였어. 바로 해야 할 조치는?",
                   "현장에서 작업자가 감전되어 쓰러졌어",
                   "굴착 현장에서 작업자가 토사에 매몰됐어"],
    "작업 전 확인": ["슬레이트 지붕 위에서 작업할 때 주의할 점은?",
                 "달비계로 외벽 작업을 할 때 꼭 확인할 것은?",
                 "지게차가 다니는 작업장에서 보행자 안전 수칙은?",
                 "설비를 청소·정비하기 전에 해야 할 일은?"],
    "통계": ["건설업에서 가장 많이 발생하는 재해유형과 예방대책은?",
           "제조업 끼임 사고는 주로 어떤 기인물 때문에 발생해?",
           "서비스업(기타의사업)에서는 어떤 사고가 많아?"],
}


def page_faq():
    st.header("❓ 자주 묻는 질문")
    st.caption("질문을 누르면 챗봇에서 바로 답변합니다.")
    for group, qs in FAQ.items():
        st.subheader(group)
        cols = st.columns(2)
        for k, q in enumerate(qs):
            if cols[k % 2].button(q, key=f"faq_{group}_{k}", width="stretch"):
                st.session_state["nav_to"] = "💬 챗봇"   # 메뉴 위젯이 만들어진 뒤라 다음 실행 때 반영
                st.session_state["pending_question"] = True
                st.session_state["_pending_text"] = q
                st.rerun()


# ---------------- 📊 통계·사례 검색 ----------------

def page_stats():
    st.header("📊 통계 · 사례 검색")
    df = cases()
    c1, c2, c3, c4 = st.columns(4)
    src = c1.multiselect("출처", sorted(df["출처"].unique()))
    ind = c2.multiselect("업종", df["업종"].value_counts().index.tolist())
    typ = c3.multiselect("재해유형", TYPES)
    sido = c4.multiselect("지역 (공단 재해사례만)", sorted(x for x in df["시도"].unique() if x))
    kw = st.text_input("검색어 (뜻으로 검색, 예: 채광창 파손 추락)")

    m = np.ones(len(df), dtype=bool)
    for col, sel in (("출처", src), ("업종", ind), ("재해유형", typ), ("시도", sido)):
        if sel:
            m &= df[col].isin(sel).to_numpy()
    d = df[m]
    st.metric("조건에 맞는 사례", f"{len(d):,}건")
    if d.empty:
        return

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("재해유형")
        hbar(d["재해유형"].value_counts().rename_axis("재해유형").reset_index(name="건수"), "재해유형")
    with g2:
        st.subheader("기인물 상위 10")
        hbar(d["기인물"].replace("미상", pd.NA).dropna().value_counts().head(10)
             .rename_axis("기인물").reset_index(name="건수"), "기인물")
    yrs = d[d["연도"] != ""]["연도"].astype(int)
    yrs = yrs[(yrs >= 2016) & (yrs <= 2023)]
    if len(yrs):
        st.subheader("연도별 사례 수 (SIF, 2016~2023)")
        hbar(yrs.value_counts().sort_index().rename_axis("연도").reset_index(name="건수").astype({"연도": str}),
             "연도", height=260)

    st.subheader("사례")
    if kw:
        hits = search(kw, m, 20)
        for _, r in hits.iterrows():
            case_card(r)
            st.divider()
    else:
        st.dataframe(d[["case_id", "출처", "재해유형", "연관유형", "업종", "시도", "기인물", "사고개요"]],
                     width="stretch", hide_index=True)


# ---------------- 메인 ----------------

PAGES = {"💬 챗봇": page_chat, "📁 사고 데이터 분석": page_upload, "❓ 자주 묻는 질문": page_faq,
         "📊 통계·사례 검색": page_stats}

if "nav_to" in st.session_state:
    st.session_state["page"] = st.session_state.pop("nav_to")

with st.sidebar:
    st.title("🦺 중대재해 예방")
    st.caption("GPT 기반 중대재해 유형 분류 시스템 (V2.2)")
    st.radio("메뉴", list(PAGES), key="page")
    st.divider()

PAGES[st.session_state["page"]]()
