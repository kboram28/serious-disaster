"""V2 검색 인덱스 만들기: 사례마다 임베딩(의미 벡터)을 계산해 파일로 저장

입력  data/processed/cases_v2.csv (split=db 만 사용, 테스트 제외)
출력  vectordb/v2_emb.npy     사례별 임베딩 (행 순서 = v2_ids.csv)
      vectordb/v2_ids.csv     case_id 목록과 임베딩에 쓴 텍스트

벡터 저장소는 별도 DB(Chroma 등) 대신 numpy 파일을 쓴다.
6,441건 규모에서는 전체와 유사도를 계산해도 수 밀리초라 충분하고, 설치 문제도 없다.
데이터가 바뀌면 다시 실행. 이미 계산한 case_id 는 재사용한다.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/cases_v2.csv"
DB_DIR = ROOT / "vectordb"
EMB = DB_DIR / "v2_emb.npy"
IDS = DB_DIR / "v2_ids.csv"
EMBED_MODEL = "text-embedding-3-small"
BATCH = 200


def doc_text(r) -> str:
    """임베딩할 텍스트: 검색에 도움이 되는 항목을 한 문단으로"""
    parts = [
        f"재해유형: {r['재해유형']} (연관: {r['연관유형']})",
        f"업종: {r['업종']} {r['업종_세부'] or ''}".strip(),
        f"작업: {r['작업분류'] or ''} {r['작업'] or ''}".strip(),
        f"기인물: {r['기인물']}",
        f"사고개요: {r['사고개요']}",
        f"재해유발요인: {r['재해유발요인']}",
    ]
    return "\n".join(p for p in parts if not p.endswith(": "))


def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    v = np.array([d.embedding for d in r.data], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # 정규화 → 내적 = 코사인 유사도


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())

    df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    df = df[df["split"] == "db"].reset_index(drop=True)
    df["text"] = df.apply(doc_text, axis=1)

    old = {}
    if EMB.exists() and IDS.exists():
        prev = pd.read_csv(IDS, dtype=str, keep_default_na=False)
        vecs = np.load(EMB)
        old = {(c, t): vecs[i] for i, (c, t) in enumerate(zip(prev["case_id"], prev["text"]))}

    todo = [i for i, r in df.iterrows() if (r["case_id"], r["text"]) not in old]
    print(f"DB {len(df):,}건 / 재사용 {len(df) - len(todo):,} / 새로 계산 {len(todo):,} ({EMBED_MODEL})")
    new = {}
    for s in range(0, len(todo), BATCH):
        idx = todo[s:s + BATCH]
        for i, v in zip(idx, embed(client, df.loc[idx, "text"].tolist())):
            new[i] = v
        print(f"  {min(s + BATCH, len(todo))}/{len(todo)}")

    mat = np.stack([new[i] if i in new else old[(r["case_id"], r["text"])] for i, r in df.iterrows()])
    DB_DIR.mkdir(exist_ok=True)
    np.save(EMB, mat)
    df[["case_id", "text"]].to_csv(IDS, index=False, encoding="utf-8-sig")
    print(f"저장: {EMB.relative_to(ROOT)} {mat.shape}")


if __name__ == "__main__":
    main()
