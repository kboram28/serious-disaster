"""4-2단계: 엑셀 사례에서 기인물·작업·재해유발요인 추출 (SIF와 같은 항목으로 맞추기)

입력  data/processed/cases_labeled.csv
출력  data/processed/excel_extract.jsonl    GPT 추출 캐시

문장에 적힌 내용만 추출한다. 원인을 추측해서 만들어 내지 않는다
(예: 문장에 '안전대 미착용'이 없으면 그런 말을 쓰지 않는다).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/cases_labeled.csv"
CACHE = ROOT / "data/processed/excel_extract.jsonl"
MODEL = "gpt-5.4-mini"

GUIDE = """너는 산업재해 사고 내용에서 정보를 추출하는 도우미다.
아래 세 항목을 사고 내용 문장에 적힌 내용만으로 짧게 추출한다. 문장에 없는 내용은 추측하지 않는다.

- 기인물: 재해를 일으킨 물체·설비·장소를 명사로 (예: 채광창, 비계, 굴착기, 지게차, 롤러, 개구부, 토사)
  여러 개면 가장 직접적인 것 하나. 알 수 없으면 "미상"
- 작업: 재해자가 하던 작업을 짧은 명사구로 (예: 비계 해체 작업, 지붕 판넬 교체 작업, 설비 청소)
  알 수 없으면 "미상"
- 재해유발요인: 사고가 어떻게 일어났는지 문장에 적힌 직접적인 경위를 한 문장으로
  (예: "채광창을 밟아 파손되며 6.5m 아래로 추락", "후진하던 덤프트럭에 깔림")
  안전조치 미흡 같은 근본 원인은 문장에 명시된 경우에만 쓴다."""

SCHEMA = {
    "type": "object",
    "properties": {"기인물": {"type": "string"}, "작업": {"type": "string"}, "재해유발요인": {"type": "string"}},
    "required": ["기인물", "작업", "재해유발요인"],
    "additionalProperties": False,
}


def extract(client: OpenAI, text: str) -> dict:
    for i in range(5):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": GUIDE},
                          {"role": "user", "content": f"사고 내용: {text}"}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "extract", "schema": SCHEMA, "strict": True}},
            )
            out = json.loads(r.choices[0].message.content)
            out["_tokens"] = {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
            return out
        except Exception as e:
            if i == 4:
                raise
            print(f"  재시도 {i + 1} ({type(e).__name__})")
            time.sleep(5 * 2 ** i)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())

    df = pd.read_csv(SRC)
    cache = {}
    if CACHE.exists():
        cache = {r["번호"]: r for r in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines()) if r}
    todo = [row for _, row in df.iterrows() if row["번호"] not in cache]
    print(f"전체 {len(df)} / 캐시 {len(cache)} / 새로 추출 {len(todo)}")

    def work(row):
        r = extract(client, row["사고내용_교정"])
        r["번호"] = int(row["번호"])
        return r

    with ThreadPoolExecutor(max_workers=6) as pool, CACHE.open("a", encoding="utf-8") as f:
        for r in pool.map(work, todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            cache[r["번호"]] = r

    ex = pd.DataFrame(cache.values())
    print("기인물 미상:", (ex["기인물"] == "미상").sum(), "/ 작업 미상:", (ex["작업"] == "미상").sum())
    print("기인물 상위:", ex["기인물"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
