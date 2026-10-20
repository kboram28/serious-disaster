"""V2.2 (최종): V2 + 검증된 개선만 채택 (docs/decisions.md [13])

- 사고대응 질문: V2.1 형식 ('지금 바로 할 일' 먼저) — 새 질문 15개 블라인드 평가에서 사람·GPT 모두 5:0
- 그 외 질문: V2 형식 (V2.1 형식은 효과 없음)
- 인용 ID 검증 유지
- 버그 수정: 조건(업종·유형·지역)이 있으면 의도와 상관없이 집계 제공
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag

rag.STYLE = "v2.2"
ask = rag.ask
classify = rag.classify

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    res = ask(" ".join(sys.argv[1:]) or "건설업에서 가장 많이 발생하는 재해유형과 예방대책은?")
    print(res["answer"])
    print("\n--- 의도:", res["query"]["의도"], "/ 지어낸 ID:", res["invalid_citations"], "/ 사용량:", res["usage"])
