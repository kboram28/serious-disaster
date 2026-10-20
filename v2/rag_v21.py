"""V2.1: V2와 같은 데이터·검색에 답변 단계만 개선 (docs/decisions.md [12])

- 질문 의도별 답변 형식: 사고대응은 '지금 바로 할 일'을 맨 앞에 짧게, 사례·예방은 예방대책 필수, 통계는 숫자부터
- 첫 문장은 바로 답 ("직접 일치하는 사례는 없지만…"으로 시작하지 않음)
- 검색된 사례가 아닌 ID를 인용하면 '(확인되지 않은 사례)'로 바꿈

사용법
  python v2/rag_v21.py "동료가 기계에 팔이 끼였어"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag

rag.STYLE = "v2.1"
ask = rag.ask
classify = rag.classify

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    res = ask(" ".join(sys.argv[1:]) or "동료가 기계에 팔이 끼였어. 바로 해야 할 조치는?")
    print(res["answer"])
    print("\n--- 의도:", res["query"]["의도"], "/ 지어낸 ID:", res["invalid_citations"], "/ 사용량:", res["usage"])
