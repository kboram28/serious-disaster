"""2-2단계: 블라인드 라벨 + 이견 조정 → 테스트 정답 확정

입력  eval/test_blind.xlsx      사람이 블라인드로 라벨링한 시트 (+ 숨김 key 시트)
출력  eval/test_gold.csv        테스트 정답 (블라인드 원래 값도 함께 보존)

조정 절차: 블라인드 라벨과 GPT 판정이 다른 4건만 다시 검토해 사람이 최종 결정.
연관유형이 둘 이상이면 쉼표로 구분한다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BLIND = ROOT / "eval/test_blind.xlsx"
GOLD = ROOT / "eval/test_gold.csv"

# 문항ID: (정답_유형, 정답_연관유형, 조정 사유)
ADJUST = {
    "T035": ("맞음", "맞음", "떨어진 것은 슬라브(물체) → 물체가 떨어져 사람에게 닿음 = 맞음"),
    "T075": ("부딪힘", "떨어짐", "지게차가 넘어지며 바닥에 머리를 부딪힘. 높은 곳에서 떨어진 것은 아님"),
    "T033": ("맞음", "떨어짐", "떨어진 로프 연결부에 맞은 것이 주원인, 그 충격으로 떨어짐"),
    "T082": ("기타", "폭발·파열,맞음", "에어믹서기 폭발이 주원인(기타), 파편에 맞음"),
}


def main():
    s = pd.read_excel(BLIND, sheet_name=None)
    gold = s["라벨링"].merge(s["key"], on="문항ID")
    gold = gold.rename(columns={"정답_유형": "블라인드_유형", "정답_연관유형": "블라인드_연관유형"})
    gold["정답_유형"] = gold["블라인드_유형"]
    gold["정답_연관유형"] = gold["블라인드_연관유형"]
    gold["조정사유"] = ""
    for qid, (t, rel, why) in ADJUST.items():
        i = gold.index[gold["문항ID"] == qid]
        assert len(i) == 1, qid
        gold.loc[i, ["정답_유형", "정답_연관유형", "조정사유"]] = [t, rel, why]

    cols = ["문항ID", "번호", "사고내용", "정답_유형", "정답_연관유형",
            "블라인드_유형", "블라인드_연관유형", "조정사유", "메모"]
    gold[cols].to_csv(GOLD, index=False, encoding="utf-8-sig")
    print(f"저장: {GOLD.relative_to(ROOT)} ({len(gold)}건, 조정 {len(ADJUST)}건)")
    print("정답 분포:", gold["정답_유형"].value_counts().to_dict())


if __name__ == "__main__":
    main()
