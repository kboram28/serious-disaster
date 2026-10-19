"""2-2단계: 평가용 테스트셋 분리 + 블라인드 라벨링 시트 생성

입력  data/processed/cases_labeled.csv
출력  data/processed/split.csv          번호별 db/test 구분 (V1·V2 모두 test는 DB에 넣지 않음)
      eval/test_blind.xlsx              사람이 원본·GPT 라벨을 보지 않고 분류하는 시트

테스트 100건은 유형_검증 비율대로 층화 추출하고, 시드를 고정해 언제 다시 실행해도 같은 결과가 나온다.
"""
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/cases_labeled.csv"
SPLIT = ROOT / "data/processed/split.csv"
BLIND = ROOT / "eval/test_blind.xlsx"

N_TEST = 100
SEED = 42
TYPES = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘", "기타"]
RELATED = ["떨어짐", "끼임", "깔림", "맞음", "부딪힘", "무너짐", "넘어짐", "감전", "폭발·파열",
           "화재", "화상·이상온도", "질식·중독", "익사", "찔림", "기타"]

GUIDE = [
    ("기본 원칙", "사람이 무엇에 의해 다치거나 사망했는가(결과)로 분류한다. 물체에 일어난 일(쓰러짐, 붕괴)로 분류하지 않는다."),
    ("떨어짐", "사람이 높은 곳(비계, 지붕, 사다리, 개구부, 고소작업대 등)에서 떨어짐"),
    ("끼임", "기계·설비·장비의 움직이는 부분이나 물체 사이에 신체가 끼이거나 말려 들어감"),
    ("깔림", "물체·구조물·장비가 쓰러지거나 넘어져 사람이 그 아래에 깔림, 차량 바퀴에 깔림"),
    ("맞음", "날아오거나 위에서 떨어지는 물체(인양물, 낙하물, 파편 등)에 사람이 맞음"),
    ("부딪힘", "움직이는 차량·장비에 사람이 부딪히거나, 이동 중인 사람이 물체에 부딪힘"),
    ("기타", "위 5종이 아닌 경우: 매몰(토사·굴착면·건물 붕괴), 감전, 폭발, 화재, 화상, 질식, 익사, 넘어짐 등"),
    ("규칙 1", "물체가 쓰러져 사람이 깔림 → 깔림 / 토사·굴착면·건물 붕괴로 매몰 → 기타(연관: 무너짐)"),
    ("규칙 2", "구조물 붕괴로 사람이 떨어짐 → 떨어짐(연관: 무너짐)"),
    ("규칙 3", "두 가지 이상이면 사람이 다치거나 사망한 가장 큰 원인"),
    ("규칙 4", "문장에 없는 내용은 추측하지 않는다 (예: '사다리 위에서 감전되어 사망' → 기타(연관: 감전))"),
    ("연관유형", "결과에 함께 관여한 원인(계기·후속). 기타면 구체 유형(감전 등). 따로 없으면 정답_유형과 같은 값"),
]


def main():
    df = pd.read_csv(SRC)

    test = pd.concat([
        g.sample(round(N_TEST * len(g) / len(df)), random_state=SEED)
        for _, g in df.groupby("유형_검증")
    ])
    # 반올림 오차로 100건이 안 맞으면 가장 많은 유형에서 조정
    diff = N_TEST - len(test)
    if diff > 0:
        extra = df[~df["번호"].isin(test["번호"]) & (df["유형_검증"] == "떨어짐")].sample(diff, random_state=SEED)
        test = pd.concat([test, extra])
    elif diff < 0:
        test = test.drop(test[test["유형_검증"] == "떨어짐"].sample(-diff, random_state=SEED).index)

    df["split"] = df["번호"].isin(test["번호"]).map({True: "test", False: "db"})
    df[["번호", "split"]].to_csv(SPLIT, index=False, encoding="utf-8-sig")

    # 블라인드 시트: 순서를 섞고 번호 대신 문항ID만 보여줌 (원본·GPT 라벨 없음)
    t = test.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sheet = pd.DataFrame({
        "문항ID": [f"T{i + 1:03d}" for i in range(len(t))],
        "사고내용": t["사고내용"],
        "정답_유형": "",
        "정답_연관유형": "",
        "메모": "",
    })
    key = pd.DataFrame({"문항ID": sheet["문항ID"], "번호": t["번호"]})

    BLIND.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(BLIND, engine="openpyxl") as xw:
        sheet.to_excel(xw, index=False, sheet_name="라벨링")
        pd.DataFrame(GUIDE, columns=["항목", "설명"]).to_excel(xw, index=False, sheet_name="분류기준")
        key.to_excel(xw, index=False, sheet_name="key")

        ws = xw.sheets["라벨링"]
        ws.freeze_panes = "C2"
        for col, w in zip("ABCDE", (8, 90, 11, 13, 30)):
            ws.column_dimensions[col].width = w
        edit = PatternFill("solid", fgColor="DDEBF7")
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for r in range(2, ws.max_row + 1):
            ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top")
            for c in (3, 4, 5):
                ws.cell(r, c).fill = edit
        for col, values in (("C", TYPES), ("D", RELATED)):
            dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"')
            dv.add(f"{col}2:{col}{ws.max_row}")
            ws.add_data_validation(dv)

        g = xw.sheets["분류기준"]
        g.column_dimensions["A"].width = 12
        g.column_dimensions["B"].width = 110
        xw.book["key"].sheet_state = "hidden"   # 채점용 연결표 (라벨링 중에는 보지 않음)

    print(f"split: db {sum(df['split'] == 'db')} / test {sum(df['split'] == 'test')}")
    print("test 유형 분포:", test["유형_검증"].value_counts().to_dict())
    print(f"블라인드 시트: {BLIND.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
