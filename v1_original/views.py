from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import AccidentReport
from django.http import JsonResponse
import pandas as pd
import json
import os
import openai

# OpenAI API 키 설정
# [실행 수정 1] 원본은 키를 코드에 직접 적음 → .env 의 OPENAI_API_KEY 에서 읽음 (settings.py 에서 load_dotenv)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 엑셀 파일 로드
# [실행 수정 2] 원본은 팀원 PC의 절대 경로라 다른 PC에서 서버 시작 시 오류 → 이 파일과 같은 폴더
EXCEL_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serious disaster accident.xlsx')
df = pd.read_excel(EXCEL_FILE_PATH)

# 동의어 사전 설정
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
    '떨어짐': ['떨어짐', '추락', '낙하', '낙상'],
    '끼임': ['끼임', '압착', '기계 끼임', '눌림'],
    '부딪힘': ['부딪힘', '충돌', '접촉 사고'],
    '깔림': ['깔림', '매몰', '덮임'],
    '맞음': ['맞음', '타격', '충격', '부상'],
    '기타': ['기타', '알 수 없는 사고', '기타 사고']
}

# 동의어 매핑 함수
def map_synonyms(text, synonyms_dict):
    for key, synonyms in synonyms_dict.items():
        for synonym in synonyms:
            if synonym in text:
                return key
    return text

# 업종과 사고 유형 추출 함수
def extract_keywords(text):
    industry = map_synonyms(text, industry_synonyms)
    disaster = map_synonyms(text, disaster_synonyms)
    return industry, disaster

# 엑셀 데이터를 기반으로 정보 조회
def get_disaster_info(industry, disaster):
    # 업종 및 사고 유형 필터링
    filtered_data = df[
        (df['업종'].str.contains(industry, na=False)) &
        (df['재해 유형'].str.contains(disaster, na=False))
    ]

    if filtered_data.empty:
        disaster_info = f"{industry} 업종에서 '{disaster}' 사고에 대한 구체적인 정보는 찾을 수 없습니다."
        prevention_info = "따라서 예방 대책에 대한 정보도 제공되지 않았습니다."
        return disaster_info, prevention_info

    # 최대 5개의 사고 사례 반환
    case_count = min(len(filtered_data), 5)
    limited_data = filtered_data.head(case_count)

    # 사고 사례 메시지
    disaster_info = f"{industry}에서 발생한 '{disaster}' 사고 사례는 다음과 같습니다:\n\n"
    prevention_info = f"'{disaster}' 사고에 대한 예방 대책은 다음과 같습니다:\n\n"

    for idx, row in limited_data.iterrows():
        disaster_info += f"{idx + 1}. 사고 내용: {row['사고 내용']}\n"
        prevention_info += f"{idx + 1}. 예방 대책: {row['예방 대책']}\n\n"

    return disaster_info.strip(), prevention_info.strip()



@csrf_exempt
def chatbot_api(request):
    if request.method == 'POST':
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            try:
                # 엑셀 파일 읽기
                data = pd.read_excel(uploaded_file)

                for _, row in data.iterrows():
                    industry = row.get('업종', '알 수 없음')
                    disaster_type = row.get('사고 유형', '알 수 없음')

                    # 중복 확인
                    existing_data = AccidentReport.objects.filter(
                        industry=industry,
                        disaster_type=disaster_type
                    )
                    if not existing_data.exists():
                        # 중복이 아닌 경우 저장
                        AccidentReport.objects.create(
                            file=uploaded_file,
                            description=f"{industry} - {disaster_type}",
                            industry=industry,
                            disaster_type=disaster_type,
                            result=row.to_json()  # 데이터를 JSON 형식으로 저장
                        )

                bot_response = "파일이 업로드되었으며, 중복되지 않은 데이터가 성공적으로 저장되었습니다."
            except Exception as e:
                bot_response = f"파일 처리 중 오류가 발생했습니다: {str(e)}"
            return JsonResponse({'response': bot_response})

        # 텍스트 입력 처리
        # [실행 수정 3] 화면(gpt_chatbot.html)은 FormData 로 보내는데 남은 원본 파일은 JSON 만 읽어 채팅이 오류남.
        #   졸작 당시 고쳤던 부분이지만 고치기 전 파일만 남아 있어 다시 수정 → FormData·JSON 둘 다 처리
        if request.content_type == 'application/json':
            data = json.loads(request.body)
            user_input = data.get('text', '')
            chat_history = data.get('history', [])  # 클라이언트에서 보낸 대화 이력
        else:
            user_input = request.POST.get('text', '')
            try:
                chat_history = json.loads(request.POST.get('history', '[]'))
            except json.JSONDecodeError:
                chat_history = []

        # 키워드 추출
        industry, disaster = extract_keywords(user_input)

        if industry and disaster:
            # 엑셀 데이터에서 사고 정보 및 예방 대책 검색
            disaster_info, prevention_info = get_disaster_info(industry, disaster)

            # GPT 모델 호출을 위한 프롬프트 생성
            prompt = f"다음은 '{industry}' 업종에서 발생한 '{disaster}' 사고 사례 및 예방 대책입니다:\n\n"
            prompt += f"{disaster_info}\n예방 대책: {prevention_info}\n\n이 정보를 간단히 요약해 주세요."
            try:
                # [실행 수정 4] 원본 모델 gpt-4-turbo 는 서비스 종료(deprecated, 2026-11-19 호출 시 확인) → 같은 계열 gpt-4.1-mini
                response = openai.ChatCompletion.create(
                    model="gpt-4.1-mini",
                    messages=chat_history + [{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7,
                )
                bot_response = response['choices'][0]['message']['content'].strip()
            except Exception as e:
                bot_response = f"OpenAI API 호출 중 오류가 발생했습니다: {str(e)}"
        else:
            bot_response = "질문에서 업종이나 사고 유형을 확인할 수 없습니다."

        # 대화 이력에 추가
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": bot_response})

        return JsonResponse({'response': bot_response})

    return JsonResponse({'error': 'Invalid request'}, status=400)


def analyze_data(data):
    """
    업로드된 데이터를 분석하는 함수
    """
    try:
        # 간단한 통계 분석 예제
        summary = data.describe().to_string()
        return f"데이터 요약:\n{summary}"
    except Exception as e:
        return f"데이터 분석 중 오류 발생: {str(e)}"


# 챗봇 페이지 렌더링
def gpt_chatbot_view(request):
    return render(request, 'chatbot/gpt_chatbot.html')
