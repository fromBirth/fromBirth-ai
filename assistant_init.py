import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI API 키 설정 (환경 변수로 관리하는 것이 좋음)
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# 신경발달장애 설문 항목
survey_items = [
    "아이가 눈을 맞추지 않는다",
    "이름을 불러도 반응하지 않는다",
    "장난감에 지나치게 집착한다",
    "어떤 소리에 지나치게 예민하다",
    "반복적인 행동을 한다"
]

# survey_items을 한 문장으로 연결
survey_items_str = ', '.join(survey_items)

# instruction에 survey_items을 포함한 문자열 작성
SURVEY_INSTRUCTIONS = (
    "You are an assistant that helps analyze parenting diaries written in Korean. "
    "You will receive multiple diary entries, and you must analyze each diary entry independently. "
    "For each diary entry, compare its content with a predefined list of survey items related to neurodevelopmental disorders. "
    "These survey items include: " + survey_items_str + ". "
    "Count the number of survey items that are mentioned or clearly implied in the diary entry. "
    "Each survey item should only be counted once per diary entry, even if it is mentioned multiple times or implied in different ways. "
    "Do not count the same survey item more than once in a single diary entry. "
    "After analyzing all the diary entries, return the total number of matches across all diary entries as a single integer. "
    "Do not provide any explanations, context, or reasoning. "
    "Do not return any other text or symbols, only the integer representing the total number of matches."
)

ADVICE_INSTRUCTIONS = (
    "You are an assistant that provides feedback and advice based on parenting diaries written in Korean. "
    "You will receive multiple diary entries, each written in Korean, describing the experiences of a parent over the course of one week. "
    "Analyze the entire week’s entries as a whole and identify key areas for advice. "
    "Instead of describing each diary entry or mentioning specific events in detail, focus on providing concise and practical advice that addresses the overall themes and concerns. "
    "Do not summarize the diary entries or mention them in a sequential manner. "
    "Your feedback should be focused, helpful, and constructive, addressing any concerns while offering encouragement. "
    "All responses must be in Korean."
)

# Assistant 생성
surveyAssistant = client.beta.assistants.create(  # OpenAI API를 사용하여 Assistant를 생성
    name="Diary Survey Assistant",
    instructions=SURVEY_INSTRUCTIONS,
    model="gpt-3.5-turbo"
)

adviceAssistant = client.beta.assistants.create(  # OpenAI API를 사용하여 Assistant를 생성
    name="Diary Advice Assistant",
    instructions=ADVICE_INSTRUCTIONS,
    model="gpt-3.5-turbo"
)

# 생성된 어시스턴트의 ID 값을 출력
surveyAssistant_id = surveyAssistant.id
print(f"Survey Assistant ID: {surveyAssistant_id}")

adviseAssistant_id = adviceAssistant.id
print(f"Advice Assistant ID: {adviseAssistant_id}")
