import os
import asyncio
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from CheckVideo import check_video

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정 (환경 변수로 관리하는 것이 좋음)
API_KEY = os.getenv("OPENAI_API_KEY")
SURVEY_ASSISTANT_ID = os.getenv("OPENAI_SURVEY_ASSISTANT_ID")
ADVICE_ASSISTANT_ID = os.getenv("OPENAI_ADVICE_ASSISTANT_ID")
client = OpenAI(api_key=API_KEY)


# 요청 모델 정의 (여러 개의 일기를 받을 수 있도록 리스트 형태로 수정)
class DiaryRequest(BaseModel):
    diary_content: List[str]


async def poll_run_async(run, thread):
    while run.status != "completed":
        # 디버깅용 출력
        #print(f"Polling run: {run.status}")
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        await asyncio.sleep(0.5)  # 비동기 sleep
    return run


async def create_run_async(thread_id, assistant_id):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    # 디버깅용 출력
    #print(f"Run created: {run.id} with status {run.status}")
    return run


async def delete_thread(thread_id):
    try:
        client.beta.threads.delete(thread_id=thread_id)
        print(f"Thread {thread_id} deleted.")
    except Exception as e:
        print(f"Failed to delete thread {thread_id}: {str(e)}")


# 질문 처리 엔드포인트 (일주일치 일기 분석 후 스레드 삭제)
@app.post("/analyze_diary/")
async def analyze_diary(request: DiaryRequest):
    try:
        # 새로운 thread 생성 (하나의 스레드에서 모든 일기 처리)
        thread = client.beta.threads.create()
        #print(f"New thread created: {thread.id}")

        # 모든 일기에 대해 사용자 메시지 생성
        for diary_content in request.diary_content:
            if not diary_content:
                raise HTTPException(status_code=400, detail="Diary content cannot be empty")

            # 사용자 메시지 생성
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=diary_content
            )
            #print(f"User message created: {message.id}")

        # 비동기적으로 run 생성 및 상태 확인
        run = await create_run_async(thread.id, SURVEY_ASSISTANT_ID)
        completed_run = await poll_run_async(run, thread)

        # 메시지 리스트에서 assistant의 모든 응답 찾기
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        messages_list = list(messages)
        #print(f"Messages List: {messages_list}")  # 디버깅용 출력

        # 모든 assistant 메시지의 일치 항목 수 합산
        total_matches = 0
        for message in messages_list:
            if message.role == "assistant":
                # 디버깅용으로 message.content의 타입과 내용 출력
                #print(f"Message content type: {type(message.content)}, content: {message.content}")

                if isinstance(message.content, list):
                    for block in message.content:
                        print(f"Block: {block}")  # 각 block의 실제 구조 출력

                        # block이 올바른 구조인지 확인 후 처리
                        if hasattr(block, 'text') and hasattr(block.text, 'value'):
                            value = block.text.value
                            #print(f"Extracted value: {value}")  # 추출된 값 출력
                            try:
                                matches = int(value.strip())
                                total_matches += matches
                            except ValueError:
                                raise HTTPException(status_code=500, detail="Invalid response from assistant block")

        # 스레드를 삭제하고 결과 반환
        await delete_thread(thread.id)
        return {"total_matches": total_matches}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 육아 조언에 대한 API 엔드포인트
@app.post("/give_advice/")
async def give_advice(request: DiaryRequest):
    try:
        # 새로운 thread 생성 (하나의 스레드에서 모든 일기 처리)
        thread = client.beta.threads.create()
        #print(f"New thread created: {thread.id}")

        # 모든 일기에 대해 사용자 메시지 생성
        for diary_content in request.diary_content:
            if not diary_content:
                raise HTTPException(status_code=400, detail="Diary content cannot be empty")

            # 사용자 메시지 생성
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=diary_content
            )
            #print(f"User message created: {message.id}")

        # 비동기적으로 run 생성 및 상태 확인
        run = await create_run_async(thread.id, ADVICE_ASSISTANT_ID)
        completed_run = await poll_run_async(run, thread)

        # 메시지 리스트에서 assistant의 모든 응답 찾기
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        messages_list = list(messages)

        # 모든 assistant 메시지에서 text value 값만 추출
        advice_responses = []
        for message in messages_list:
            if message.role == "assistant" and isinstance(message.content, list):
                for block in message.content:
                    advice_responses.append(block.text.value)

        # 스레드를 삭제하고 결과 반환
        await delete_thread(thread.id)
        return {"advice_responses": advice_responses}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analysisVideo")
async def upload_file(file: UploadFile = File(...)):
    upload_path = "upload"

    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    with open(f"{upload_path}/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(file.filename)

    check_video(upload_path + '/' + file.filename)

    return {"result": file.filename}