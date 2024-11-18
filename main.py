import os
import asyncio
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from CheckVideo import check_video
from Opticalflow import getvideoresult

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

@app.post("/analysisVideo")
async def upload_file(file: UploadFile = File(...)):
    upload_path = "upload"

    if not os.path.exists(upload_path):
        os.makedirs(upload_path)
    with open(f"{upload_path}/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(file.filename)

    check_video(upload_path + '/' + file.filename)

    return getvideoresult(upload_path+'/'+file.filename)