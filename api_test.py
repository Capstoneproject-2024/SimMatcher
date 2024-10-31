from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
from Extractor import *
from SimilarityMatcher import *

# Type "uvicorn [file name]:app --reload" to start server

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 요청 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

extractor = Extractor()

@app.post("/submit")
# For testing
async def submit_message(request: Request):
    data = await request.json()
    message = data.get("message")
    print("Received message:", message)  # 콘솔에 메시지 출력
    return {"message": f"Received: {message}"}

@app.post("/extractp")
async def extract_keyword(request: Request):
    data = await request.json()
    review = data.get("review")
    keywords = extractor.extract_keyword_string(review)
    print(f"Received review: {review}")
    print(f"Extracted Keywords: {keywords}")
    return {"keywords": keywords}
# 처음에는 Get을 고려했으나 Post가 더 나아보임 (리뷰는 길기 때문에)

@app.get("/extract")
async def extract_keyword(review: str):
    #keywords = review.split(' ')
    keywords = extractor.extract_keyword_string(review)
    print(f"Received review: {review}")
    print(f"Extracted Keywords: {keywords}")
    return {"keywords": keywords}

