"""
app_fastapi.py
---------------
FastAPI REST API for the chatbot. Exposes:
    POST /chat   { "message": "..." }  -> { "reply": "..." }
    GET  /health -> { "status": "ok" }

Run:
    uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000

Then test with:
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message": "I am stressed about exams"}'

Or visit the auto-generated docs at http://localhost:8000/docs
"""

from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.logger import logging
from src.exception import ChatbotException
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Mental Health Support Chatbot API",
    description=(
        "A fine-tuned DistilGPT2 chatbot for gentle, empathetic responses. "
        "Not a substitute for professional mental health care."
    ),
    version="1.0.0",
)

pipeline: Optional[PredictionPipeline] = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.on_event("startup")
def load_model():
    global pipeline
    logging.info("Loading prediction pipeline at API startup...")
    pipeline = PredictionPipeline()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message field cannot be empty")

    try:
        reply = pipeline.generate_reply(request.message)
        return ChatResponse(reply=reply)
    except ChatbotException as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Internal error while generating reply")
