from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    message: str


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
