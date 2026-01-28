from fastapi import FastAPI, UploadFile, File
from backend.models import UploadResponse, ChatRequest, ChatResponse
from backend import rag

app = FastAPI(title="Enterprise RAG API")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    rag.process_pdf(pdf_bytes)
    return UploadResponse(message="PDF indexed successfully")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = rag.chat_with_pdf(req.question)
    return ChatResponse(answer=answer)
