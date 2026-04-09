# api/chatApi.py
# FastAPI router for chat endpoints
# receives query from React frontend → runs RAG pipeline → returns answer

from fastapi import APIRouter
from pydantic import BaseModel
from rag.pipeline import run_rag_pipeline

chatRouter = APIRouter()

# request schema — what frontend sends
class ChatRequest(BaseModel):
    query: str
    userId: str
    examTarget: str       # "JEE" or "NEET"

# POST /api/chat — main chat endpoint
@chatRouter.post("/chat")
async def chat(request: ChatRequest):
    print(f"[API: chat] Received query from user '{request.userId}': '{request.query}'")
    print(f"[API: chat] Exam target: {request.examTarget}")

    # run agentic RAG pipeline
    result = run_rag_pipeline(request.query)

    return {
        "message": "success",
        "payload": {
            "answer": result["answer"],
            "sources": result["sources"]
        }
    }