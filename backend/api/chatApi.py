# api/chatApi.py
# main chat endpoint — runs agentic RAG pipeline
# saves every message to MongoDB for persistence
# supports stop signal via AbortController from frontend

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from rag.pipeline import run_rag_pipeline
from database.mongodb import get_db
import uuid
from datetime import datetime

chatRouter = APIRouter()


def now():
    return datetime.utcnow().isoformat()


class ChatRequest(BaseModel):
    query: str
    userId: str
    examTarget: str
    sessionId: str    
    
    # which session this message belongs to


@chatRouter.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    1. Runs agentic RAG pipeline
    2. Saves user message to MongoDB
    3. Saves assistant response to MongoDB
    4. Returns answer + sources
    """
    db = get_db()

    print(f"[API: chat] Query from '{request.userId}': '{request.query[:60]}...'")
    print(f"[API: chat] Session: {request.sessionId} | Exam: {request.examTarget}")

    # save user message first
    user_message = {
        "role": "user",
        "content": request.query,
        "sources": [],
        "timestamp": now()
    }

    await db.sessions.update_one(
        {"sessionId": request.sessionId, "userId": request.userId},
        {
            "$push": {"messages": user_message},
            "$set": {"updatedAt": now()}
        }
    )

    # run RAG pipeline
    result = run_rag_pipeline(request.query)

    # save assistant response
    assistant_message = {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "timestamp": now()
    }

    await db.sessions.update_one(
        {"sessionId": request.sessionId, "userId": request.userId},
        {
            "$push": {"messages": assistant_message},
            "$set": {"updatedAt": now()}
        }
    )
    

    print(f"[API: chat] Response saved to session '{request.sessionId}'")

    return {
        "message": "success",
        "payload": {
            "answer": result["answer"],
            "sources": result["sources"]
        }
    }