# api/chatApi.py
# main chat endpoint — runs agentic RAG pipeline
# saves every message to MongoDB for persistence
# supports stop signal via AbortController from frontend

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from rag.pipeline import run_rag_pipeline
from database.mongodb import get_db
import uuid
from datetime import datetime
import asyncio

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
async def chat(request: ChatRequest, req: Request):
    """
    Chat endpoint with disconnect detection.
    If client disconnects (stop button) — saves partial state and exits.
    """
    import logging
    db = get_db()
    # Log the incoming request for debugging
    logging.warning(f"[DEBUG] Incoming chat request: {request.dict()}")
    # fetch personalization context
    profile = await db.personalization.find_one(
        {"userId": request.userId}, {"_id": 0}
    )
    user_context = ""
    if profile:
        parts = []
        if profile.get("needsBasics"):
            parts.append("Explain simply step by step.")
        if profile.get("difficultyLevel") == "hard":
            parts.append("Student is advanced.")
        if profile.get("weakTopics"):
            parts.append(f"Weak in: {', '.join(profile['weakTopics'][:2])}")
        user_context = " ".join(parts)

    print(f"[API: chat] Query from '{request.userId}': '{request.query[:60]}...'")
    print(f"[API: chat] Session: {request.sessionId} | Exam: {request.examTarget}")

    # save user message immediately
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

    # run pipeline in executor so we can check disconnect
    loop = asyncio.get_event_loop()

    try:
        # run blocking RAG pipeline in thread pool
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                run_rag_pipeline,
                request.query,
                request.examTarget,
                user_context
            ),
            timeout=180.0      # 3 min max — complex queries need more time
        )
    except asyncio.TimeoutError:
        print(f"[API: chat] Pipeline timed out for query: {request.query[:40]}")
        return {"message": "timeout", "payload": {"answer": "I took too long thinking about that. Let me try again — could you rephrase your question more concisely?", "sources": []}}
    except Exception as e:
        print(f"[API: chat] Pipeline error: {e}")
        return {"message": "error", "payload": {"answer": "Something went wrong.", "sources": []}}

    # check if client already disconnected (stop button pressed)
    if await req.is_disconnected():
        print(f"[API: chat] Client disconnected — skipping save.")
        return {"message": "cancelled", "payload": {"answer": "", "sources": []}}

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
    
    # trigger personalization analysis in background
    asyncio.create_task(
        analyze_chat(request.userId, request.examTarget, request.query, result["answer"])
    )

    print(f"[API: chat] Response saved to session '{request.sessionId}'")

    return {
        "message": "success",
        "payload": {
            "answer": result["answer"],
            "sources": result["sources"]
        }
    }


async def analyze_chat(userId, examTarget, query, answer):
    """Background task — updates personalization after every chat."""
    try:
        from api.personalizationApi import analyzeAndUpdate
        from pydantic import BaseModel as BM

        class Req(BM):
            userId: str
            examTarget: str
            query: str
            answer: str

        await analyzeAndUpdate(Req(
            userId=userId,
            examTarget=examTarget,
            query=query,
            answer=answer
        ))
    except Exception as e:
        print(f"[CHAT] Personalization analysis failed silently: {e}")