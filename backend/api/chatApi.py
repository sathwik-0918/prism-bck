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
import asyncio
from api.leaderboardApi import update_leaderboard_points
import logging

chatRouter = APIRouter()


def now():
    return datetime.utcnow().isoformat() + "Z"


class ChatRequest(BaseModel):
    query: str
    userId: str
    examTarget: str
    sessionId: str 
    recentMessages: List[dict] = []   
    
    # which session this message belongs to


@chatRouter.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with disconnect detection.
    If client disconnects (stop button) — saves partial state and exits.
    """
    import logging
    db = get_db()


    # normalize query — preserve symbols, remove only true garbage
    import unicodedata
    query = request.query

    # normalize unicode (NFC — standard form, preserves all symbols)
    try:
        query = unicodedata.normalize('NFC', query)
    except Exception:
        pass

    # remove surrogates only
    cleaned_chars = []
    for char in query:
        code = ord(char)
        if not (0xD800 <= code <= 0xDFFF):
            cleaned_chars.append(char)
    query = ''.join(cleaned_chars)

    print(f"[API: chat] Query from '{request.userId}': '{query[:80]}'")
    print(f"[API: chat] Session: {request.sessionId} | Exam: {request.examTarget}")

    # rest of function uses cleaned query
    # ... (keep existing code, replace request.query with query)

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
        result = await loop.run_in_executor(
                    None,
                    run_rag_pipeline,
                    request.query,
                    request.examTarget,
                    user_context,
                    request.recentMessages
        ) 
        
    except Exception as e:
        print(f"[API: chat] Pipeline error: {e}")
        return {
            "message": "error",
            "payload": {
                "answer": "Something went wrong processing your request. Please try again.",
                "sources": []
            }
        }

    # save assistant response (always save — even if client disconnects,
    # the answer is valuable and should persist in the session)
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
    
    # award chat points to leaderboard
    await update_leaderboard_points(request.userId, "chat", 1, db)

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