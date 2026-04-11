# api/historyApi.py
# handles all chat history operations
# create session, get all sessions, get single session,
# delete session, update session title
# think of sessions like articles in blog app

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from database.mongodb import get_db
from database.schemas import ChatSessionSchema, MessageSchema
import uuid
from datetime import datetime

historyRouter = APIRouter()


def now():
    """Returns current UTC timestamp as string."""
    return datetime.utcnow().isoformat()


# ── GENERATE SMART TITLE ───────────────────────────────────
async def generate_smart_title(first_message: str) -> str:
    """
    Generates a clean 4-6 word title from first message.
    Better than truncating raw text.
    """
    from rag.nodes import llm
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        response = llm.invoke([
            SystemMessage(content="""Generate a short, clear 4-6 word chat title 
            from this student question. No quotes, no punctuation at end.
            Example: 'Newton Laws of Motion' or 'JEE 2023 Maths PYQs'"""),
            HumanMessage(content=first_message[:200])
        ])
        title = response.content.strip().strip('"').strip("'")
        return title[:60] if title else first_message[:50]
    except Exception:
        return first_message[:50]


# ── CREATE NEW SESSION ─────────────────────────────────────

class CreateSessionRequest(BaseModel):
    userId: str
    examTarget: str
    firstMessage: str               # used to auto-generate title



@historyRouter.post("/sessions")
async def createSession(req: CreateSessionRequest):
    """
    Creates a new chat session.
    Generates smart title from first message using LLM.
    Like creating a new article in blog app.
    """
    import logging
    db = get_db()
    # Log the incoming request for debugging
    logging.warning(f"[DEBUG] Incoming createSession request: {req.dict()}")
    # auto-generate smart title from first message
    title = await generate_smart_title(req.firstMessage)
    session = {
        "sessionId": str(uuid.uuid4()),
        "userId": req.userId,
        "title": title,
        "examTarget": req.examTarget,
        "messages": [],
        "createdAt": now(),
        "updatedAt": now(),
        "isActive": True
    }
    await db.sessions.insert_one(session)
    session.pop("_id", None)        # remove MongoDB _id before sending
    print(f"[API: history] Created session '{session['sessionId']}' for user '{req.userId}'")
    return {"message": "session created", "payload": session}


# ── GET ALL SESSIONS FOR USER ──────────────────────────────
@historyRouter.get("/sessions/{userId}")
async def getUserSessions(userId: str):
    """
    Gets all chat sessions for a user — sorted newest first.
    Used to populate the sidebar.
    """
    db = get_db()

    cursor = db.sessions.find(
        {"userId": userId, "isActive": True},
        {"_id": 0}                  # exclude MongoDB _id
    ).sort("updatedAt", -1)         # newest first

    sessions = await cursor.to_list(length=100)

    print(f"[API: history] Found {len(sessions)} sessions for user '{userId}'")
    return {"message": "sessions", "payload": sessions}


# ── GET SINGLE SESSION WITH MESSAGES ──────────────────────
@historyRouter.get("/sessions/{userId}/{sessionId}")
async def getSession(userId: str, sessionId: str):
    """
    Gets a single session with all messages.
    Called when user clicks a session in sidebar.
    """
    db = get_db()

    session = await db.sessions.find_one(
        {"sessionId": sessionId, "userId": userId},
        {"_id": 0}
    )

    if not session:
        return {"message": "session not found", "payload": None}

    print(f"[API: history] Loaded session '{sessionId}'")
    return {"message": "session", "payload": session}


# ── ADD MESSAGE TO SESSION ─────────────────────────────────
class AddMessageRequest(BaseModel):
    sessionId: str
    userId: str
    role: str
    content: str
    sources: List[str] = []


@historyRouter.post("/sessions/message")
async def addMessage(req: AddMessageRequest):
    """
    Adds a message to existing session.
    Called after every user query and assistant response.
    """
    db = get_db()

    message = {
        "role": req.role,
        "content": req.content,
        "sources": req.sources,
        "timestamp": now()
    }

    await db.sessions.update_one(
        {"sessionId": req.sessionId, "userId": req.userId},
        {
            "$push": {"messages": message},
            "$set": {"updatedAt": now()}
        }
    )

    print(f"[API: history] Added {req.role} message to session '{req.sessionId}'")
    return {"message": "message added", "payload": message}


# ── DELETE SESSION (SOFT DELETE) ───────────────────────────
@historyRouter.delete("/sessions/{userId}/{sessionId}")
async def deleteSession(userId: str, sessionId: str):
    """
    Soft deletes a session — sets isActive=False.
    Like soft delete in your blog app articles.
    """
    db = get_db()

    await db.sessions.update_one(
        {"sessionId": sessionId, "userId": userId},
        {"$set": {"isActive": False, "updatedAt": now()}}
    )

    print(f"[API: history] Deleted session '{sessionId}'")
    return {"message": "session deleted"}


# ── UPDATE SESSION TITLE ───────────────────────────────────
class UpdateTitleRequest(BaseModel):
    title: str


@historyRouter.put("/sessions/{userId}/{sessionId}/title")
async def updateTitle(userId: str, sessionId: str, req: UpdateTitleRequest):
    """Updates the title of a chat session."""
    db = get_db()

    await db.sessions.update_one(
        {"sessionId": sessionId, "userId": userId},
        {"$set": {"title": req.title, "updatedAt": now()}}
    )

    return {"message": "title updated"}