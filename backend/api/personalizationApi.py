# api/personalizationApi.py
# tracks and stores user learning behavior
# extracts topics, difficulty signals, weak areas from every chat
# ollama uses this profile to personalize responses

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from database.mongodb import get_db
from rag.nodes import llm
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime

personalizationRouter = APIRouter()

def now():
    return datetime.utcnow().isoformat()


# ── GET USER PROFILE ──────────────────────────────────────
@personalizationRouter.get("/personalization/{userId}")
async def getProfile(userId: str):
    """
    Returns full personalization profile for a user.
    Shown in Personalization section of frontend.
    """
    db = get_db()
    profile = await db.personalization.find_one(
        {"userId": userId}, {"_id": 0}
    )
    if not profile:
        return {"message": "no profile", "payload": None}
    return {"message": "profile", "payload": profile}


# ── ANALYZE AND UPDATE PROFILE FROM CHAT ─────────────────
class AnalyzeRequest(BaseModel):
    userId: str
    examTarget: str
    query: str
    answer: str


@personalizationRouter.post("/personalization/analyze")
async def analyzeAndUpdate(req: AnalyzeRequest):
    """
    Called after every chat message.
    LLM extracts signals from query+answer and updates user profile.
    Signals: topics discussed, difficulty requested, weak areas shown.
    """
    db = get_db()

    print(f"[API: personalization] Analyzing query for user '{req.userId}'")

    # ask LLM to extract learning signals
    system_prompt = """You are a learning behavior analyzer for a JEE/NEET AI tutor.
Analyze the student query and extract learning signals.
Return ONLY valid JSON with these exact keys:
{
  "topics": ["topic1", "topic2"],
  "difficulty_signal": "easy|medium|hard|unknown",
  "needs_basics": true|false,
  "weak_indicators": ["specific weakness if any"],
  "strong_indicators": ["specific strength if any"]
}
Topics should be specific: "Thermodynamics", "Rotational Motion", "Organic Chemistry" etc.
difficulty_signal = "easy" if user says "simple", "basic", "easy way"
difficulty_signal = "hard" if user asks advanced or complex questions
needs_basics = true if user seems confused or asks for step by step"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {req.query}\nAnswer given: {req.answer[:500]}")
        ])

        import json
        import re
        # extract JSON from response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if not json_match:
            return {"message": "analysis failed", "payload": None}

        signals = json.loads(json_match.group())

        # get existing profile
        existing = await db.personalization.find_one({"userId": req.userId})

        if not existing:
            # create new profile
            profile = {
                "userId": req.userId,
                "examTarget": req.examTarget,
                "topicsDiscussed": signals.get("topics", []),
                "weakTopics": signals.get("weak_indicators", []),
                "strongTopics": signals.get("strong_indicators", []),
                "difficultyLevel": signals.get("difficulty_signal", "medium"),
                "needsBasics": signals.get("needs_basics", False),
                "totalQueries": 1,
                "lastActive": now(),
                "createdAt": now()
            }
            await db.personalization.insert_one(profile)
            profile.pop("_id", None)
        else:
            # merge with existing profile
            topics = list(set(
                (existing.get("topicsDiscussed") or []) +
                (signals.get("topics") or [])
            ))[:20]  # keep max 20 topics

            weak = list(set(
                (existing.get("weakTopics") or []) +
                (signals.get("weak_indicators") or [])
            ))[:10]

            strong = list(set(
                (existing.get("strongTopics") or []) +
                (signals.get("strong_indicators") or [])
            ))[:10]

            # update difficulty if signal is clear
            difficulty = existing.get("difficultyLevel", "medium")
            if signals.get("difficulty_signal") != "unknown":
                difficulty = signals.get("difficulty_signal", difficulty)

            update = {
                "topicsDiscussed": topics,
                "weakTopics": weak,
                "strongTopics": strong,
                "difficultyLevel": difficulty,
                "needsBasics": signals.get("needs_basics", existing.get("needsBasics", False)),
                "totalQueries": existing.get("totalQueries", 0) + 1,
                "lastActive": now()
            }

            await db.personalization.update_one(
                {"userId": req.userId},
                {"$set": update}
            )
            profile = {**existing, **update}
            profile.pop("_id", None)

        print(f"[API: personalization] Profile updated — topics: {signals.get('topics', [])}")
        return {"message": "profile updated", "payload": profile}

    except Exception as e:
        print(f"[API: personalization] Error: {e}")
        return {"message": "error", "payload": None}


# ── GET PERSONALIZATION CONTEXT FOR RAG ───────────────────
async def get_user_context(userId: str, db) -> str:
    """
    Returns personalization context string for RAG pipeline.
    Injected into LLM prompt to personalize responses.
    """
    profile = await db.personalization.find_one(
        {"userId": userId}, {"_id": 0}
    )
    if not profile:
        return ""

    context_parts = []

    if profile.get("difficultyLevel") == "easy" or profile.get("needsBasics"):
        context_parts.append("Student prefers simple, step-by-step explanations.")

    if profile.get("difficultyLevel") == "hard":
        context_parts.append("Student is advanced — can handle complex derivations.")

    if profile.get("weakTopics"):
        context_parts.append(f"Student struggles with: {', '.join(profile['weakTopics'][:3])}")

    if profile.get("strongTopics"):
        context_parts.append(f"Student is strong in: {', '.join(profile['strongTopics'][:3])}")

    return " ".join(context_parts)