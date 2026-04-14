# api/studyPlannerApi.py
# generates personalized study timetables using RAG data
# fetches syllabus coverage, chapter weightage from vector store
# returns structured day-by-day plan

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from rag.nodes import llm, vector_store
from database.mongodb import get_db
from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_ollama import ChatOllama
# from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from langchain_groq import ChatGroq
from config import GROQ_API_KEY
from datetime import datetime, date
import json, re, uuid

studyPlannerRouter = APIRouter()

# dedicated LLM for planner — needs more tokens than the shared one
# study plans are long structured JSON, so we need higher num_predict and num_ctx
# planner_llm = ChatOllama(
#     base_url=OLLAMA_BASE_URL,
#     model=OLLAMA_MODEL,
#     temperature=0.1,
#     num_predict=2048,       # study plan JSON can be 1500+ tokens
#     num_ctx=4096,           # larger context for prompt + response
#     repeat_penalty=1.1,
#     top_k=20,
#     top_p=0.9,
# )
planner_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=2048,
)

def now():
    return datetime.utcnow().isoformat() + "Z"


class StudyPlanRequest(BaseModel):
    userId: str
    examTarget: str
    examDate: str              # "2025-05-15"
    weakSubjects: List[str]
    dailyHours: int            # hours available per day
    currentLevel: str          # beginner | intermediate | advanced


class TaskUpdateRequest(BaseModel):
    completed: bool


@studyPlannerRouter.post("/study-planner/generate")
async def generateStudyPlan(req: StudyPlanRequest):
    """
    Generates a personalized study plan using:
    1. Syllabus from vector store
    2. Chapter weightage data
    3. User's weak subjects and exam date
    """
    db = get_db()

    # calculate days remaining
    exam_date = datetime.strptime(req.examDate, "%Y-%m-%d")
    days_remaining = (exam_date - datetime.now()).days

    if days_remaining <= 0:
        return {"message": "exam date passed", "payload": None}

    print(f"[API: planner] Generating plan — {days_remaining} days, {req.dailyHours}h/day")

    # retrieve syllabus/weightage context from vector store
    results = vector_store.query(
        query_text=f"{req.examTarget} syllabus chapter weightage important topics",
        top_k=6
    )
    context = "\n\n".join([
        r["metadata"].get("text", "")
        for r in results if r["metadata"]
    ])

    # build prompt
    system_prompt = f"""You are an expert {req.examTarget} study planner with 10 years experience.
Generate a realistic, optimized study timetable as JSON.
Return ONLY valid JSON, no other text.

JSON Structure:
{{
  "planId": "unique_id",
  "title": "Study Plan Title",
  "totalDays": number,
  "dailyHours": number,
  "phases": [
    {{
      "phase": "Phase name",
      "days": "Day 1-15",
      "focus": "What to focus on",
      "subjects": ["subject1", "subject2"]
    }}
  ],
  "weeklySchedule": {{
    "Monday": [{{"time": "9-11 AM", "subject": "Physics", "topic": "topic name"}}],
    "Tuesday": [...],
    "Wednesday": [...],
    "Thursday": [...],
    "Friday": [...],
    "Saturday": [...],
    "Sunday": [...]
  }},
  "priorityChapters": [
    {{"subject": "Physics", "chapter": "chapter name", "weightage": "high|medium|low", "days": 3}}
  ],
  "milestones": [
    {{"day": 15, "target": "Complete all Physics basics"}}
  ],
  "dailyChecklist": [
    "Study for {req.dailyHours} hours",
    "Solve 10 PYQs",
    "Revise yesterday's topics",
    "Practice formulas for 15 mins"
  ]
}}"""

    user_message = f"""Create a {req.examTarget} study plan:
- Days remaining: {days_remaining}
- Daily study hours: {req.dailyHours}
- Weak subjects: {', '.join(req.weakSubjects)}
- Current level: {req.currentLevel}
- Syllabus context: {context[:1000]}

Prioritize weak subjects. Include revision days. Last 7 days = full revision only."""

    try:
        plan_data = invoke_and_parse_json(system_prompt, user_message)

        if not plan_data:
            return {"message": "generation failed", "payload": None}

        plan_data["planId"] = str(uuid.uuid4())
        plan_data["userId"] = req.userId
        plan_data["examDate"] = req.examDate
        plan_data["daysRemaining"] = days_remaining
        plan_data["createdAt"] = now()
        plan_data["taskProgress"] = {}  # tracks completed tasks

        # save to MongoDB
        await db.studyplans.update_one(
            {"userId": req.userId},
            {"$set": plan_data},
            upsert=True
        )

        plan_data.pop("_id", None)
        print(f"[API: planner] Plan generated — {days_remaining} days")
        return {"message": "plan generated", "payload": plan_data}

    except Exception as e:
        print(f"[API: planner] Error: {e}")
        return {"message": "error", "payload": None}


def repair_json(text: str) -> str:
    """Attempts to repair common LLM JSON mistakes."""
    # strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    # remove trailing commas before } or ]
    text = re.sub(r',\s*([\]}])', r'\1', text)
    # replace single quotes with double quotes (careful approach)
    # only if there are no double quotes already
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    return text


def extract_json_from_text(text: str) -> dict | None:
    """
    Robust JSON extraction from LLM output.
    Tries multiple strategies in order of reliability.
    """
    # Strategy 1: direct parse of the full text
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # Strategy 2: strip markdown fences and retry
    cleaned = repair_json(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Strategy 3: find the outermost { ... } with brace matching
    start = cleaned.find('{')
    if start == -1:
        return None

    depth = 0
    end = start
    for i in range(start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = cleaned[start:end]
    try:
        return json.loads(json_str)
    except Exception:
        pass

    # Strategy 4: repair and retry the extracted block
    repaired = repair_json(json_str)
    try:
        return json.loads(repaired)
    except Exception:
        return None


def invoke_and_parse_json(system_prompt: str, user_message: str, max_retries: int = 2) -> dict | None:
    """
    Invokes LLM and robustly parses JSON, retrying with stricter prompts on failure.
    """
    for attempt in range(max_retries):
        if attempt > 0:
            # on retry, add stricter instruction
            extra = "\n\nIMPORTANT: Your previous response had invalid JSON. Return ONLY a valid JSON object. No markdown, no explanation, no trailing commas. Just pure JSON."
            messages = [
                SystemMessage(content=system_prompt + extra),
                HumanMessage(content=user_message)
            ]
            print(f"[API: planner] Retry {attempt} — requesting cleaner JSON...")
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]

        response = planner_llm.invoke(messages)
        result = extract_json_from_text(response.content)

        if result:
            print(f"[API: planner] JSON parsed successfully on attempt {attempt + 1}.")
            return result
        else:
            print(f"[API: planner] JSON parse failed on attempt {attempt + 1}. Raw output: {response.content[:200]}...")

    return None


@studyPlannerRouter.get("/study-planner/{userId}")
async def getStudyPlan(userId: str):
    """Gets existing study plan for user."""
    db = get_db()
    plan = await db.studyplans.find_one({"userId": userId}, {"_id": 0})
    if not plan:
        return {"message": "no plan", "payload": None}
    return {"message": "plan", "payload": plan}


@studyPlannerRouter.post("/study-planner/{userId}/task/{taskId}")
async def updateTask(userId: str, taskId: str, req: TaskUpdateRequest):
    """Marks a checklist task as complete/incomplete."""
    db = get_db()
    await db.studyplans.update_one(
        {"userId": userId},
        {"$set": {f"taskProgress.{taskId}": req.completed}}
    )
    return {"message": "task updated"}