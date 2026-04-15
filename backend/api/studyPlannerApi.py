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
    # ── NEW FIELDS ──────────────────────────────────────────────
    completedChapters: List[str] = []   # chapters already done
    strongSubjects: List[str] = []      # subjects user is strong in
    targetScore: Optional[str] = None   # "95+ percentile", "AIR < 1000"
    studySessionLength: int = 2         # hours per session (2-3 hrs)
    hasCoaching: bool = False           # attending coaching or self-study
    revisionDaysBuffer: int = 7         # days for pure revision before exam
    priorityTopics: List[str] = []      # topics student wants to focus extra on


class TaskUpdateRequest(BaseModel):
    completed: bool


def build_fallback_plan(req: StudyPlanRequest, days_remaining: int, effective_days: int, total_hours: int) -> dict:
    subjects = req.weakSubjects or ([req.examTarget] if req.examTarget else ["Physics", "Chemistry"])
    first_subject = subjects[0] if subjects else "Physics"
    second_subject = subjects[1] if len(subjects) > 1 else first_subject
    third_subject = subjects[2] if len(subjects) > 2 else second_subject

    return {
        "title": f"{req.examTarget} Smart Study Plan",
        "executiveSummary": f"Focused {req.examTarget} plan balancing weak subjects, revision, and mock practice over {days_remaining} days.",
        "daysRemaining": days_remaining,
        "dailyHours": req.dailyHours,
        "totalStudyHours": total_hours,
        "phases": [
            {
                "phase": "Foundation and Coverage",
                "days": f"Day 1-{max(1, effective_days // 2)}",
                "focus": "Cover high-weightage backlog and strengthen weak areas",
                "subjects": subjects[:3] or [first_subject]
            },
            {
                "phase": "Revision and Testing",
                "days": f"Day {max(2, effective_days // 2 + 1)}-{effective_days}",
                "focus": "Mixed revision, PYQs, and timed tests",
                "subjects": subjects[:3] or [first_subject]
            }
        ],
        "weeklySchedule": {
            "Monday": [
                {"time": "6-8 AM", "subject": first_subject, "topic": "Concept revision"},
                {"time": "7-9 PM", "subject": second_subject, "topic": "Practice questions"}
            ],
            "Tuesday": [
                {"time": "6-8 AM", "subject": second_subject, "topic": "Weak chapter drill"},
                {"time": "7-9 PM", "subject": third_subject, "topic": "Previous year questions"}
            ],
            "Wednesday": [
                {"time": "6-8 AM", "subject": first_subject, "topic": "Formula and notes revision"},
                {"time": "7-9 PM", "subject": second_subject, "topic": "Mixed test practice"}
            ],
            "Thursday": [
                {"time": "6-8 AM", "subject": third_subject, "topic": "Important examples"},
                {"time": "7-9 PM", "subject": first_subject, "topic": "Numerical/problem solving"}
            ],
            "Friday": [
                {"time": "6-8 AM", "subject": second_subject, "topic": "NCERT and short notes"},
                {"time": "7-9 PM", "subject": third_subject, "topic": "Topic-wise PYQs"}
            ],
            "Saturday": [
                {"time": "9-12 AM", "subject": first_subject, "topic": "Mock test / sectional test"},
                {"time": "5-7 PM", "subject": second_subject, "topic": "Error analysis"}
            ],
            "Sunday": [
                {"time": "8-10 AM", "subject": first_subject, "topic": "Revision backlog"},
                {"time": "6-8 PM", "subject": third_subject, "topic": "Light revision and planning"}
            ]
        },
        "priorityChapters": [
            {"subject": first_subject, "chapter": req.priorityTopics[0] if req.priorityTopics else "High-weightage topic", "weightage": "high", "days": 3},
            {"subject": second_subject, "chapter": req.completedChapters[0] if req.completedChapters else "Revision chapter", "weightage": "medium", "days": 2}
        ],
        "milestones": [
            {"day": max(7, effective_days // 4), "target": "Complete first revision cycle"},
            {"day": max(14, effective_days // 2), "target": "Finish one full mock + analysis"},
            {"day": max(21, effective_days - req.revisionDaysBuffer), "target": "Enter final revision phase"}
        ],
        "dailyChecklist": [
            "Revise one concept block from notes",
            "Solve timed practice questions",
            "Review mistakes and update short notes",
            "Do one revision slot before sleep"
        ],
        "taskProgress": {}
    }


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
    ])[:2000]

    # study hours calculation
    effective_days = max(1, days_remaining - req.revisionDaysBuffer)
    total_hours = effective_days * req.dailyHours
    sessions_per_day = max(1, req.dailyHours // req.studySessionLength)

    # build prompt
    system_prompt = f"""You are India's top {req.examTarget} study planner with 15 years experience.
Create the most accurate and personalized study timetable possible.
Return ONLY valid JSON — no backslashes in values.

Student Profile:
- Days remaining: {days_remaining} (including {req.revisionDaysBuffer} days pure revision)
- Daily hours: {req.dailyHours}h ({sessions_per_day} sessions of {req.studySessionLength}h each)
- Level: {req.currentLevel}
- Weak subjects: {', '.join(req.weakSubjects) or 'None specified'}
- Strong subjects: {', '.join(req.strongSubjects) or 'None specified'}
- Already completed: {', '.join(req.completedChapters) or 'Nothing yet'}
- Priority topics: {', '.join(req.priorityTopics) or 'Standard'}
- Target: {req.targetScore or 'Best performance'}
- Has coaching: {'Yes (plan home study only)' if req.hasCoaching else 'No (full self-study)'}

JSON Structure:
{{
  "planTitle": "Personalized title",
  "executiveSummary": "2-3 line overview of the strategy",
  "totalDays": {days_remaining},
  "effectiveStudyDays": {effective_days},
  "revisionDays": {req.revisionDaysBuffer},
  "totalStudyHours": {total_hours},
  "phases": [
    {{
      "phase": "Phase name",
      "days": "Day 1-{effective_days//3}",
      "focus": "What to achieve",
      "strategy": "How to approach this phase",
      "subjects": ["subject1"],
      "hoursPerSubject": {{"Physics": 2, "Chemistry": 1}}
    }}
  ],
  "weeklySchedule": {{
    "Monday": [
      {{"time": "9-11 AM", "subject": "Physics", "topic": "Specific topic", "type": "new learning"}},
      {{"time": "11-12 PM", "subject": "Chemistry", "topic": "Revision", "type": "revision"}},
      {{"time": "7-9 PM", "subject": "Maths", "topic": "Practice", "type": "practice"}}
    ],
    "Tuesday": [...],
    "Wednesday": [...],
    "Thursday": [...],
    "Friday": [...],
    "Saturday": [...],
    "Sunday": [...]
  }},
  "subjectStrategy": {{
    "Physics": "Subject-specific strategy",
    "Chemistry": "Subject-specific strategy"
  }},
  "priorityChapters": [
    {{
      "subject": "Physics",
      "chapter": "Chapter name",
      "weightage": "high",
      "daysToAllocate": 3,
      "whyImportant": "Reason"
    }}
  ],
  "milestones": [
    {{"day": 15, "target": "Milestone description", "checkMethod": "How to verify"}}
  ],
  "dailyChecklist": [
    "Specific actionable task 1",
    "Specific actionable task 2"
  ],
  "lastMinuteTips": ["Tip for final days"],
  "mockTestSchedule": "Mock test strategy (when, how many)"
}}"""

    try:
        system_prompt = f"""You are India's top {req.examTarget} study planner with 15 years experience.
Create the most accurate and personalized study timetable possible.
Return ONLY valid JSON — no backslashes in values.

Student Profile:
- Days remaining: {days_remaining} (including {req.revisionDaysBuffer} days pure revision)
- Daily hours: {req.dailyHours}h ({sessions_per_day} sessions of {req.studySessionLength}h each)
- Level: {req.currentLevel}
- Weak subjects: {', '.join(req.weakSubjects) or 'None specified'}
- Strong subjects: {', '.join(req.strongSubjects) or 'None specified'}
- Already completed: {', '.join(req.completedChapters) or 'Nothing yet'}
- Priority topics: {', '.join(req.priorityTopics) or 'Standard'}
- Target: {req.targetScore or 'Best performance'}
- Has coaching: {'Yes (plan home study only)' if req.hasCoaching else 'No (full self-study)'}

JSON Structure:
{{
  "planTitle": "Personalized title",
  "executiveSummary": "2-3 line overview of the strategy",
  "totalDays": {days_remaining},
  "effectiveStudyDays": {effective_days},
  "revisionDays": {req.revisionDaysBuffer},
  "totalStudyHours": {total_hours},
  "phases": [
    {{
      "phase": "Phase name",
      "days": "Day 1-{effective_days//3}",
      "focus": "What to achieve",
      "strategy": "How to approach this phase",
      "subjects": ["subject1"],
      "hoursPerSubject": {{"Physics": 2, "Chemistry": 1}}
    }}
  ],
  "weeklySchedule": {{
    "Monday": [
      {{"time": "9-11 AM", "subject": "Physics", "topic": "Specific topic", "type": "new learning"}},
      {{"time": "11-12 PM", "subject": "Chemistry", "topic": "Revision", "type": "revision"}},
      {{"time": "7-9 PM", "subject": "Maths", "topic": "Practice", "type": "practice"}}
    ],
    "Tuesday": [...],
    "Wednesday": [...],
    "Thursday": [...],
    "Friday": [...],
    "Saturday": [...],
    "Sunday": [...]
  }},
  "subjectStrategy": {{
    "Physics": "Subject-specific strategy",
    "Chemistry": "Subject-specific strategy"
  }},
  "priorityChapters": [
    {{
      "subject": "Physics",
      "chapter": "Chapter name",
      "weightage": "high",
      "daysToAllocate": 3,
      "whyImportant": "Reason"
    }}
  ],
  "milestones": [
    {{"day": 15, "target": "Milestone description", "checkMethod": "How to verify"}}
  ],
  "dailyChecklist": [
    "Specific actionable task 1",
    "Specific actionable task 2"
  ],
  "lastMinuteTips": ["Tip for final days"],
  "mockTestSchedule": "Mock test strategy (when, how many)"
}}"""

    except Exception as e:
        print(f"[API: planner] Error: {e}")
        return {"message": "error", "payload": None}

    user_message = f"""Reference material:
{context}

Create a realistic plan for this student. Use the requested JSON structure exactly.
Prioritize {', '.join(req.priorityTopics) if req.priorityTopics else 'high-weightage topics'}.
Respect already completed chapters and coaching/home-study constraints."""

    plan_json = invoke_and_parse_json(system_prompt, user_message, max_retries=3)
    if not plan_json:
        print("[API: planner] Falling back to deterministic plan.")
        plan_json = build_fallback_plan(req, days_remaining, effective_days, total_hours)

    plan = {
        "userId": req.userId,
        "title": plan_json.get("planTitle") or plan_json.get("title") or f"{req.examTarget} Study Plan",
        "executiveSummary": plan_json.get("executiveSummary", ""),
        "daysRemaining": plan_json.get("totalDays", days_remaining),
        "effectiveStudyDays": plan_json.get("effectiveStudyDays", effective_days),
        "revisionDays": plan_json.get("revisionDays", req.revisionDaysBuffer),
        "dailyHours": req.dailyHours,
        "totalStudyHours": plan_json.get("totalStudyHours", total_hours),
        "phases": plan_json.get("phases", []),
        "weeklySchedule": plan_json.get("weeklySchedule", {}),
        "subjectStrategy": plan_json.get("subjectStrategy", {}),
        "priorityChapters": plan_json.get("priorityChapters", []),
        "milestones": plan_json.get("milestones", []),
        "dailyChecklist": plan_json.get("dailyChecklist", []),
        "lastMinuteTips": plan_json.get("lastMinuteTips", []),
        "mockTestSchedule": plan_json.get("mockTestSchedule", ""),
        "taskProgress": plan_json.get("taskProgress", {}),
        "createdAt": now(),
        "updatedAt": now()
    }

    await db.studyplans.update_one(
        {"userId": req.userId},
        {"$set": plan},
        upsert=True
    )

    return {"message": "plan generated", "payload": plan}


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
