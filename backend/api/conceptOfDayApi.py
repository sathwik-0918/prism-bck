# api/conceptOfDayApi.py
# Generates one high-value JEE/NEET concept daily
# Cached in MongoDB for 24 hours — same concept for all users on same day
# Uses Ollama main_llm for high quality generation

from fastapi import APIRouter
from pydantic import BaseModel
from database.mongodb import get_db
from rag.nodes import main_llm, vector_store
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime, date
import json, re
from api.leaderboardApi import update_leaderboard_points

conceptRouter = APIRouter()

def now():
    return datetime.utcnow().isoformat() + "Z"

# curated high-value topics that rotate
JEE_TOPICS = [
    "Newton's Laws of Motion",
    "Gauss's Law in Electrostatics",
    "Thermodynamics First Law",
    "Simple Harmonic Motion",
    "Binomial Theorem",
    "Chemical Equilibrium",
    "Wave Optics — Interference",
    "Rotational Mechanics",
    "Organic Chemistry — Mechanisms",
    "Electromagnetic Induction",
    "Complex Numbers",
    "p-Block Elements",
    "Integration Techniques",
    "Projectile Motion",
    "Electrochemistry",
    "Coordinate Geometry — Ellipse",
    "Modern Physics — Photoelectric Effect",
    "Chemical Bonding",
    "Limits and Continuity",
    "Magnetic Effects of Current",
]

NEET_TOPICS = [
    "Cell Division — Mitosis vs Meiosis",
    "Photosynthesis — Calvin Cycle",
    "Human Circulatory System",
    "Genetics — Mendel's Laws",
    "Ecosystem and Food Chains",
    "Human Nervous System",
    "DNA Replication",
    "Respiration — Krebs Cycle",
    "Plant Hormones",
    "Biotechnology — PCR",
    "Excretory System",
    "Reproductive System",
    "Evolution — Natural Selection",
    "Digestion and Absorption",
    "Coordination — Endocrine System",
]


def safe_json_parse(text: str) -> dict:
    """
    Parses LLM JSON output that may contain LaTeX backslashes.
    Tries multiple strategies before giving up.
    """
    # strategy 1 — clean markdown fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # strategy 2 — direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # strategy 3 — find outermost braces
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        candidate = text[start:end]
        return json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        pass

    # strategy 4 — escape unescaped backslashes in string values
    # LaTeX like \omega \frac \sqrt cause JSONDecodeError
    try:
        # replace \letter patterns with \\letter (valid JSON escape)
        import re as _re
        fixed = _re.sub(
            r'(?<!\\)\\(?!["\\/bfnrtu])',  # backslash not followed by JSON-valid escape
            r'\\\\',
            text[start:end] if 'start' in dir() else text
        )
        return json.loads(fixed)
    except Exception:
        pass

    # strategy 5 — use ast.literal_eval on cleaned text
    try:
        import ast
        # replace true/false/null with Python equivalents
        py_text = text.replace('true', 'True').replace('false', 'False').replace('null', 'None')
        return ast.literal_eval(py_text)
    except Exception:
        pass

    print(f"[ConceptOfDay] All JSON parse strategies failed.")
    return None


async def generate_concept_question(topic: str, exam_target: str, context: str) -> dict:
    """
    Generates one MCQ — uses plain text formulas to avoid LaTeX JSON issues.
    """
    system_prompt = f"""You are a {exam_target} question setter.
Generate ONE medium MCQ on '{topic}' requiring application, not just recall.

CRITICAL RULES for JSON:
- Do NOT use LaTeX backslashes (no \\omega, \\frac etc.) inside JSON strings
- Write formulas as plain text: "omega", "v^2/r", "F = ma"
- Return ONLY valid JSON, nothing else

Format:
{{
  "question": "question text with plain text formulas only",
  "options": {{"A": "option", "B": "option", "C": "option", "D": "option"}},
  "answer": "A",
  "explanation": "clear explanation, no LaTeX backslashes"
}}"""

    try:
        response = main_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {topic}\nContext (use for accuracy): {context[:1000]}")
        ])
        result = safe_json_parse(response.content)
        if result and "question" in result:
            return result
    except Exception as e:
        print(f"[ConceptOfDay] Question error: {e}")
    return None


async def generate_concept(topic: str, exam_target: str) -> dict:
    """
    Generates concept — uses plain text to avoid JSON escape issues.
    """
    results = vector_store.query(
        query_text=f"{exam_target} {topic} explanation formulas examples",
        top_k=6
    )
    context = "\n\n".join([
        r["metadata"].get("text", "")
        for r in results if r["metadata"]
    ])[:3000]

    system_prompt = f"""You are Prism — best {exam_target} concept explainer.
Generate a concept explanation. 

CRITICAL: Do NOT use LaTeX backslashes in JSON strings.
Write formulas as plain text: "v = u + at", "F = ma", "E = mc^2"
Return ONLY valid JSON, no markdown, no extra text.

{{
  "topic": "{topic}",
  "tagline": "one catchy sentence",
  "whyImportant": "2-3 sentences on exam importance",
  "coreIdea": "simple 2-3 line explanation",
  "keyPoints": ["point 1", "point 2", "point 3", "point 4"],
  "formulas": [
    {{"name": "Formula name", "formula": "plain text formula like F = ma", "meaning": "what it means"}}
  ],
  "mnemonic": "memory trick",
  "commonMistakes": ["mistake 1", "mistake 2"],
  "pyqs": [
    {{"year": "2023", "question": "question text", "answer": "answer and brief explanation"}}
  ],
  "difficulty": "medium",
  "estimatedMarks": "4-8 marks"
}}"""

    try:
        response = main_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create concept for: {topic}\nContext:\n{context[:2000]}")
        ])

        result = safe_json_parse(response.content)
        if result:
            # generate question separately
            question = await generate_concept_question(topic, exam_target, context)
            if question:
                result["dailyQuestion"] = question
            return result
    except Exception as e:
        print(f"[ConceptOfDay] Concept error: {e}")

    # fallback — minimal valid concept
    return {
        "topic": topic,
        "tagline": f"Master {topic} for {exam_target}",
        "whyImportant": f"{topic} is an important topic in {exam_target}.",
        "coreIdea": "Loading details...",
        "keyPoints": [f"Study {topic} thoroughly"],
        "formulas": [],
        "mnemonic": "",
        "commonMistakes": [],
        "pyqs": [],
        "difficulty": "medium",
        "estimatedMarks": "4-8 marks"
    }


@conceptRouter.get("/concept-of-day/{exam_target}")
async def getConceptOfDay(exam_target: str):
    """
    Returns today's concept. Same for all users on same day.
    Auto-generates if expired or missing.
    """
    db = get_db()
    today = date.today().isoformat()

    # check if today's concept exists
    existing = await db.conceptofday.find_one(
        {"date": today, "examTarget": exam_target},
        {"_id": 0}
    )

    if existing:
        print(f"[ConceptOfDay] Serving cached concept for {today}")
        return {"message": "concept", "payload": existing}

    # generate new concept
    print(f"[ConceptOfDay] Generating new concept for {today} — {exam_target}")

    # pick topic based on day of year
    day_of_year = date.today().timetuple().tm_yday
    if exam_target == "NEET":
        topic = NEET_TOPICS[day_of_year % len(NEET_TOPICS)]
    else:
        topic = JEE_TOPICS[day_of_year % len(JEE_TOPICS)]

    concept = await generate_concept(topic, exam_target)
    concept["date"] = today
    concept["examTarget"] = exam_target
    concept["generatedAt"] = now()

    # save to MongoDB
    await db.conceptofday.insert_one(concept)

    # return without _id
    concept_out = {k: v for k, v in concept.items() if k != "_id"}
    return {"message": "concept", "payload": concept_out}


@conceptRouter.get("/concept-of-day/preview/{exam_target}")
async def getConceptPreview(exam_target: str):
    """
    Returns minimal concept data for header badge preview.
    Fast — no full generation.
    """
    db = get_db()
    today = date.today().isoformat()

    existing = await db.conceptofday.find_one(
        {"date": today, "examTarget": exam_target},
        {"_id": 0, "topic": 1, "tagline": 1, "difficulty": 1, "estimatedMarks": 1}
    )

    if existing:
        return {"message": "preview", "payload": existing}

    # trigger generation in background
    return {"message": "generating", "payload": None}

# New endpoints for question attempts:
class ConceptAttemptRequest(BaseModel):
    userId: str
    date: str
    examTarget: str
    selectedAnswer: str
    correctAnswer: str


@conceptRouter.post("/concept-of-day/attempt")
async def recordAttempt(req: ConceptAttemptRequest):
    """
    Records user's attempt at the daily concept question.
    Each user can only attempt once per day.
    """
    db = get_db()

    # check if already attempted
    existing = await db.conceptquestions.find_one({
        "userId": req.userId,
        "date": req.date
    })

    if existing:
        return {"message": "already_attempted", "payload": existing}

    is_correct = req.selectedAnswer == req.correctAnswer
    attempt = {
        "userId": req.userId,
        "date": req.date,
        "examTarget": req.examTarget,
        "selectedAnswer": req.selectedAnswer,
        "correctAnswer": req.correctAnswer,
        "correct": is_correct,
        "attemptedAt": datetime.utcnow().isoformat()
    }

    await db.conceptquestions.insert_one(attempt)
    attempt.pop("_id", None)

    # award leaderboard points
    points = 15 if is_correct else 5
    await update_leaderboard_points(req.userId, "conceptQuestion", points, db)

    # update personalization
    await db.personalization.update_one(
        {"userId": req.userId},
        {"$inc": {"conceptsAttempted": 1, "conceptsCorrect": 1 if is_correct else 0}},
        upsert=True
    )

    print(f"[ConceptOfDay] Attempt recorded — {'correct' if is_correct else 'wrong'}")
    return {"message": "recorded", "payload": attempt}


@conceptRouter.get("/concept-of-day/attempt/{userId}/{date}")
async def getAttempt(userId: str, date: str):
    """Check if user has already attempted today's question."""
    db = get_db()
    attempt = await db.conceptquestions.find_one(
        {"userId": userId, "date": date},
        {"_id": 0}
    )
    return {"message": "attempt", "payload": attempt}