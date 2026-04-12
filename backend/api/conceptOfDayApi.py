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


async def generate_concept_question(topic: str, exam_target: str, context: str) -> dict:
    """
    Generates one medium-difficulty MCQ that requires thinking,
    not just direct recall from theory.
    """
    system_prompt = f"""You are an expert {exam_target} question setter.
Generate ONE medium-difficulty MCQ on '{topic}' that requires conceptual thinking and formula application.
The question should NOT be directly from theory — mix concepts with application.
Return ONLY valid JSON:
{{
  "question": "question text here (can include simple LaTeX like $formula$)",
  "options": {{"A": "option text", "B": "option text", "C": "option text", "D": "option text"}},
  "answer": "A",
  "explanation": "clear explanation with why other options are wrong too",
  "difficulty": "medium"
}}"""

    try:
        response = main_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {topic}\nContext: {context[:1500]}\nGenerate one challenging MCQ.")
        ])
        text = response.content.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[ConceptOfDay] Question generation error: {e}")

    return None

async def generate_concept(topic: str, exam_target: str) -> dict:
    """
    Generates a beautiful, structured concept explanation using Ollama.
    Retrieves from vector store first for accurate content.
    """
    # get context from vector store
    results = vector_store.query(
        query_text=f"{exam_target} {topic} explanation formulas examples",
        top_k=6
    )
    context = "\n\n".join([
        r["metadata"].get("text", "")
        for r in results if r["metadata"]
    ])[:3000]

    system_prompt = f"""You are Prism — the best {exam_target} concept explainer in India.
Generate a beautifully structured concept explanation that any student will love to read.
Return ONLY valid JSON — no markdown fences, no extra text.

JSON format:
{{
  "topic": "topic name",
  "tagline": "One catchy sentence about why this topic matters",
  "whyImportant": "2-3 sentences on exam importance and frequency",
  "coreIdea": "The simplest possible explanation of the concept in 2-3 lines",
  "keyPoints": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "formulas": [
    {{"name": "Formula name", "formula": "LaTeX formula here", "meaning": "what each symbol means"}}
  ],
  "mnemonic": "A memorable trick to remember this concept",
  "commonMistakes": ["mistake 1", "mistake 2", "mistake 3"],
  "pyqs": [
    {{"year": "2023", "question": "question text", "answer": "answer with brief explanation"}}
  ],
  "difficulty": "easy|medium|hard",
  "estimatedMarks": "marks this topic typically carries"
}}"""

    user_msg = f"""Create a concept of the day for: {topic}
Exam: {exam_target}

Context from study materials:
{context}

Make it engaging, accurate, and memorable. Include real PYQ questions if available in context."""

    try:
        response = main_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg)
        ])

        text = response.content.strip()
        # clean up any markdown fences
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        # find JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            concept = json.loads(match.group())
            
            # also generate a practice question
            question = await generate_concept_question(topic, exam_target, context)
            if question:
                concept["dailyQuestion"] = question
                
            return concept
    except Exception as e:
        print(f"[ConceptOfDay] Generation error: {e}")

    # fallback
    return {
        "topic": topic,
        "tagline": f"Master {topic} for {exam_target}",
        "coreIdea": "Loading concept details...",
        "keyPoints": [],
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
    concept["generatedAt"] = datetime.utcnow().isoformat()

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