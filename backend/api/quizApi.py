# api/quizApi.py
# generates quizzes from RAG knowledge base
# topic-specific, difficulty-aware quiz generation
# uses Ollama LLM to create MCQ questions

import re
import json
from typing import List
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
from rag.nodes import vector_store
from langchain_groq import ChatGroq
from config import GROQ_API_KEY
# from langchain_ollama import ChatOllama
# from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from langchain_core.messages import SystemMessage, HumanMessage
from database.mongodb import get_db
from api.leaderboardApi import update_leaderboard_points

quizRouter = APIRouter()

def now():
    return datetime.utcnow().isoformat() + "Z"

class QuizResultRequest(BaseModel):
    userId: str
    examTarget: str
    topic: str
    difficulty: str
    totalQuestions: int
    correct: int
    wrong: int
    skipped: int
    scorePercent: int
    weakAreas: List[str] = []
    questions: List[dict] = []      # ← full questions saved
    userAnswers: dict = {} 
    isPYQMode: bool = False          # ← new
    pyqExamType: str = "JEE Mains"   # JEE Mains | JEE Advanced | NEET



# dedicated quiz LLM — needs more tokens than default 512
# 5 questions with options + answers + explanations = ~1500+ tokens
# quiz_llm = ChatOllama(
#     base_url=OLLAMA_BASE_URL,
#     model=OLLAMA_MODEL,
#     temperature=0.3,
#     num_predict=2048,       # enough for 5+ questions
#     num_ctx=4096,           # larger context for chunked content + response
#     repeat_penalty=1.1,
#     top_k=20,
#     top_p=0.9,
# )

quiz_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=4000,
)

# QuizRequest — for GENERATING a quiz (only these fields needed)
class QuizRequest(BaseModel):
    topic: str
    examTarget: str             # JEE or NEET
    difficulty: str = "medium"  # easy / medium / hard
    numQuestions: int = 5
    questionType: str = "mcq"
    isPYQMode: bool = False
    pyqExamType: str = "JEE Mains"

# PYQ source file patterns
PYQ_SOURCES = {
    "JEE Mains": ["2013pyq", "2014pyq", "2015pyq", "2016pyq",
                  "2017pyq", "2018pyq", "2019pyq", "jee main",
                  "mains", "que_17333"],
    "JEE Advanced": ["jee advanced", "iit jee", "advanced", "jee adv",
                     "JEE_ADVANCED", "JEE ADVANCED"],
    "NEET": ["neet", "biology", "neet pyq", "neet 20"]
}


async def generate_pyq_questions(
    topic: str, exam_type: str, difficulty: str, count: int
) -> list:
    """
    Generates questions using ONLY PYQ datasets.
    Filters vector store results to PYQ sources.
    """
    # search specifically in PYQ data
    search_query = f"{exam_type} {topic} previous year question"
    results = vector_store.query(query_text=search_query, top_k=12)

    # filter to PYQ sources only
    pyq_keywords = PYQ_SOURCES.get(exam_type, [])
    pyq_results = []
    for r in results:
        source = r.get("metadata", {}).get("source", "").lower()
        if any(kw.lower() in source for kw in pyq_keywords):
            pyq_results.append(r)

    # if not enough PYQ-specific results, use all results
    context_results = pyq_results if len(pyq_results) >= 3 else results
    context = "\n\n".join([
        r["metadata"].get("text", "") for r in context_results[:8] if r["metadata"]
    ])[:3500]

    system_prompt = f"""You are an expert {exam_type} question compiler.
Generate exactly {count} authentic PYQ-style MCQ questions on '{topic}'.
These should be from or inspired by actual {exam_type} previous year papers.
Difficulty: {difficulty}

IMPORTANT:
- Questions must be exam-accurate
- Include year if you know it (e.g., "[JEE Mains 2019]")
- No LaTeX backslashes in JSON values
- Include detailed explanations with concepts used

Return ONLY valid JSON array:
[
  {{
    "question": "question text [Year if known]",
    "options": {{"A": "option", "B": "option", "C": "option", "D": "option"}},
    "answer": "A",
    "explanation": "step-by-step solution with concept used",
    "year": "2019",
    "source": "{exam_type}"
  }}
]"""

    try:
        response = quiz_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic: {topic}\nPYQ Context:\n{context}")
        ])
        text = response.content.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())[:count]
    except Exception as e:
        print(f"[Quiz] PYQ generation error: {e}")

    return []


async def generate_regular_questions(
    topic: str, exam_target: str, difficulty: str, count: int, question_type: str
) -> list:
    """
    Generates regular quiz questions from the main RAG knowledge base.
    """
    results = vector_store.query(
        query_text=f"{topic} {exam_target} {question_type} questions",
        top_k=8
    )

    context = "\n\n".join([
        r["metadata"].get("text", "")
        for r in results
        if r["metadata"]
    ])

    if not context:
        return []

    system_prompt = f"""You are an expert {exam_target} question setter.
Generate exactly {count} {question_type} questions on '{topic}'.
Difficulty level: {difficulty}

You MUST format EVERY question EXACTLY like this (no bold, no markdown, no numbering):

Q: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [A or B or C or D]
Explanation: [one line explanation]

Rules:
- Start each question with "Q: "
- Each option starts with "A) ", "B) ", "C) ", "D) " (letter + closing paren + space)
- Answer line must be "Answer: " followed by ONLY the letter (A, B, C, or D)
- Explanation line must start with "Explanation: "
- Separate questions with a blank line
- Use ONLY the provided context
- Do NOT use bold (**), numbering (1.), or any other formatting"""

    response = quiz_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Context:\n{context[:3000]}\n\nGenerate {count} {difficulty} {question_type} questions on '{topic}' now."
        )
    ])

    raw_text = response.content
    print(f"[API: quiz] Raw LLM output ({len(raw_text)} chars):\n{raw_text[:600]}...")

    questions = parse_quiz_response(raw_text)

    if not questions:
        print("[API: quiz] Strict parser returned 0, trying flexible parser...")
        questions = parse_quiz_flexible(raw_text)

    if not questions:
        print("[API: quiz] Flexible parser returned 0, trying JSON fallback...")
        questions = parse_quiz_json_fallback(raw_text)

    print(f"[API: quiz] Generated {len(questions)} questions.")
    return questions

@quizRouter.post("/quiz")
async def generateQuiz(req: QuizRequest):
    """
    Generates MCQ quiz from RAG knowledge base.
    Retrieves relevant content then asks LLM to create questions.
    Generates quiz — regular or PYQ mode.
    """
    print(f"[API: quiz] Generating {req.numQuestions} {req.difficulty} questions on '{req.topic}'"
          f"{' [PYQ MODE: ' + req.pyqExamType + ']' if req.isPYQMode else ''}")

    if req.isPYQMode:
        questions = await generate_pyq_questions(
            req.topic, req.pyqExamType, req.difficulty, req.numQuestions
        )
    else:
        questions = await generate_regular_questions(
            req.topic, req.examTarget, req.difficulty, req.numQuestions, req.questionType
        )

    if not questions:
        return {"message": "no questions generated", "payload": []}

    return {"message": "quiz generated", "payload": questions}


# ─────────────────────────────────────────────
# PARSER 1: Strict — splits on "Q:" markers
# ─────────────────────────────────────────────

def parse_quiz_response(text: str) -> list:
    """
    Parses quiz text that uses "Q:" format.
    Most reliable when LLM follows the exact prompt format.
    """
    questions = []

    # split on Q: at start of line
    blocks = re.split(r'\n(?=Q\s*[:\.])', text)

    for block in blocks:
        block = block.strip()
        if not block or not re.match(r'^Q\s*[:\.]', block):
            continue

        q = parse_single_block(block)
        if q:
            questions.append(q)

    return questions


# ─────────────────────────────────────────────
# PARSER 2: Flexible — handles many LLM formats
# ─────────────────────────────────────────────

def parse_quiz_flexible(text: str) -> list:
    """
    Handles various LLM output formats:
    - **Question 1** / **Q1:** / 1. / Question 1: etc.
    - Question text may be on the same line or next line
    - Answer/Explanation may be missing
    """
    questions = []

    # split on question markers: **Question N**, Q1:, 1., Question N:, etc.
    split_pattern = r'\n\s*(?:\*{1,2}\s*)?(?:Q(?:uestion)?\s*\d*\s*[:.)\]]*\s*(?:\*{1,2})?\s*|\d+\s*[.:)]\s*(?:\*{1,2})?\s*)'
    parts = re.split(split_pattern, '\n' + text, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()
        if not part or len(part) < 20:
            continue

        q = parse_single_block(part)
        if q:
            questions.append(q)

    return questions


# ─────────────────────────────────────────────
# SHARED: Parse a single question block
# ─────────────────────────────────────────────

def parse_single_block(block: str) -> dict | None:
    """
    Parses a single question block into a structured dict.
    Handles various option and answer formats.
    """
    lines = block.strip().split('\n')
    if not lines:
        return None

    # extract question text — may span multiple lines before options start
    question_lines = []
    option_start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # check if this line is an option line
        if re.match(r'^\(?[A-Da-d][.):\]]\s', stripped):
            option_start_idx = i
            break
        # clean and collect question text
        cleaned = re.sub(r'^Q\s*[:.]\s*', '', stripped)  # remove "Q:" prefix
        cleaned = re.sub(r'^\*+\s*', '', cleaned)        # remove ** markers
        cleaned = re.sub(r'\*+$', '', cleaned)            # remove trailing **
        cleaned = re.sub(r'^\d+[.:)]\s*', '', cleaned)   # remove "1." prefix
        cleaned = cleaned.strip()
        if cleaned:
            question_lines.append(cleaned)

    question_text = ' '.join(question_lines).strip()
    if len(question_text) < 10:
        return None

    # extract options, answer, explanation from remaining lines
    options = {}
    answer = ""
    explanation = ""

    for line in lines[option_start_idx:]:
        stripped = line.strip()

        # match options: "A)", "A.", "(A)", "a)", "**A)**", "A )" etc.
        opt_match = re.match(
            r'^\(?(?:\*{0,2})\s*([A-Da-d])\s*[.):\]]\s*(?:\*{0,2})\s*(.+)',
            stripped
        )
        if opt_match:
            key = opt_match.group(1).upper()
            val = opt_match.group(2).strip().rstrip('*')
            if val:
                options[key] = val
            continue

        # match answer line: "Answer: A", "**Answer:** B)", "Correct: C" etc.
        ans_match = re.match(
            r'^(?:\*{0,2})\s*(?:Answer|Correct(?:\s+Answer)?)\s*[:.=]\s*(?:\*{0,2})\s*(.+)',
            stripped, re.IGNORECASE
        )
        if ans_match:
            ans_text = ans_match.group(1).strip().rstrip('*')
            # extract just the letter
            letter_match = re.match(r'^([A-Da-d])', ans_text)
            if letter_match:
                answer = letter_match.group(1).upper()
            continue

        # match explanation
        exp_match = re.match(
            r'^(?:\*{0,2})\s*Explanation\s*[:.=]\s*(?:\*{0,2})\s*(.+)',
            stripped, re.IGNORECASE
        )
        if exp_match:
            explanation = exp_match.group(1).strip().rstrip('*')
            continue

    # we need at least 2 options; answer is nice but not required
    # if no answer, default to "A" so the question still shows up
    if question_text and len(options) >= 2:
        if not answer:
            # try to find answer embedded in options (some LLMs mark correct with *)
            answer = "A"  # fallback — at least the question is usable
            print(f"[API: quiz] Warning: No answer found for '{question_text[:50]}', defaulting to A")

        return {
            "question": question_text,
            "options": options,
            "answer": answer,
            "explanation": explanation
        }

    return None


# ─────────────────────────────────────────────
# PARSER 3: JSON fallback
# ─────────────────────────────────────────────

def parse_quiz_json_fallback(text: str) -> list:
    """
    Fallback: if the LLM returned JSON-formatted questions.
    """
    import json as _json

    try:
        # try to find a JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            data = _json.loads(match.group())
            if isinstance(data, list):
                questions = []
                for q in data:
                    if isinstance(q, dict) and "question" in q:
                        questions.append({
                            "question": q.get("question", ""),
                            "options": q.get("options", {}),
                            "answer": q.get("answer", "A"),
                            "explanation": q.get("explanation", "")
                        })
                return questions
    except Exception:
        pass

    return []

@quizRouter.post("/quiz/save-result")
async def saveQuizResult(req: QuizResultRequest):
    """
    Saves quiz result to MongoDB for history + analysis.
    Called from frontend after quiz submission.
    """
    db = get_db()

     # prevent duplicate saves (idempotency check)
    recent = await db.quizhistory.find_one({
        "userId": req.userId,
        "completedAt": {"$gt": (datetime.utcnow().replace(second=0, microsecond=0)).isoformat()}
    })
    if recent and recent.get("topic") == req.topic:
        print(f"[API: quiz] Duplicate save prevented for user '{req.userId}'")
        recent.pop("_id", None)
        return {"message": "already saved", "payload": recent}

    result = {
        "userId": req.userId,
        "examTarget": req.examTarget,
        "topic": req.topic,
        "difficulty": req.difficulty,
        "totalQuestions": req.totalQuestions,
        "correct": req.correct,
        "wrong": req.wrong,
        "skipped": req.skipped,
        "scorePercent": req.scorePercent,
        "weakAreas": req.weakAreas,
        "questions": req.questions,
        "userAnswers": req.userAnswers,
        "completedAt": now()
    }

    await db.quizhistory.insert_one(result)
    result.pop("_id", None)

    # update personalization with quiz weak areas
    if req.weakAreas:
        await db.personalization.update_one(
            {"userId": req.userId},
            {
                "$addToSet": {"weakTopics": {"$each": req.weakAreas}},
                "$inc": {"quizCount": 1},
                "$set": {"lastActive": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

     # update leaderboard points
    await update_leaderboard_points(req.userId, "quiz", req.scorePercent, db)

    print(f"[API: quiz] Result saved — {req.scorePercent}% on {req.topic}")
    return {"message": "result saved", "payload": result}


@quizRouter.get("/quiz/history/{userId}")
async def getQuizHistory(userId: str):
    """Gets all quiz results for a user."""
    db = get_db()
    cursor = db.quizhistory.find(
        {"userId": userId}, {"questions": 0}         # exclude heavy questions from list view
    ).sort("completedAt", -1)
    history = await cursor.to_list(length=50)
    # convert ObjectId to string for JSON
    for h in history:
        if "_id" in h:
            h["_id"] = str(h["_id"])
    return {"message": "quiz history", "payload": history}

@quizRouter.get("/quiz/history/{userId}/{quizId}")
async def getQuizById(userId: str, quizId: str):
    """Gets a specific quiz result with all questions and answers."""
    
    db = get_db()
    try:
        from bson import ObjectId
        quiz = await db.quizhistory.find_one(
            {"_id": ObjectId(quizId), "userId": userId}
        )
        if not quiz:
            return {"message": "not found", "payload": None}
        quiz["_id"] = str(quiz["_id"])
        return {"message": "quiz", "payload": quiz}
    except Exception as e:
        print(f"[API: quiz] getQuizById error: {e}")
        return {"message": "error", "payload": None}


@quizRouter.get("/quiz/overall-analysis/{userId}")
async def getOverallAnalysis(userId: str):
    """
    Returns aggregated analysis across all quizzes.
    Average score, most attempted topics, improvement trend.
    """
    db = get_db()
    history = await db.quizhistory.find(
        {"userId": userId}, {"_id": 0}
    ).to_list(length=100)

    if not history:
        return {"message": "no history", "payload": None}

    total = len(history)
    avg_score = round(sum(h["scorePercent"] for h in history) / total)
    total_correct = sum(h["correct"] for h in history)
    total_wrong = sum(h["wrong"] for h in history)
    total_questions = sum(h["totalQuestions"] for h in history)

    # most attempted topics
    from collections import Counter
    topic_counts = Counter(h["topic"] for h in history)
    top_topics = [{"topic": t, "count": c} for t, c in topic_counts.most_common(5)]

    # weak areas across all quizzes
    all_weak = []
    for h in history:
        all_weak.extend(h.get("weakAreas", []))
    weak_counts = Counter(all_weak).most_common(5)

    # score trend (last 10)
    trend = [{"topic": h["topic"], "score": h["scorePercent"],
               "date": h["completedAt"][:10]} for h in history[-10:]]

    return {
        "message": "analysis",
        "payload": {
            "totalQuizzes": total,
            "averageScore": avg_score,
            "totalQuestionsAttempted": total_questions,
            "totalCorrect": total_correct,
            "totalWrong": total_wrong,
            "topTopics": top_topics,
            "commonWeakAreas": [w[0] for w in weak_counts],
            "scoreTrend": trend
        }
    }
