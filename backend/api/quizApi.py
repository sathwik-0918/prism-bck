# api/quizApi.py
# generates quizzes from RAG knowledge base
# topic-specific, difficulty-aware quiz generation
# uses Ollama LLM to create MCQ questions

import re
from fastapi import APIRouter
from pydantic import BaseModel
from rag.nodes import vector_store
from langchain_ollama import ChatOllama
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from langchain_core.messages import SystemMessage, HumanMessage

quizRouter = APIRouter()

# dedicated quiz LLM — needs more tokens than default 512
# 5 questions with options + answers + explanations = ~1500+ tokens
quiz_llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=0.3,
    num_predict=2048,       # enough for 5+ questions
    num_ctx=4096,           # larger context for chunked content + response
    repeat_penalty=1.1,
    top_k=20,
    top_p=0.9,
)

class QuizRequest(BaseModel):
    topic: str                  # e.g. "Thermodynamics"
    examTarget: str             # JEE or NEET
    difficulty: str = "medium"  # easy / medium / hard
    numQuestions: int = 5
    questionType: str = "MCQ (Single correct)"  # MCQ type


@quizRouter.post("/quiz")
async def generateQuiz(req: QuizRequest):
    """
    Generates MCQ quiz from RAG knowledge base.
    Retrieves relevant content then asks LLM to create questions.
    """
    print(f"[API: quiz] Generating {req.numQuestions} {req.difficulty} questions on '{req.topic}'")

    # retrieve relevant content for topic
    results = vector_store.query(
        query_text=f"{req.topic} {req.examTarget} questions",
        top_k=8
    )

    context = "\n\n".join([
        r["metadata"].get("text", "")
        for r in results
        if r["metadata"]
    ])

    if not context:
        return {"message": "no content found", "payload": []}

    # strict format prompt — forces consistent output
    system_prompt = f"""You are an expert {req.examTarget} question setter.
Generate exactly {req.numQuestions} multiple choice questions on '{req.topic}'.
Difficulty level: {req.difficulty}

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
        HumanMessage(content=f"Context:\n{context[:3000]}\n\nGenerate {req.numQuestions} {req.difficulty} MCQ questions on '{req.topic}' now.")
    ])

    raw_text = response.content
    print(f"[API: quiz] Raw LLM output ({len(raw_text)} chars):\n{raw_text[:600]}...")

    # parse response into structured format
    questions = parse_quiz_response(raw_text)

    # if strict parser fails, try flexible parser
    if not questions:
        print("[API: quiz] Strict parser returned 0, trying flexible parser...")
        questions = parse_quiz_flexible(raw_text)

    # if flexible parser fails, try JSON fallback
    if not questions:
        print("[API: quiz] Flexible parser returned 0, trying JSON fallback...")
        questions = parse_quiz_json_fallback(raw_text)

    print(f"[API: quiz] Generated {len(questions)} questions.")
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