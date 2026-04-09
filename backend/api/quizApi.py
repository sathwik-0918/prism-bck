# api/quizApi.py
# generates quizzes from RAG knowledge base
# topic-specific, difficulty-aware quiz generation
# uses Groq/Ollama LLM to create MCQ questions

from fastapi import APIRouter
from pydantic import BaseModel
from rag.nodes import llm
from rag.vectorstore import FaissVectorStore
from config import FAISS_STORE_PATH
from langchain_core.messages import SystemMessage, HumanMessage

quizRouter = APIRouter()

vector_store = FaissVectorStore(persist_dir=FAISS_STORE_PATH)

class QuizRequest(BaseModel):
    topic: str                  # e.g. "Thermodynamics"
    examTarget: str             # JEE or NEET
    difficulty: str = "medium"  # easy / medium / hard
    numQuestions: int = 5


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

    system_prompt = f"""You are an expert {req.examTarget} question setter.
Generate exactly {req.numQuestions} multiple choice questions on '{req.topic}'.
Difficulty level: {req.difficulty}

Format each question EXACTLY like this:
Q: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [correct letter]
Explanation: [brief explanation]

Use ONLY the provided context. Make questions exam-relevant."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context}\n\nGenerate {req.numQuestions} questions now.")
    ])

    # parse response into structured format
    questions = parse_quiz_response(response.content)

    print(f"[API: quiz] Generated {len(questions)} questions.")
    return {"message": "quiz generated", "payload": questions}


def parse_quiz_response(text: str) -> list:
    """Parses LLM quiz output into structured list of question dicts."""
    questions = []
    blocks = text.strip().split("\nQ:")

    for block in blocks:
        if not block.strip():
            continue

        try:
            lines = block.strip().split("\n")
            question_text = lines[0].replace("Q:", "").strip()
            options = {}
            answer = ""
            explanation = ""

            for line in lines[1:]:
                line = line.strip()
                if line.startswith("A)"):
                    options["A"] = line[2:].strip()
                elif line.startswith("B)"):
                    options["B"] = line[2:].strip()
                elif line.startswith("C)"):
                    options["C"] = line[2:].strip()
                elif line.startswith("D)"):
                    options["D"] = line[2:].strip()
                elif line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()
                elif line.startswith("Explanation:"):
                    explanation = line.replace("Explanation:", "").strip()

            if question_text and options and answer:
                questions.append({
                    "question": question_text,
                    "options": options,
                    "answer": answer,
                    "explanation": explanation
                })
        except Exception:
            continue

    return questions