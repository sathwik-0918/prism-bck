# rag/nodes.py
# contains all LangGraph node functions
# each node receives AgentState, does one job, returns updated state
# nodes: router → retrieve → grade → generate / rewrite

import os
# os.environ["OLLAMA_NUM_THREAD"] = "8"

from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from rag.agent_state import AgentState
from rag.vectorstore import FaissVectorStore
from rag.data_loader import load_all_documents
# from config import OLLAMA_BASE_URL, OLLAMA_MODEL, FAISS_STORE_PATH, DATA_PATH
from config import GROQ_API_KEY, FAISS_STORE_PATH, DATA_PATH

# print(f"[INFO] Initializing Ollama LLMs...")
print(f"[INFO] Initializing Groq LLMs...")

# fast_llm — for routing, grading, rewriting
# low token limit = instant decisions
# fast_llm = ChatOllama(
#     base_url=OLLAMA_BASE_URL,
#     model="llama3.2:1b",     # ← much smaller/faster
#     temperature=0,           # deterministic — yes/no decisions
#     num_predict=64,          # only needs short answers
#     num_ctx=1024,            # small context = fast
#     top_k=10,
# )
fast_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=64,
)

# main_llm — for final answer generation only
# higher quality, more tokens
# main_llm = ChatOllama(
#     base_url=OLLAMA_BASE_URL,
#     model=OLLAMA_MODEL,
#     temperature=0.1,
#     num_predict=1024,        # full answers
#     num_ctx=4096,            # full context
#     top_k=20,
#     top_p=0.9,
#     repeat_penalty=1.1,
#     num_thread=8,             # ← use all CPU cores
# )
main_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024,
)

# backward compat — keep llm for quiz/planner imports
llm = main_llm

# print(f"[INFO] Ollama LLMs ready — fast_llm + main_llm")
print(f"[INFO] Groq LLMs ready — fast_llm + main_llm")

print("[INFO] Initializing FAISS vector store...")
vector_store = FaissVectorStore(persist_dir=FAISS_STORE_PATH)

# load existing index if it exists, else build from data
faiss_index_path = os.path.join(FAISS_STORE_PATH, "faiss_index")
meta_path = os.path.join(FAISS_STORE_PATH, "metadata.pkl")

if os.path.exists(faiss_index_path) and os.path.exists(meta_path):
    print("[INFO] Found existing FAISS index — loading...")
    vector_store.load()
else:
    print("[INFO] No existing FAISS index — building from data folder...")
    from rag.data_loader import load_all_documents
    docs = load_all_documents(DATA_PATH)
    if docs:
        vector_store.build_from_documents(docs)
        print(f"[INFO] Built FAISS index from {len(docs)} documents.")
    else:
        print("[WARN] No documents found in data folder. Vector store is empty.")

print("[INFO] Vector store ready.")


# ─────────────────────────────────────────────
# QUERY PREPROCESSING
# condenses long/complex queries into clean search queries
# keeps original query intact for final generation
# ─────────────────────────────────────────────

def condense_query(query: str) -> str:
    """
    For long/complex queries (quiz reviews, multi-part questions, etc.),
    extract the core search intent. This condensed form is used for
    routing, retrieval, and grading — NOT for final generation.
    
    Short queries pass through unchanged.
    """
    # if query is short enough, no condensing needed
    if len(query) < 300:
        return query

    print(f"[PREPROCESS] Long query detected ({len(query)} chars) — condensing...")

    system_prompt = """You are a query condenser for an educational AI system.
The student sent a long message. Extract the CORE question or topic they need help with.
Return ONLY a short, focused query (1-2 sentences max). No explanation.

Examples:
- Long quiz review about SN1 reactions → "SN1 reaction mechanism concepts and preparation strategy"
- Detailed question about thermodynamics derivation → "First law of thermodynamics derivation"
- Performance review asking for guidance → "Study guidance for weak areas in [topic]"
"""

    try:
        response = fast_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Condense this student's message into a short search query:\n\n{query[:1000]}")
        ])
        condensed = response.content.strip()
        # sanity check — condensed should be shorter
        if len(condensed) < len(query) and len(condensed) > 5:
            print(f"[PREPROCESS] Condensed to: '{condensed}'")
            return condensed
    except Exception as e:
        print(f"[PREPROCESS] Condensing failed: {e}")

    # fallback — take last 200 chars (usually contains the actual question)
    fallback = query[-200:].strip()
    print(f"[PREPROCESS] Fallback condensed: '{fallback[:80]}...'")
    return fallback


# ─────────────────────────────────────────────
# NODE 1 — router_node
# decides: does this query need document retrieval or not?
# simple greetings / general questions → no retrieval
# subject questions, formulas, PYQs → retrieval needed
# ─────────────────────────────────────────────

# # router_node — use fast_llm
# def router_node(state: AgentState) -> AgentState:
#     query = state["query"]
#     print(f"[NODE: router] Query: '{query[:60]}'")
# 
#     # content safety check first
#     blocked_keywords = [
#         "porn", "sex", "drugs", "hack", "weapon",
#         "nude", "xxx", "kill", "bomb", "illegal"
#     ]
#     if any(kw in query.lower() for kw in blocked_keywords):
#         print(f"[NODE: router] Blocked — inappropriate content.")
#         return {**state, "retrieval_needed": False,
#                 "generation_count": 0, "grade_passed": False,
#                 "blocked": True}
# 
#     # build context-aware query for router
#     conversation = state.get("conversationContext", "")
#     context_hint = f"\n\nRecent conversation:\n{conversation}" if conversation else ""
# 
# 
#     system_prompt = """You are a query router for a JEE/NEET exam prep AI.
# Reply with ONLY one word:
# - 'retrieve' if the question is about: physics, chemistry, maths, biology, 
#   NCERT, formulas, JEE, NEET, derivations, syllabus, PYQs, study plans.
# - 'direct' for: greetings, thanks, small talk, unrelated topics.
# If the message is a follow-up like 'next', 'continue', 'more', 'explain further' — reply 'retrieve'."""
# 
# 
#     response = fast_llm.invoke([        # ← fast_llm
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=f"Query: {query}{context_hint}"[:600])   # cap input
#     ])
# 
#     decision = response.content.strip().lower()[:10]
#     print(f"[NODE: router] Decision: '{decision}'")
# 
#     return {
#         **state,
#         "retrieval_needed": "retrieve" in decision,
#         "generation_count": 0,
#         "grade_passed": False,
#         "blocked": False
#     }


# Simple greetings and small talk — never retrieve
DIRECT_PATTERNS = [
    "hi", "hello", "hey", "hola", "namaste", "thanks", "thank you",
    "ok", "okay", "bye", "good morning", "good night", "how are you",
    "what's up", "sup", "nice", "great", "cool", "awesome"
]

def router_node(state: AgentState) -> AgentState:
    query = state["query"].strip()
    query_lower = query.lower()

    print(f"[NODE: router] Query: '{query[:60]}'")

    # content safety check
    blocked_keywords = ["porn", "sex", "drugs", "hack", "weapon", "nude", "xxx", "bomb", "illegal"]
    if any(kw in query_lower for kw in blocked_keywords):
        return {**state, "retrieval_needed": False,
                "generation_count": 0, "grade_passed": False, "blocked": True}

    # ← hardcoded direct check — no LLM call needed for obvious cases
    is_direct = (
        query_lower in DIRECT_PATTERNS or
        len(query.split()) <= 2 and not any(
            kw in query_lower for kw in
            ["what", "how", "why", "when", "where", "which", "explain",
             "jee", "neet", "formula", "derive", "ncert", "chapter",
             "physics", "chemistry", "maths", "biology", "question",
             "define", "calculate", "find", "prove"]
        )
    )

    if is_direct:
        print(f"[NODE: router] Direct (no LLM needed): '{query}'")
        return {**state, "retrieval_needed": False,
                "generation_count": 0, "grade_passed": False, "blocked": False}

    # academic content — use fast_llm only for ambiguous cases
    conversation = state.get("conversationContext", "")
    system_prompt = """Route this student query. Reply ONLY 'retrieve' or 'direct'.
'retrieve' = academic question, formula, concept, PYQ, syllabus, follow-up on academic topic
'direct' = greeting, thanks, general chat, unclear"""

    response = fast_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {query[:300]}")
    ])

    decision = response.content.strip().lower()[:15]
    retrieve = "retrieve" in decision
    print(f"[NODE: router] Decision: '{decision}' → retrieve={retrieve}")

    return {
        **state,
        "retrieval_needed": retrieve,
        "generation_count": 0,
        "grade_passed": False,
        "blocked": False
    }



# ─────────────────────────────────────────────
# NODE 2 — retrieve_node
# fetches top-k relevant chunks from FAISS
# uses condensed/rewritten query for search, NOT raw query
# ─────────────────────────────────────────────
# Use llama3.2:1b for fast tasks (routing, grading)
# Keep llama3.2 for generation
# This is the biggest speed win possible

def retrieve_node(state: AgentState) -> AgentState:
    """
    Retriever node — queries FAISS vector store.
    Uses condensed/rewritten query for better search accuracy.
    """
    # use rewritten query (which may be condensed from preprocessing)
    query = state.get("rewritten_query") or state["query"]
    exam_target = state.get("examTarget", "")
    conversation = state.get("conversationContext", "")

    # if follow-up query, extract topic from conversation
    followup_words = ["next", "continue", "more", "explain", "further", "previous", "that", "point", "above", "mentioned", "guidance", "elaborate"]
    is_followup = any(w in query.lower() for w in followup_words)

    if is_followup and conversation:
        # extract last assistant response topic + last student query for context
        lines = conversation.split("\n")
        prev_parts = []
        for line in reversed(lines):
            if line.startswith("Prism:") and not prev_parts:
                prev_parts.append(line.replace("Prism:", "").strip()[:150])
            elif line.startswith("Student:") and len(prev_parts) < 2:
                prev_parts.append(line.replace("Student:", "").strip())
                break
        if prev_parts:
            context_hint = " | ".join(reversed(prev_parts))
            query = f"{context_hint} — {query}"
            print(f"[NODE: retrieve] Follow-up detected. Expanded query: '{query[:80]}'")

    search_query = f"{exam_target} {query}" if exam_target else query
    print(f"[NODE: retrieve] Search query: '{search_query[:80]}'")

    results = vector_store.query(query_text=search_query, top_k=8)

    documents = []
    sources = []

    for result in results:
        meta = result.get("metadata", {})
        text = meta.get("text", "")
        source = meta.get("source", "Unknown Source")

        if text:
            documents.append(text)
            if source not in sources:
                sources.append(source)

    print(f"[NODE: retrieve] Retrieved {len(documents)} document chunks.")
    print(f"[NODE: retrieve] Sources: {sources}")

    return {
        **state,
        "documents": documents,
        "sources": sources
    }


# ─────────────────────────────────────────────
# NODE 3 — grade_node
# checks if retrieved docs are actually relevant to the query
# if yes → go to generate
# if no → go to rewrite_node
# ─────────────────────────────────────────────

# grade_node — use fast_llm
def grade_node(state: AgentState) -> AgentState:
    query = state["query"]
    documents = state["documents"]
    generation_count = state["generation_count"]

    print(f"[NODE: grade] Grading {len(documents)} docs...")

    if not documents:
        return {**state, "grade_passed": False,
                "generation_count": generation_count + 1}

    context_preview = "\n\n".join(documents[:2])[:800]  # cap input

    system_prompt = """Grade if these documents are relevant to the query.
Reply ONLY 'yes' or 'no'."""

    response = fast_llm.invoke([        # ← fast_llm
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {query[:200]}\n\nDocs:\n{context_preview}")
    ])

    grade = response.content.strip().lower()[:5]
    passed = "yes" in grade
    print(f"[NODE: grade] Grade: '{grade}' → passed={passed}")

    return {
        **state,
        "grade_passed": passed,
        "generation_count": generation_count if passed else generation_count + 1
    }


# ─────────────────────────────────────────────
# NODE 4 — rewrite_node
# reformulates the query to get better retrieval results
# triggered when graded docs are not relevant
# ─────────────────────────────────────────────

# rewrite_node — use fast_llm
def rewrite_node(state: AgentState) -> AgentState:
    query = state["query"]
    conversation = state.get("conversationContext", "")
    print(f"[NODE: rewrite] Rewriting: '{query[:60]}'")

    # include conversation context so the rewriter understands follow-ups
    conv_hint = ""
    if conversation:
        conv_hint = f"\n\nRecent conversation for context:\n{conversation[-500:]}"

    system_prompt = """Rewrite this student query to be more specific for 
searching JEE/NEET educational documents. If it's a follow-up referencing
previous conversation, use that context to form a complete, specific query.
Return ONLY the rewritten query."""

    response = fast_llm.invoke([        # ← fast_llm
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{query[:300]}{conv_hint}")
    ])

    rewritten = response.content.strip()[:300]
    print(f"[NODE: rewrite] Rewritten: '{rewritten[:60]}'")
    return {**state, "rewritten_query": rewritten}


# ─────────────────────────────────────────────
# NODE 5 — generate_node
# generates final answer using retrieved context + LLM
# uses the ORIGINAL query for generation (not condensed)
# ─────────────────────────────────────────────

# generate_node — use main_llm (full quality)
def generate_node(state: AgentState) -> AgentState:
    # blocked content check
    if state.get("blocked"):
        return {**state,
                "answer": "I'm sorry, I can only help with JEE and NEET exam preparation topics.",
                "sources": []}

    query = state["query"]
    documents = state["documents"]
    sources = state["sources"]
    user_context = state.get("userContext", "")
    conversation = state.get("conversationContext", "")

    print(f"[NODE: generate] Generating for: '{query[:60]}'")

    context = "\n\n---\n\n".join(documents) if documents else "No context available."

    personalization = f"\n\nStudent Profile: {user_context}" if user_context else ""

    # inject conversation history for follow-up awareness
    conversation_section = ""
    if conversation:
        conversation_section = f"\n\nPrevious conversation (for context):\n{conversation}\n"


    system_prompt = f"""You are Prism — expert AI tutor for JEE and NEET preparation.
Answer using ONLY the provided context. Format using markdown:
- **bold** for key terms
- Numbered steps for derivations  
- $formula$ for inline math (LaTeX)
- $$formula$$ for block equations
- Bullet points for lists
- Tables where helpful

If the student says 'next topic', 'continue', or similar — continue from where you left off in the conversation.
Never make up information not in the context.{personalization}"""


    user_message = f"""Question: {query}
{conversation_section}
Context from study materials:
{context[:3500]}

Provide a clear, well-formatted answer. If this is a follow-up, continue naturally from the previous response."""



    response = main_llm.invoke([        # ← main_llm
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    answer = response.content.strip()
    print(f"[NODE: generate] Answer: {len(answer)} chars")
    return {**state, "answer": answer, "sources": sources}


# ─────────────────────────────────────────────
# NODE 6 — direct_generate_node
# handles queries that don't need retrieval
# greetings, small talk, general questions
# ─────────────────────────────────────────────

# direct_generate_node — use fast_llm (fast for greetings)
def direct_generate_node(state: AgentState) -> AgentState:
    # blocked content
    if state.get("blocked"):
        return {**state,
                "answer": "I'm sorry, I can only help with JEE and NEET exam preparation.",
                "sources": []}

    query = state["query"]
    print(f"[NODE: direct] Answering directly: '{query[:60]}'")

    system_prompt = """You are Prism — friendly JEE/NEET prep assistant.
For greetings, respond warmly and briefly. Guide the student to ask academic questions.
Keep response under 3 sentences."""

    response = fast_llm.invoke([        # ← fast_llm for speed
        SystemMessage(content=system_prompt),
        HumanMessage(content=query[:200])
    ])

    answer = response.content.strip()
    print(f"[NODE: direct] Done.")
    return {**state, "answer": answer, "sources": []}