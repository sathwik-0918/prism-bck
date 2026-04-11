# rag/nodes.py
# contains all LangGraph node functions
# each node receives AgentState, does one job, returns updated state
# nodes: router → retrieve → grade → generate / rewrite

import os
os.environ["OLLAMA_NUM_THREAD"] = "8"   # use all CPU cores
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rag.agent_state import AgentState
from rag.vectorstore import FaissVectorStore
from rag.data_loader import load_all_documents
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, FAISS_STORE_PATH, DATA_PATH


# ─────────────────────────────────────────────
# initialize shared LLM and vector store once
# ─────────────────────────────────────────────

# main LLM — used for generation (needs larger context for complex queries)
llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=0.1,
    num_predict=1024,       # enough for detailed answers
    num_ctx=4096,           # handles long queries + context + response
    repeat_penalty=1.1,
    top_k=20,
    top_p=0.9,
)

# lightweight LLM — used for routing, grading, rewriting (fast, small output)
fast_llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=0.0,
    num_predict=32,         # only needs a word or two
    num_ctx=2048,           # enough for truncated query
    repeat_penalty=1.1,
    top_k=10,
    top_p=0.9,
)

print("[INFO] Ollama LLM ready.")

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

def router_node(state: AgentState) -> AgentState:
    """
    Agent router — decides if retrieval is needed.
    Returns state with retrieval_needed = True or False.
    Uses condensed query for routing decision.
    """
    query = state["query"]
    condensed = condense_query(query)
    
    # store condensed query for retrieval use
    print(f"[NODE: router] Query received: '{condensed[:100]}'")

    system_prompt = """You are a query router for an educational AI assistant called Prism.
Your job is to decide if the user's question needs document retrieval from our knowledge base.

Reply with ONLY one word:
- 'retrieve' if the question is about: physics, chemistry, maths, biology concepts, formulas, 
  NCERT topics, JEE/NEET questions, derivations, syllabus, PYQs, exam strategy, study guidance.
- 'direct' if the question is: a greeting, small talk, thanks, asking about time, or completely unrelated to studies.

IMPORTANT: If the student is asking for study advice, help, or guidance about their exam performance, reply 'retrieve'."""

    response = fast_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=condensed[:500])  # only first 500 chars for routing
    ])

    decision = response.content.strip().lower()
    # extract just the keyword — LLM sometimes adds extra text
    if "retrieve" in decision:
        decision = "retrieve"
    elif "direct" in decision:
        decision = "direct"
    else:
        # default to retrieve for academic content
        decision = "retrieve"
    
    print(f"[NODE: router] Decision: '{decision}'")

    return {
        **state,
        "retrieval_needed": decision == "retrieve",
        "generation_count": 0,
        "rewritten_query": condensed if condensed != query else None,
    }


# ─────────────────────────────────────────────
# NODE 2 — retrieve_node
# fetches top-k relevant chunks from FAISS
# uses condensed/rewritten query for search, NOT raw query
# ─────────────────────────────────────────────

def retrieve_node(state: AgentState) -> AgentState:
    """
    Retriever node — queries FAISS vector store.
    Uses condensed/rewritten query for better search accuracy.
    """
    # use rewritten query (which may be condensed from preprocessing)
    query = state.get("rewritten_query") or state["query"]
    exam_target = state.get("examTarget", "")

    # for retrieval, use a clean short query
    search_query = query[:200]  # cap search query length
    if exam_target:
        search_query = f"{exam_target} {search_query}"

    print(f"[NODE: retrieve] Search query: '{search_query[:100]}'")

    results = vector_store.query(query_text=search_query, top_k=5)

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

def grade_node(state: AgentState) -> AgentState:
    """
    Grades retrieved documents for relevance.
    Sets grade_passed=True if relevant, False if not.
    Uses condensed query for grading to avoid confusing the LLM.
    """
    query = state.get("rewritten_query") or state["query"]
    documents = state["documents"]
    generation_count = state["generation_count"]

    print(f"[NODE: grade] Grading {len(documents)} docs for relevance...")

    if not documents:
        print("[NODE: grade] No documents — triggering rewrite.")
        return {**state, "grade_passed": False, "generation_count": generation_count + 1}

    # use only first 2 docs for grading (speed)
    context_preview = "\n\n".join(documents[:2])[:1000]
    # use condensed query for grading
    grade_query = query[:200]

    system_prompt = """You are a document relevance grader.
Given a user query and retrieved document chunks, decide if the chunks contain useful information to answer the query.
Reply with ONLY the word 'yes' or 'no'. Nothing else."""

    response = fast_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {grade_query}\n\nDocuments:\n{context_preview}")
    ])

    grade = response.content.strip().lower()
    print(f"[NODE: grade] Relevance grade: '{grade[:20]}'")

    # check if "yes" appears anywhere in response (LLM sometimes adds extra text)
    if "yes" in grade:
        print("[NODE: grade] Documents relevant — proceeding to generate.")
        return {**state, "grade_passed": True, "generation_count": generation_count}
    else:
        print("[NODE: grade] Documents NOT relevant — triggering rewrite.")
        return {**state, "grade_passed": False, "generation_count": generation_count + 1}


# ─────────────────────────────────────────────
# NODE 4 — rewrite_node
# reformulates the query to get better retrieval results
# triggered when graded docs are not relevant
# ─────────────────────────────────────────────

def rewrite_node(state: AgentState) -> AgentState:
    """
    Query rewriter node — improves the query for better retrieval.
    Works on the condensed/rewritten query, not the raw original.
    """
    # rewrite from the condensed query, not the massive original
    current_query = state.get("rewritten_query") or state["query"]
    print(f"[NODE: rewrite] Rewriting query: '{current_query[:100]}'")

    system_prompt = """You are a query optimization expert for an educational RAG system.
Rewrite the given student query to make it more specific and likely to match educational document chunks.
Focus on: subject name, chapter, concept keywords, exam name (JEE/NEET).
Return ONLY the rewritten query — one short sentence, nothing else."""

    response = fast_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=current_query[:300])
    ])

    rewritten = response.content.strip()
    # cap rewritten query length
    rewritten = rewritten[:200]
    print(f"[NODE: rewrite] Rewritten query: '{rewritten}'")

    return {**state, "rewritten_query": rewritten}


# ─────────────────────────────────────────────
# NODE 5 — generate_node
# generates final answer using retrieved context + LLM
# uses the ORIGINAL query for generation (not condensed)
# ─────────────────────────────────────────────

def generate_node(state: AgentState) -> AgentState:
    """
    Generator node with personalization context injection.
    Uses the FULL original query for generation — student needs
    a response to their actual question, not the condensed version.
    """
    query = state["query"]  # always use ORIGINAL query for generation
    documents = state["documents"]
    sources = state["sources"]
    user_context = state.get("userContext", "")

    print(f"[NODE: generate] Generating answer for query ({len(query)} chars)")
    print(f"[NODE: generate] Using {len(documents)} chunks | User context: {user_context[:80]}")

    context = "\n\n---\n\n".join(documents) if documents else "No context available."

    # inject personalization into system prompt
    personalization_instruction = ""
    if user_context:
        personalization_instruction = f"\n\nStudent Profile: {user_context}"

    system_prompt = f"""You are Prism — an expert AI tutor for JEE and NEET exam preparation.
You help students with concepts, formulas, study guidance, quiz analysis, and exam strategy.

When the student shares quiz results or performance data:
- Analyze their weak areas based on incorrect answers
- Identify specific topics they need to revise
- Give actionable study advice with specific chapters/concepts to focus on
- Be encouraging but honest about gaps

When answering concept questions:
- Use the provided context from NCERT textbooks and study materials
- Format using clean markdown with **bold**, bullet points, numbered steps
- Use $formula$ for inline math and $$formula$$ for block equations
- Cite which subject/chapter the answer comes from

Rules:
1. Always be helpful and student-friendly
2. Give detailed, actionable responses
3. If context is available, use it. If not, still try to help with general guidance.{personalization_instruction}"""

    # truncate query if it's extremely long, but keep enough for context
    display_query = query if len(query) < 2000 else query[:2000] + "\n\n[Note: message truncated for processing]"

    user_message = f"""Student's message:
{display_query}

Reference materials:
{context[:2000]}

Provide a helpful, detailed response."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    answer = response.content.strip()
    print(f"[NODE: generate] Answer generated ({len(answer)} chars).")

    return {**state, "answer": answer, "sources": sources}


# ─────────────────────────────────────────────
# NODE 6 — direct_generate_node
# handles queries that don't need retrieval
# greetings, small talk, general questions
# ─────────────────────────────────────────────

def direct_generate_node(state: AgentState) -> AgentState:
    """
    Direct generator — answers without retrieval.
    For greetings, general questions, or off-topic queries.
    """
    query = state["query"]
    user_context = state.get("userContext", "")
    print(f"[NODE: direct_generate] Answering directly (no retrieval): '{query[:80]}'")

    personalization = ""
    if user_context:
        personalization = f"\nStudent Profile: {user_context}"

    system_prompt = f"""You are Prism — a friendly AI assistant for JEE and NEET exam preparation.
For general greetings or off-topic questions, respond warmly and guide the student back to studying.
Keep responses short and encouraging.{personalization}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query[:500])
    ])

    answer = response.content.strip()
    print(f"[NODE: direct_generate] Direct answer generated.")

    return {**state, "answer": answer, "sources": []}