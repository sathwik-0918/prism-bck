# rag/nodes.py
# contains all LangGraph node functions
# each node receives AgentState, does one job, returns updated state
# nodes: router → retrieve → grade → generate / rewrite

import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rag.agent_state import AgentState
from rag.vectorstore import FaissVectorStore
from rag.data_loader import load_all_documents
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, FAISS_STORE_PATH, DATA_PATH


# ─────────────────────────────────────────────
# initialize shared LLM and vector store once
# ─────────────────────────────────────────────


print(f"[INFO] Initializing Ollama LLM: {OLLAMA_MODEL}...")
llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    temperature=0.1        # low temp = more factual, less creative
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
# NODE 1 — router_node
# decides: does this query need document retrieval or not?
# simple greetings / general questions → no retrieval
# subject questions, formulas, PYQs → retrieval needed
# ─────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """
    Agent router — decides if retrieval is needed.
    Returns state with retrieval_needed = True or False.
    """
    query = state["query"]
    print(f"[NODE: router] Query received: '{query}'")

    system_prompt = """You are a query router for an educational AI assistant called Prism.
Your job is to decide if the user's question needs document retrieval from our knowledge base.

Reply with ONLY one word:
- 'retrieve' if the question is about: physics, chemistry, maths, biology concepts, formulas, 
  NCERT topics, JEE/NEET questions, derivations, syllabus, study plans, PYQs, exam strategy.
- 'direct' if the question is: a greeting, small talk, thanks, or completely unrelated to exam prep."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])

    decision = response.content.strip().lower()
    print(f"[NODE: router] Decision: '{decision}'")

    return {
        **state,
        "retrieval_needed": decision == "retrieve",
        "generation_count": 0
    }


# ─────────────────────────────────────────────
# NODE 2 — retrieve_node
# fetches top-k relevant chunks from FAISS
# uses rewritten_query if available, else original query
# ─────────────────────────────────────────────

def retrieve_node(state: AgentState) -> AgentState:
    """
    Retriever node — queries FAISS vector store.
    Uses rewritten query if the previous retrieval was not relevant.
    """
    # use rewritten query if available (after a failed retrieval)
    query = state.get("rewritten_query") or state["query"]
    print(f"[NODE: retrieve] Retrieving docs for: '{query}'")

    results = vector_store.query(query_text=query, top_k=5)

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
    Document grader node — checks relevance of retrieved docs.
    Sets generation_count to signal if docs are relevant or not.
    Uses generation_count as a flag:
      - keeps value if docs are relevant (proceed to generate)
      - increments if docs are NOT relevant (trigger rewrite)
    """
    query = state["query"]
    documents = state["documents"]
    generation_count = state["generation_count"]

    print(f"[NODE: grade] Grading {len(documents)} docs for relevance...")

    if not documents:
        print("[NODE: grade] No documents retrieved — triggering rewrite.")
        return {**state, "generation_count": generation_count + 1}

    # ask LLM if documents are relevant
    context_preview = "\n\n".join(documents[:3])  # grade first 3 chunks

    system_prompt = """You are a document relevance grader.
Given a user query and some retrieved document chunks, decide if the chunks are useful.
Reply with ONLY 'yes' if the documents are relevant to the query.
Reply with ONLY 'no' if the documents are NOT relevant."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {query}\n\nDocuments:\n{context_preview}")
    ])

    grade = response.content.strip().lower()
    print(f"[NODE: grade] Relevance grade: '{grade}'")

    if grade == "yes":
        print("[NODE: grade] Documents are relevant — proceeding to generate.")
        return {**state, "generation_count": generation_count}
    else:
        print("[NODE: grade] Documents NOT relevant — triggering query rewrite.")
        return {**state, "generation_count": generation_count + 1}


# ─────────────────────────────────────────────
# NODE 4 — rewrite_node
# reformulates the query to get better retrieval results
# triggered when graded docs are not relevant
# ─────────────────────────────────────────────

def rewrite_node(state: AgentState) -> AgentState:
    """
    Query rewriter node — improves the query for better retrieval.
    Called when retrieved documents were not relevant.
    """
    original_query = state["query"]
    print(f"[NODE: rewrite] Rewriting query: '{original_query}'")

    system_prompt = """You are a query optimization expert for an educational RAG system.
Rewrite the given student query to make it more specific and likely to match educational document chunks.
Focus on: subject name, chapter, concept keywords, exam name (JEE/NEET).
Return ONLY the rewritten query — nothing else."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=original_query)
    ])

    rewritten = response.content.strip()
    print(f"[NODE: rewrite] Rewritten query: '{rewritten}'")

    return {**state, "rewritten_query": rewritten}


# ─────────────────────────────────────────────
# NODE 5 — generate_node
# generates final answer using retrieved context + Groq LLM
# ─────────────────────────────────────────────

def generate_node(state: AgentState) -> AgentState:
    """
    Generator node — produces the final answer using retrieved context.
    Builds a domain-specific prompt for JEE/NEET preparation.
    """
    query = state["query"]
    documents = state["documents"]
    sources = state["sources"]

    print(f"[NODE: generate] Generating answer for: '{query}'")
    print(f"[NODE: generate] Using {len(documents)} document chunks as context.")

    context = "\n\n---\n\n".join(documents) if documents else "No context available."

    system_prompt = """You are Prism — an expert AI tutor for JEE and NEET exam preparation.
You answer questions based ONLY on the provided context from NCERT textbooks and official study materials.

Rules:
1. Answer clearly and precisely using the context provided.
2. If the answer involves formulas, write them clearly.
3. If the context doesn't contain enough information, say so honestly.
4. Always be helpful, accurate, and student-friendly.
5. Do NOT make up information that is not in the context."""

    user_message = f"""Question: {query}

Context from study materials:
{context}

Please provide a clear, accurate answer based on the context above."""

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
    print(f"[NODE: direct_generate] Answering directly (no retrieval): '{query}'")

    system_prompt = """You are Prism — a friendly AI assistant for JEE and NEET exam preparation.
For general greetings or off-topic questions, respond warmly and guide the student back to studying.
Keep responses short and encouraging."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])

    answer = response.content.strip()
    print(f"[NODE: direct_generate] Direct answer generated.")

    return {**state, "answer": answer, "sources": []}