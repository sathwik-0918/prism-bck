# rag/agent_state.py
# defines the AgentState — the shared state that flows through all LangGraph nodes
# every node reads from and writes to this state
# think of it like req/res in Express — carries data across the pipeline

from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph agent.
    
    Fields:
    - query         : original user question
    - rewritten_query: reformulated query (if docs not relevant)
    - documents     : list of retrieved doc chunks from FAISS
    - answer        : final generated answer
    - sources       : list of source references for the answer
    - generation_count: tracks how many times we've tried retrieval (max 2)
    - retrieval_needed: True if agent decided retrieval is required
    """
    query: str
    rewritten_query: Optional[str]
    documents: List[str]
    answer: Optional[str]
    sources: List[str]
    generation_count: int
    retrieval_needed: bool