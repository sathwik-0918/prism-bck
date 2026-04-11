# rag/pipeline.py
# main entry point for the agentic RAG pipeline
# called by chatApi.py — takes a query, runs through LangGraph, returns answer + sources

from rag.graph import rag_graph
from rag.agent_state import AgentState
from typing import List

def run_rag_pipeline(
    query: str,
    exam_target: str = "",
    user_context: str = "",
    recent_messages: List[dict] = None
) -> dict:
    """
    Runs agentic RAG pipeline with personalization.
    
    Args:
        query: student's question string
        exam_target: exam (JEE/NEET) for context filtering
        user_context: injected from MongoDB personalization profile
    
    Returns:
        dict with 'answer' and 'sources' keys

    Runs agentic RAG with conversation history for follow-up support.
    recent_messages: last 6 messages from the session for context.
    """
    print(f"\n{'='*50}")
    print(f"[PIPELINE] Query: '{query}'")
    print(f"[PIPELINE] Exam: {exam_target} | Context: {user_context[:60]}")

    # build conversation context string from recent messages
    conversation_context = ""
    if recent_messages:
        # take last 6 messages (3 exchanges)
        last_msgs = recent_messages[-6:] if len(recent_messages) > 6 else recent_messages
        conv_lines = []
        for msg in last_msgs:
            role = "Student" if msg.get("role") == "user" else "Prism"
            content = msg.get("content", "")[:300]  # cap each message
            conv_lines.append(f"{role}: {content}")
        conversation_context = "\n".join(conv_lines)
        print(f"[PIPELINE] Conversation history: {len(last_msgs)} messages injected")

    # initial state — all fields required by AgentState
    initial_state: AgentState = {
        "query": query,
        "examTarget": exam_target,
        "userContext": user_context,
        "rewritten_query": None,
        "documents": [],
        "answer": None,
        "sources": [],
        "generation_count": 0,
        "retrieval_needed": False,
        "grade_passed": False,
    }

    # run through LangGraph
    final_state = rag_graph.invoke(initial_state)

    answer = final_state.get("answer", "Sorry, I could not generate an answer.")
    sources = final_state.get("sources", [])

    print(f"[PIPELINE] Final answer ready. Sources: {sources}")
    print(f"{'='*50}\n")

    return {
        "answer": answer,
        "sources": sources
    }