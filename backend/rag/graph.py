# rag/graph.py
# assembles all nodes into a LangGraph StateGraph
# defines edges (transitions) between nodes
# conditional edges = agent decision making

from langgraph.graph import StateGraph, END
from rag.agent_state import AgentState
from rag.nodes import (
    router_node,
    retrieve_node,
    grade_node,
    rewrite_node,
    generate_node,
    direct_generate_node
)

# ─────────────────────────────────────────────
# conditional edge functions
# these decide WHICH node to go to next
# based on current state values
# ─────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """
    After router_node decides — go to retrieve or direct_generate.
    """
    if state["retrieval_needed"]:
        print("[EDGE: route_after_router] → retrieve")
        return "retrieve"
    else:
        print("[EDGE: route_after_router] → direct_generate")
        return "direct_generate"


def route_after_grade(state: AgentState) -> str:
    """
    After grade_node — if docs relevant go to generate,
    if not relevant AND under retry limit go to rewrite,
    if retry limit exceeded go to generate anyway (best effort).
    """
    generation_count = state["generation_count"]

    if generation_count == 0:
        # docs were relevant
        print("[EDGE: route_after_grade] Docs relevant → generate")
        return "generate"
    elif generation_count < 2:
        # docs not relevant, retry allowed
        print(f"[EDGE: route_after_grade] Docs not relevant (attempt {generation_count}) → rewrite")
        return "rewrite"
    else:
        # max retries reached — generate with whatever we have
        print("[EDGE: route_after_grade] Max retries reached → generate with best effort")
        return "generate"


# ─────────────────────────────────────────────
# build the LangGraph
# ─────────────────────────────────────────────

def build_rag_graph():
    """
    Builds and compiles the Agentic RAG LangGraph.
    
    Graph flow:
    START → router → [retrieve | direct_generate]
    retrieve → grade → [generate | rewrite]
    rewrite → retrieve  (loop — max 2 retries)
    generate → END
    direct_generate → END
    """
    print("[INFO] Building Agentic RAG LangGraph...")

    graph = StateGraph(AgentState)

    # add all nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("direct_generate", direct_generate_node)

    # set entry point
    graph.set_entry_point("router")

    # router → conditional split
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retrieve",
            "direct_generate": "direct_generate"
        }
    )

    # retrieve always goes to grade
    graph.add_edge("retrieve", "grade")

    # grade → conditional split
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    # rewrite loops back to retrieve
    graph.add_edge("rewrite", "retrieve")

    # terminal nodes → END
    graph.add_edge("generate", END)
    graph.add_edge("direct_generate", END)

    compiled = graph.compile()
    print("[INFO] Agentic RAG LangGraph compiled successfully.")
    return compiled


# compile graph once at module level — reused across all API calls
rag_graph = build_rag_graph()