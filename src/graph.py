"""LangGraph definition for the RAG pipeline."""

from typing import TypedDict

from langgraph.graph import END, StateGraph


class GraphState(TypedDict, total=False):
    """State schema for the RAG pipeline graph."""

    question_id: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    route: str
    context: str
    answer: str
    code_executed: str
    code_output: str


def build_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    from src.nodes.logic import logic_solver_node
    from src.nodes.rag import knowledge_rag_node, safety_guard_node
    from src.nodes.router import route_question, router_node

    workflow = StateGraph(GraphState)

    workflow.add_node("router", router_node)
    workflow.add_node("knowledge_rag", knowledge_rag_node)
    workflow.add_node("logic_solver", logic_solver_node)
    workflow.add_node("safety_guard", safety_guard_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            "knowledge_rag": "knowledge_rag",
            "logic_solver": "logic_solver",
            "safety_guard": "safety_guard",
        },
    )

    workflow.add_edge("knowledge_rag", END)
    workflow.add_edge("logic_solver", END)
    workflow.add_edge("safety_guard", END)

    return workflow.compile()


graph = None


def get_graph():
    """Get or create the compiled graph singleton."""
    global graph
    if graph is None:
        graph = build_graph()
    return graph

