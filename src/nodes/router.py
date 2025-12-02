"""Router node for classifying questions and directing to appropriate handlers."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.graph import GraphState

ROUTER_SYSTEM_PROMPT = """Bạn là một bộ phân loại câu hỏi. Nhiệm vụ của bạn là phân loại câu hỏi vào một trong các danh mục sau:

1. "knowledge" - Câu hỏi về lịch sử, địa lý, văn hóa, kiến thức tổng quát
2. "math" - Câu hỏi toán học, tính toán, logic số học
3. "toxic" - Câu hỏi độc hại, nhạy cảm, yêu cầu thông tin nguy hiểm

Chỉ trả lời MỘT từ duy nhất: knowledge, math, hoặc toxic."""

ROUTER_USER_PROMPT = """Phân loại câu hỏi sau:

Câu hỏi: {question}

Các đáp án:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Danh mục:"""


def get_router_llm() -> ChatGoogleGenerativeAI:
    """Initialize router LLM."""
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


def router_node(state: GraphState) -> dict:
    """Analyze question and determine routing path."""
    llm = get_router_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", ROUTER_USER_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "option_a": state["option_a"],
        "option_b": state["option_b"],
        "option_c": state["option_c"],
        "option_d": state["option_d"],
    })

    route = response.content.strip().lower()

    if "math" in route or "logic" in route:
        route_type = "math"
    elif "toxic" in route or "danger" in route or "harmful" in route:
        route_type = "toxic"
    else:
        route_type = "knowledge"

    return {"route": route_type}


def route_question(state: GraphState) -> Literal["knowledge_rag", "logic_solver", "safety_guard"]:
    """Conditional edge function to route to appropriate node."""
    route = state.get("route", "knowledge")

    if route == "math":
        return "logic_solver"
    elif route == "toxic":
        return "safety_guard"
    else:
        return "knowledge_rag"

