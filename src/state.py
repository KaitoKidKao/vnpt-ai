"""State schema definitions for the RAG pipeline graph."""

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """State schema for the RAG pipeline graph."""

    question_id: str
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    all_choices: list[str]  # All choices for questions with more than 4 options
    route: str
    context: str
    answer: str
    code_executed: str
    code_output: str
