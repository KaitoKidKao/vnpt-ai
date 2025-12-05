"""Direct Answer node for reading comprehension or general questions without RAG."""

import re
import string

from langchain_core.prompts import ChatPromptTemplate

from src.state import GraphState
from src.utils.llm import get_large_model
from src.utils.logging import print_log

DIRECT_SYSTEM_PROMPT = """Bạn là chuyên gia đọc hiểu và phân tích. 
Nhiệm vụ: Trả lời câu hỏi dựa trên thông tin được cung cấp trong đề bài (nếu có) hoặc kiến thức chung.

Lưu ý:
1. Nếu đề bài có đoạn văn, CHỈ dựa vào đoạn văn đó để suy luận.
2. Suy luận ngắn gọn, logic.
3. Kết thúc bằng: "Đáp án: X" (X là một trong các lựa chọn A, B, C, D, ...)."""

DIRECT_USER_PROMPT = """Câu hỏi: {question}
{choices}"""


def _format_choices_prompt(choices: list[str]) -> str:
    """Format choices for prompt."""
    option_labels = string.ascii_uppercase
    lines = []
    for i, choice in enumerate(choices):
        if i < len(option_labels):
            lines.append(f"{option_labels[i]}. {choice}")
    return "\n".join(lines)


def direct_answer_node(state: GraphState) -> dict:
    """Answer questions directly using Large Model (Skip Retrieval)."""
    print_log("        [Direct] Processing Reading Comprehension/General Question...")
    
    all_choices = state.get("all_choices", [])
    if not all_choices:
        all_choices = [
            state.get("option_a", ""),
            state.get("option_b", ""),
            state.get("option_c", ""),
            state.get("option_d", ""),
        ]
        all_choices = [c for c in all_choices if c]
    
    choices_text = _format_choices_prompt(all_choices)
    
    llm = get_large_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", DIRECT_SYSTEM_PROMPT),
        ("human", DIRECT_USER_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "choices": choices_text,
    })

    content = response.content.strip()
    print_log(f"        [Direct] Reasoning: {content}...")
    
    from src.nodes.rag import extract_answer 
    answer = extract_answer(content, max_choices=len(all_choices))
    
    print_log(f"        [Direct] Final Answer: {answer}")
    return {"answer": answer}
