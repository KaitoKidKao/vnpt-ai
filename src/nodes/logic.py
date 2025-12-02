"""Logic solver node implementing a Code Agent workflow."""

from typing import Annotated, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import GraphState


_python_repl = PythonREPL()

@tool
def python_interpreter(code: Annotated[str, "The python code to execute"]) -> str:
    """
    Executes Python code. Use print() to see output.
    """
    try:
        if "print(" not in code:
            return "Error: You must use print() to output the result."
        
        result = _python_repl.run(code)
        return result.strip() if result else "Executed successfully (no output)."
    except Exception as e:
        return f"Execution Error: {str(e)}"

class FinalAnswerInput(BaseModel):
    answer: Literal["A", "B", "C", "D"] = Field(
        ..., description="The final selected option (A, B, C, or D)"
    )

@tool("final_answer", args_schema=FinalAnswerInput)
def final_answer(answer: str) -> str:
    """Submit the final answer and end the task."""
    return f"Answer submitted: {answer}"


CODE_AGENT_PROMPT = """Bạn là chuyên gia giải toán và logic bằng Python (Python Code Agent).

QUY TRÌNH:
1. Đọc câu hỏi và các lựa chọn.
2. Viết code Python để TÍNH TOÁN đáp án (dùng `print` để in kết quả).
3. Dựa vào kết quả chạy code, chọn đáp án đúng nhất (A, B, C, hoặc D).
4. Gọi tool `final_answer` ngay lập tức để trả về kết quả.

QUY TẮC:
- KHÔNG tính nhẩm. Phải dùng code để tính toán.
- Code ngắn gọn, trực diện.
- Trả lời dứt khoát.
"""


def get_agent_llm() -> ChatGoogleGenerativeAI:
    """Initialize LLM with tools."""
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0, 
    )
    return llm.bind_tools([python_interpreter, final_answer])

def logic_solver_node(state: GraphState) -> dict:
    """
    Code Agent Loop: Generate Code -> Execute -> Final Answer.
    Prints execution steps to console for monitoring.
    """
    llm = get_agent_llm()
    
    question_content = f"""
Câu hỏi: {state["question"]}
A. {state["option_a"]}
B. {state["option_b"]}
C. {state["option_c"]}
D. {state["option_d"]}
"""
    
    messages: list[BaseMessage] = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=question_content)
    ]
    
    max_steps = 3 

    for _ in range(max_steps):
        # Invoke LLM
        response: AIMessage = llm.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            # Force tool use if LLM chatters
            messages.append(HumanMessage(content="Hãy dùng tool python_interpreter hoặc final_answer."))
            continue

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name == "final_answer":
                ans = tool_args.get("answer", "A")
                print(f"    ✅ Final Answer: {ans}") 
                return {"answer": ans}

            elif tool_name == "python_interpreter":
                code = tool_args.get("code", "")
                print(f"    🐍 Python Code:\n{_indent_code(code)}")
                
                output = python_interpreter.invoke(code)
                print(f"    📄 Output: {output}")
                
                # Feedback to LLM
                messages.append(ToolMessage(content=output, tool_call_id=tool_id))

    print("    ⚠️  Max steps reached. Defaulting to A.")
    return {"answer": "A"}

def _indent_code(code: str) -> str:
    """Helper to indent code for prettier console output."""
    return "\n".join(f"        {line}" for line in code.splitlines())