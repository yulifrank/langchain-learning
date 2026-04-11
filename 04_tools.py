from langchain_core.tools import tool

import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b


model = ChatGroq(
    model="llama-3.3-70b-versatile", http_client=httpx.Client(verify=False)
)

model_with_tools = model.bind_tools([multiply])
result = model_with_tools.invoke("What is 3 multiplied by 4?")
print(result.tool_calls)

print("-----------------------------------------")

tool_call = result.tool_calls[0]
tool_result = multiply.invoke(tool_call["args"])
print("tool result:", tool_result)
# print(multiply.invoke({"a": 3, "b": 4}))
