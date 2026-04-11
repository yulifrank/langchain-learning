import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


model = ChatGroq(
    model="llama-3.3-70b-versatile", http_client=httpx.Client(verify=False)
)
prompt_1 = """You are a math assistant. You have access to tools: add and multiply.

IMPORTANT RULES:
- Call only ONE tool at a time
- Wait for the result before calling the next tool
- Never nest tool calls inside each other
- Break complex calculations into sequential steps
"""

agent = create_react_agent(
    model,
    tools=[multiply, add],
    prompt=prompt_1,
)

result = agent.invoke({"messages": [HumanMessage("כמה זה 3+4+3+4+55")]})


for message in result["messages"]:
    print(type(message).__name__, ":", message.content)
    print("---")
