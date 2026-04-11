# from typing import Annotated, TypedDict
# from langchain_core.messages import BaseMessage
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode
# from langchain_core.tools import tool
# import httpx
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage

# load_dotenv()


# class State(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]


# model = ChatGroq(
#     model="llama-3.3-70b-versatile", http_client=httpx.Client(verify=False)
# )


# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiplies two numbers together."""
#     return a * b


# tools = [multiply]
# model_with_tools = model.bind_tools(tools)


# def call_model(state: State):
#     print("📨 call_model — sending to model...")
#     response = model_with_tools.invoke(state["messages"])
#     print(f"📨 call_model — got response: {type(response).__name__}")
#     return {"messages": [response]}


# def should_continue(state: State):
#     last_message = state["messages"][-1]
#     if last_message.tool_calls:
#         print(
#             f"🔧 should_continue — tool call detected: {last_message.tool_calls[0]['name']}"
#         )
#         return "tools"
#     print("✅ should_continue — no tools, going to END")
#     return END


# graph_builder = StateGraph(State)
# graph_builder.add_node("call_model", call_model)
# graph_builder.add_node("tools", ToolNode(tools))
# graph_builder.add_edge(START, "call_model")
# graph_builder.add_conditional_edges("call_model", should_continue)
# graph_builder.add_edge("tools", "call_model")

# graph = graph_builder.compile()
# print("✅ Graph compiled successfully!")
# print("---")

# result = graph.invoke({"messages": [HumanMessage("What is 3 multiplied by 4?")]})
# print("---")
# print("💬 Final answer:", result["messages"][-1].content)



import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    http_client=httpx.Client(verify=False)
)

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together."""
    return a * b

agent = create_react_agent(model, tools=[multiply])

result = agent.invoke({"messages": [HumanMessage("What is 3 multiplied by 4?")]})
print("💬 Final answer:", result["messages"][-1].content)