import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()


history = InMemoryChatMessageHistory()

history.add_user_message("My name is Yael")
history.add_ai_message("Nice to meet you Yael!")

# print(history.messages)


model = ChatGroq(
    model="llama-3.3-70b-versatile", http_client=httpx.Client(verify=False)
)

# result = model.invoke(history.messages + [HumanMessage("What is my name")])


while True:
    user_input = input("You: ")
    history.add_user_message(user_input)

    response = model.invoke(history.messages)
    history.add_ai_message(response.content)

    print(f"AI: {response.content}")
