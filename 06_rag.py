import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = certifi.where()

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.chat_history import InMemoryChatMessageHistory


load_dotenv()

# טעינה
loader = PyPDFLoader("contract.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    http_client=httpx.Client(verify=False),
)


@tool
def search_contract(query: str) -> str:
    """Search the employment contract for relevant information."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


history = InMemoryChatMessageHistory()


agent = create_react_agent(
    model,
    tools=[search_contract],
    prompt="""You are an expert assistant that helps employees understand their employment contract.

RULES:
- Only search the contract if the question is about employment terms, salary, vacation, benefits, etc.
- For greetings or unrelated questions — answer directly WITHOUT calling any tool
- Answer in Hebrew
- Call only ONE tool at a time
- Always use valid JSON format when calling tools
""",
)
print("Ask question , exit to end call")
print("---")


while True:
    question = input("Q : ")
    if question.lower() == "exit":
        break
    history.add_user_message(question)

    result = agent.invoke({"messages": history.messages})
    history.add_ai_message(result["messages"][-1].content)

    print("A: ", result["messages"][-1].content)
    print("---")
