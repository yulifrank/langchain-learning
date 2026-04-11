from dotenv import load_dotenv
from langchain_groq import ChatGroq
import httpx

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile",  http_client=httpx.Client(verify=False)
)
result =model.invoke("Who are you?")
print(result.content)