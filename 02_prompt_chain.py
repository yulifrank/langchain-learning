from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import httpx


load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", http_client=httpx.Client(verify=False)
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates text into {language}."),
        ("user", "{text}"),
    ]
)


# result =prompt.invoke("Who are you?")
# result = prompt.invoke({"language": "Hebrew", "text": "Good morning"})

# print(result)

# res = model.invoke(result)
# print(res.content)

chain = prompt | model | StrOutputParser()
result = chain.invoke({"language": "Hebrew", "text": "Good morning"})
print(result)
