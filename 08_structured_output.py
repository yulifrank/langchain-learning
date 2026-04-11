from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import httpx
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile", http_client=httpx.Client(verify=False)
)


class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    score: int = Field(description="Score from 1 to 10")
    summary: str = Field(description="Short summary of the review")


print("Model defined successfully!")

structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("Review the movie בחורים טובים")

print(type(result))
print(result.title)
print(result.score)
print(result.summary)
