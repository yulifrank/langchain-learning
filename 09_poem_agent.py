import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


@tool
def get_rhymes(word: str) -> list[str]:
    """Get rhymes for a Hebrew word from a rhyming dictionary."""
    print(f"🔍 get_rhymes called with: '{word}'")
    url = f"https://xn--9dbcb2e.com/haruzim.php?word={word}"
    response = httpx.get(url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    rhymes = [td.text.strip() for td in soup.find_all("td") if td.text.strip()]
    result = rhymes[:20]
    print(f"✅ get_rhymes returned {len(result)} rhymes: {result}")
    return result


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

agent = create_react_agent(
    llm,
    tools=[get_rhymes],
    prompt="""You are an expert Hebrew poet.

STEP 1 - MANDATORY - call get_rhymes for exactly 3 words related to the theme.

STEP 2 - Write the poem following this exact structure:
- Each line ends with a rhyming word
- The rhyming word is the LAST word of the line
- Lines 1+2 rhyme with each other, lines 3+4 rhyme with each other

Example of correct structure:
"יש לי בשבילכם איחולים המלצות ועוד כמה דברים
אז לא זה לא מושלם אחרי הכל אני מפתחת ולא כותבת ספרים
שלא תדעו מה זה ריגרסיה
שתמיד תתבצע היציאה לפונקציה"

Notice: "דברים" rhymes with "ספרים", "ריגרסיה" rhymes with "לפונקציה"

RULES:
- Use ONLY rhymes from get_rhymes results as the last word of each line
- Each line must be a complete meaningful sentence
- Write in Hebrew only
- At least 6 stanzas of 4 lines each
""",
)

print("=" * 40)
result = agent.invoke({"messages": [HumanMessage("כתוב לי בבקשה שיר מצחיק שמתאר קש של אחיות ומראה שאין אחים בלי ויכוחים")]})
print("=" * 40)

# הדפסת כל ה-messages כדי לראות מה קרה
last_message = result["messages"][-1]
if isinstance(last_message.content, list):
    for block in last_message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block["text"])
else:
    print(last_message.content)


print("=" * 40)
print("📝 Final poem:")
print(result["messages"][-1].content)
