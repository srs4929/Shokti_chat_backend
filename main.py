from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI and Groq client
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Models to rotate between
MODELS = [
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "groq/compound"
]

# Structured system prompt with clarifying question logic
SYSTEM_PROMPT = """
Identity: Shokti AI, a friendly energy-saving assistant.
Background Context: The user wants guidance on electricity usage, saving energy, and green energy alternatives.
Role: Explain energy usage, suggest actionable steps, and guide users with practical advice.
Tone: Friendly, encouraging, concise, and easy to understand.
Conversational Flow: 
- Ask at most **one or two clarifying question** if the user's input is vague.
- After the question (or if enough info is already provided), give detailed suggestions. 
- Do not repeatedly ask for clarification in the same response.
Caution: Do not give inaccurate advice or assume any data. Only respond based on user-provided information.
Notes: Use short paragraphs,  highlight key actions in bold. Suggest alternative green energy solutions whenever possible.
"""

# Pydantic model for incoming user messages
class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_with_shokti(user_message: UserMessage):
    user_text = user_message.message
    reply_text = "Sorry, I couldn't process your message. Please try again."

    # Try models one by one until one works
    for model_name in MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text}
                ],
            )

            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message and hasattr(choice.message, "content"):
                reply_text = choice.message.content.strip()
            elif hasattr(choice, "text") and choice.text:
                reply_text = choice.text.strip()
            elif isinstance(choice, dict) and "text" in choice:
                reply_text = choice["text"].strip()
            break  # Success, stop rotating models

        except Exception as e:
            print(f"API error with model {model_name}: {e}")
            continue

    return {"reply": reply_text}
