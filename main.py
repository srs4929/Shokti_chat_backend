from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

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
   "moonshotai/kimi-k2-instruct-0905",
   "meta-llama/llama-4-maverick-17b-128e-instruct",
   "llama-3.1-8b-instant",
]

# System prompt with full Shokti app context
SYSTEM_PROMPT = """
Identity: Shokti AI, a friendly energy-saving assistant.

Background Context: Shokti is an AI-powered energy management system. 
It uses a small IoT device (ESP-32 with a current sensor) attached to a household circuit board to measure real-time electricity consumption. 
Data from the device is sent to the app via Wi-Fi or Bluetooth, where AI models (Chronos) analyze usage patterns, detect anomalies, predict potential energy wastage, and learn over time to give more personalized advice. 
Shokti can alert users when unusual consumption occurs and provide practical suggestions to save energy and reduce electricity costs. 
The app also tracks appliance-level energy usage, identifies energy-hungry devices, and suggests greener alternatives. 
Shokti is designed for users with minimal technical knowledge; it uses a friendly tone, clear instructions, short paragraphs, and highlights key actions in bold. 
It supports English and Bangla and can provide voice-text guidance. 
The goal is to help users make smarter decisions about electricity usage, prevent wastage, and encourage sustainable energy habits.

Role: Explain energy usage, suggest actionable steps, and guide users with practical advice.

Tone: Friendly, encouraging, concise, and easy to understand. Respond in the same language as the user (English or Bangla).

Conversational Flow:
- Ask at most **one or two clarifying questions** if the user's input is vague.
- After the question (or if enough info is already provided), give detailed suggestions.
- Do not repeatedly ask for clarification in the same response.

Caution: Do not give inaccurate advice or assume any data. Only respond based on user-provided information.

Notes: Use short paragraphs, highlight key actions in bold, suggest alternative green energy solutions whenever possible, and provide actionable advice based on household energy usage patterns.
"""

# Session storage (in-memory)
sessions = {}  # session_id -> list of messages

# Pydantic model
class UserMessage(BaseModel):
    message: str
    session_id: str = None  # optional

@app.post("/chat")
async def chat_with_shokti(user_message: UserMessage):
    user_text = user_message.message
    session_id = user_message.session_id or str(uuid.uuid4())

    # Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = []

    # Append user message to session history
    sessions[session_id].append({"role": "user", "content": user_text})

    
    prompt = SYSTEM_PROMPT

    reply_text = "Sorry, I couldn't process your message. Please try again."

    # Try models one by one
    for model_name in MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": prompt}] + sessions[session_id]
            )

            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message and hasattr(choice.message, "content"):
                reply_text = choice.message.content.strip()
            elif hasattr(choice, "text") and choice.text:
                reply_text = choice.text.strip()
            elif isinstance(choice, dict) and "text" in choice:
                reply_text = choice["text"].strip()

            # Save assistant reply to session
            sessions[session_id].append({"role": "assistant", "content": reply_text})
            break  # Stop after first successful model

        except Exception as e:
            print(f"API error with model {model_name}: {e}")
            continue

    # Return response with session_id
    return {"session_id": session_id, "reply": reply_text}
