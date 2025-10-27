from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import pandas as pd
from supabase import create_client# for Supabase connection
from dotenv import load_dotenv

load_dotenv() # load environment variable from .env file

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend access 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# LLM models to rotate through if one fails
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
Shokti is designed for users with minimal technical knowledge and avoid too much complex calculation; it uses a friendly tone, clear instructions, short paragraphs, and highlights key actions in bold. 
It supports English and Bangla and can provide text guidance. 
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

# Load dataset
try:
    energy_data = pd.read_csv("data/cycled_hourly_power.csv", nrows=500)
    # Convert 'Datetime' column to pandas datetime
    energy_data['date_time'] = pd.to_datetime(energy_data['Datetime'])
except Exception as e:
    print(f"Error loading dataset: {e}")
    energy_data = pd.DataFrame()

sessions={}

# connectiong with supabse to take user info
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Pydantic model
class UserMessage(BaseModel):
    message: str
    user_id: str
    session_id: str = None

# fetch user profile
def fetch_user_profile(user_id):
    response = supabase.table("user_profiles").select("*").eq("auth_user_id", user_id).execute()
    data = response.data
    if data:
        return data[0]  # return first row
    return None
def get_user_energy_summary(high_energy_devices):
    summary={}
    try:
        devices = eval(high_energy_devices)  # convert string list to Python list
    except:
        devices = []
    for device in devices:
        col = None
        if device == "AC":
            col = 'Sub_metering_3'
        elif device == "Heater":
            col = 'Sub_metering_3'
        elif device == "Fridge":
            col = 'Sub_metering_2'
        elif device == "Washing Machine":
            col = 'Sub_metering_2'

        if col:
            summary[device] = round(energy_data[col].mean(), 2)
    return summary

# chat endpoint
@app.post("/chat")
async def chat_with_shoktti(user_message:UserMessage):
    user_id=user_message.user_id
    session_id = user_message.session_id or str(uuid.uuid4())
    user_text = user_message.message

    # Initialize session
    if session_id not in sessions:
        sessions[session_id] = []

    # Append user message
    sessions[session_id].append({"role": "user", "content": user_text})

    # Fetch user profile
    profile = fetch_user_profile(user_id)

    # Build personalized prompt
    personalized_prompt = SYSTEM_PROMPT
    if profile:
        user_summary = get_user_energy_summary(profile.get("high_energy_devices", "[]"))
        summary_text = "\nUser Energy Summary (based on dataset):\n"
        for device, usage in user_summary.items():
            summary_text += f"- {device}: avg usage {usage} watt-hour\n"

        personalized_prompt += f"""
User Profile Info:
- Name: {profile.get('name')}
- Home Type: {profile.get('home_type')}
- High Energy Devices: {profile.get('high_energy_devices')}
- Preferred Language: {profile.get('language')}
{summary_text}
"""
       # AI response
    reply_text = "Sorry, I couldn't process your message."
    for model_name in MODELS:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": personalized_prompt}] + sessions[session_id]
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
            break
        except Exception as e:
            print(f"API error with model {model_name}: {e}")
            continue

    return {"session_id": session_id, "reply": reply_text} 