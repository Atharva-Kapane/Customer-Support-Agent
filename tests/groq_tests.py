import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Please ensure GROQ_API_KEY is set in your environment or .env file.")

# Initialize the client pointed to Groq's endpoint
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

print("🔄 Fetching live model list from Groq...")
try:
    # 1. Dynamically retrieve all models active on your account
    model_list = client.models.list()
    # Extract just the string IDs, filtering out whisper audio models for chat completions
    available_models = [m.id for m in model_list.data if "whisper" not in m.id]
    print(f"✅ Found {len(available_models)} text models to test.\n")
except Exception as e:
    print(f"❌ Failed to fetch model list: {e}")
    available_models = []

# 2. Iterate through and test each model sequentially
for model_name in available_models:
    print(f"🚀 Testing: {model_name}...")
    try:
        response = client.chat.completions.create(
            model=model_name,  # Variable passed correctly without quotes
            messages=[{"role": "user", "content": "Say 'Groq is working' and nothing else."}],
            max_tokens=10  # Keep it small to minimize token usage during testing
        )
        # Extract the reply text cleanly
        reply = response.choices[0].message.content.strip()
        print(f"   🟢 SUCCESS -> Response: {reply}\n")
        
    except Exception as e:
        # Catch 404s, rate limits, or permission errors cleanly without crashing the script
        print(f"   🔴 FAILED  -> Error: {e}\n")

print("🏁 All model tests completed.")