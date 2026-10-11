import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
api_key = os.getenv("GEMINI_API_KEY")

print("Testing API key...")
print("=" * 60)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    response = model.generate_content("Say hello in one word")
    print("✅ SUCCESS! API key works!")
    print("Response:", response.text)
except Exception as e:
    print("❌ ERROR - API key doesn't work")
    print("Error message:", str(e))
    print()
    print("=" * 60)
    print("Key loaded:", bool(api_key))
    print("Real Gemini keys start with: AIzaSy...")
    print("=" * 60)
