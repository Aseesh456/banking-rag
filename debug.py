import os
from dotenv import load_dotenv

print("STEP 1: Starting script", flush=True)
load_dotenv(dotenv_path=".env")
print("STEP 2: Loaded .env", flush=True)

api_key = os.getenv("GEMINI_API_KEY")
print(f"STEP 4: Found = {bool(api_key)}", flush=True)

if api_key:
    print(f"STEP 5: Length = {len(api_key)}", flush=True)
else:
    print("STEP 5: Key is None or empty", flush=True)