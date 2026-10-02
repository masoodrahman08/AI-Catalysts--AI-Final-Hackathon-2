import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY", "").strip()

print("API key configured:", bool(api_key))
print("API key prefix:", api_key[:3] if api_key else "NONE")
print("API key length:", len(api_key))

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly: GEMINI TEST OK"
)

print("\nGemini response:")
print(response.text)