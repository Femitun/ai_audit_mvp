import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key from the .env file
load_dotenv()

# Make sure this matches the variable name in your .env file
# It usually defaults to GOOGLE_API_KEY or GEMINI_API_KEY
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Available Chat Models for your API Key:")
print("-" * 40)
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
print("-" * 40)