from google import genai
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env
# Only run this block for Gemini Developer API
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Why is the sky blue?',
    config=genai.types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
    ),
)
print(response)
