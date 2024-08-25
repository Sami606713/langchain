from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

# load the environment variables
load_dotenv()
api=os.environ["google_api"]
# Create Chat Model
print("Loading Model...")
model= GoogleGenerativeAI(
    google_api_key=api,
    model="gemini-1.5-pro",
    temperature=0.5,
    max_output_tokens=20
)

# Get the response
print("Getting Response...")
response=model.invoke("Which city is the biggest city in pakistan")

print("AI Response:")
print(response)

