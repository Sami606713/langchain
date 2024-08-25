from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

# load the environment variables
load_dotenv()
g_api=os.environ["google_api"]
# Create Chat Model
print("Loading Model...")
model= GoogleGenerativeAI(
    google_api_key=g_api,
    model="gemini-1.5-pro",
    temperature=0.5,
    max_output_tokens=20
)

# Get the response
print("Getting Response...")
response=model.invoke("Which city is the biggest city in pakistan")

print("AI Response:")
print(response)

#-----------------------------------------ChatAnthropic-----------------------------------------#
try:
    c_api=os.environ["claud_api"]

    model=ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2
    )
    # Get the response
    response=model.invoke("Which city is the biggest city in pakistan")

    print("Ai Response: ",response)
except Exception as e:
    print(e)