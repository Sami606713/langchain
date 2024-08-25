from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import os 
from dotenv import load_dotenv

# load the environment variables
load_dotenv()
api=os.environ["google_api"]

print('Loading Model...')
model=GoogleGenerativeAI(google_api_key=api,model="gemini-1.5-pro",temperature=0.5,max_output_tokens=50)

# setting thr message
print("Setting Mesaages..")
messages=[
    SystemMessage(content="You are a helpful chatbot"),
    HumanMessage(content="can you write some python program for me program is to add 2 nbr program should be dynamic"),
]

# generate the resposne
print("Generating Response...")
response=model.invoke(messages)

print("Ai Response: ",response)


# Set the AI messages
print("Setting AI messages...")
messages=[
    SystemMessage(content="you are a assistant to solve basic math problems"),
    HumanMessage(content="what is the product of 4 and 4"),
    AIMessage(content="product of 4 and 4 is equal to 16"),
    HumanMessage(content="what is the product of 7800 and 9000")
]

print("Generating New Response...")
response=model.invoke(messages)
print("Ai Response: ",response)