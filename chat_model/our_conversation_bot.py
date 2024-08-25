from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import os
from dotenv import load_dotenv

load_dotenv()
api=os.environ["google_api"]

print('Loading Model...')
model=GoogleGenerativeAI(google_api_key=api,model="gemini-1.5-pro",temperature=0.5,max_output_tokens=50)

# set the conversation history

history=[
    SystemMessage(content="You are a helpful assistant")
]

def get_response(message):
    global history
    # append the mesage in the history
    history.append(HumanMessage(content=message))

    # get the response
    response=model.invoke(history)

    history.append(AIMessage(content=response))

    return response

while True:
    user_input=input("Enter your message: ")
    if user_input =="q":
        break
    else:
        print("User Input: ",user_input)
        response=get_response(message=user_input)
        print("Ai Response: ",response)