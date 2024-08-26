from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import os
from dotenv import load_dotenv

# https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore_datastore/

"""
Setup the fire base
1- Create a firebase account
2- Create a project
   - Copy the project id
3- Create a database
4- Install the google cloud cli
   - https://cloud.google.com/sdk/docs/install
5- Authenticate the google cloud cli
    - https://cloud.google.com/sdk/docs/authenticating/provide-credentials-adc#local-dev
    - set the default project to the new firebase project
6- Enable the firestore api in you google cloud console

"""
project_name="chat-history"
project_id ="flask-blog-1a661"
# load the environment variables
load_dotenv()
g_api=os.environ["google_api"]

# load the model
model= ChatGoogleGenerativeAI(
    google_api_key=g_api,
    model="gemini-1.5-pro",
    temperature=0.5,
    max_output_tokens=20
)

# set the chat history
chat_history=[]

# append the system message
chat_history.append(SystemMessage(content="you are a jokes chatbot"))

# Start chat with human
try:
    while True:
        user=input("Enter Your Message: ")
        if user=="exit":
            break
        else:
            # Append the user message in chat history
            chat_history.append(HumanMessage(content=user))

            # get the response
            print("Human: ",user)
            response=model.invoke(chat_history)
            print("AI Response: ",response.content)
            # append the response in chat history
            chat_history.append(AIMessage(content=response.content))
except Exception as e:
    print(e)
        
print("---------Chat History:-----------\n",chat_history)