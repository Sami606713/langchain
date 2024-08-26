#========================================Imports========================================#
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
#========================================================================================#

# -------------------------------Load Environment Variable-------------------------------#
load_dotenv()
g_api=os.environ["google_api"]

# =================Model=================#
print("Setting Model...")
model=ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=g_api,
    temperature=0.5,
    max_output_tokens=50
)

# make the prompt template with single place holder
print("Setting Prompt Template...")
template="Tell me some thing on {topic}"

prompt_template=ChatPromptTemplate.from_messages([template])

final_promt=prompt_template.invoke({"topic":"Ai"})

# put the prompt in the model
print("Getting Response...")
response=model.invoke(final_promt)
print("==============Model Response=============")
print(response.content)

# ===========================================================================================#
# make the prompt template with multiple place holder
print("Setting Prompt Template2...")
template2="Tell me some thing on {topic2} and complete with in {nbr} words"

prompt_template2=ChatPromptTemplate.from_messages([template2])

final_prompt2=prompt_template2.invoke({
    "topic2":"Machine Learning",
    "nbr":10
})

print("Getting the response2...")
response2=model.invoke(final_prompt2)

print(response2.content)

# ===========================================================================================#

# ===============================PromptTemplate With Human and System Message=======================#
messages=[
    ("system","you are a {name} assistant"),
    ("human","tell me short story on topic {topic} and complete this with in {nbr} words and {line} lines")
]

prompt3=ChatPromptTemplate.from_messages(messages)

final_prompt3=prompt3.invoke({
    "name":"joking",
    "topic":"animals",
    "nbr":20,
    "line":5
})

response3=model.invoke(final_prompt3)
print("============Final Resposne3================")
print(response3.content)