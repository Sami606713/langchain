#=======================================Imports===========================================================#
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv
#==========================================================================================================#
load_dotenv()
g_api=os.environ["google_api"]

# load the model
print('Load the model....')
model=ChatGoogleGenerativeAI(
    google_api_key=g_api,
    model="gemini-1.5-pro",
    temperature=0.1,
    max_output_tokens=10
)

# Make a template
print("Make the template...")
template=[
    ("system","you are a {topic} assistant"),
    ("human","can you help me in this {problem}")
]

# Make a prompt template
print("Make Chat prompt template")
prompt_template=ChatPromptTemplate.from_messages(template)

# make a chain
print("Building Chain....")
chain=prompt_template | model


# pass the input parameter in chain
print("Getting Response.....")
result=chain.invoke({
    "topic":"math",
    "problem":"2+2=?"
})


print(result.content)