from langchain.agents import AgentExecutor,create_structured_chat_agent
from langchain_core.tools import Tool
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.memory import ConversationBufferMemory
from langchain import hub
from dotenv import load_dotenv
import os
load_dotenv()

# make a tool
def search_wikipedia(query):
    try:
        import wikipedia

        return wikipedia.summary(query,sentences=2)
    except Exception as e:
        return str(e)

# print(search_wikipedia("Python"))
# add tool in tools list
tools=[
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="This tool can search wikipedia for you",

    )
]

# Initilize the agens
prompt=hub.pull("hwchase17/structured-chat-agent")
# load the llm
llm=ChatCohere(cohere_api_key=os.getenv("cohere_api"))
# make a memory
memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)

# make agent
agent=create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

# Exceute  the agent
agent_excuter=AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True ,
    memory=memory,
)

while True:
    query=input("Enter your query: ")
    if query=="q":
        break
    else:
        response=agent_excuter.invoke({"input":query})
        print(response['output'])