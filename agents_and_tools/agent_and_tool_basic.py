from langchain import hub
from langchain.agents import (
    AgentExecutor,create_react_agent
)
from langchain_core.tools import Tool
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from datetime import datetime
import pprint
import os

os.environ["LANGCHAIN_API_KEY"] = "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"


def get_current_time(*args,**kwargs):
    return datetime.now().strftime("%H:%M:%S")

# print(get_current_time())

# Create tool list

tools=[
    Tool(
        name="Time",
        func=get_current_time,
        description="This agent can tell you the current time"
    )
]


# pull the hub
prompt=hub.pull("hwchase17/react")

# load the llm
llm=ChatCohere(cohere_api_key=os.getenv("cohere_api"))

# create the agent
agent=create_react_agent(
    prompt=prompt,
    llm=llm,
    tools=tools,
    stop_sequence=True
)

# Create the agent executor
agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run the agent
response=agent_executor.invoke({"input":"What is the current time?"})

print(response)