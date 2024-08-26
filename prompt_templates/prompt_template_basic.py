# -------------------------------Import Libraries--------------------------------#
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.prompts import ChatPromptTemplate
import os
# -------------------------------Import Libraries--------------------------------#

# -------------------------------Basic Chat Template with single instance--------------------------------#
template=content="tell me some joke on {topic}."

prompt_template=ChatPromptTemplate.from_messages([
    template
])

print("========Prompt Template:======")
print(prompt_template)

final_prompt=prompt_template.invoke({"topic":"animal"})
print("========finalPrompt Template:======")
print(final_prompt)

# ----------------------------------------------------------------------------------------------#

# -------------------------------Basic Chat Template with multiple instance--------------------------------#
template2="tell me {joke_nbr} jokes on {topic} each joke contain {nbr_of_word} words."

prompt_template2=ChatPromptTemplate.from_messages([
    template2
])

print("========Prompt Template2:======")
print(prompt_template2)

# Format the prompt template
final_prompt2=prompt_template2.invoke({"joke_nbr":3,"topic":"programing","nbr_of_word":10})
print("========finalPrompt Template2:======")
print(final_prompt2)
# ----------------------------------------------------------------------------------------------#

# -------------------------------Basic Chat Template with System and Human Messages--------------------------------#

messages=[
    ("system","You are the {name} assistant."),
    ( "human","can you tell me some thing on {topic}")
]

prompt_template3=ChatPromptTemplate.from_messages(messages)
print("========Prompt Template3:======")
print(prompt_template3)

final_prompt=prompt_template3.invoke({
    "name":"funny","topic":"AI"
})
print("========finalPrompt Template3:======")
print(final_prompt)