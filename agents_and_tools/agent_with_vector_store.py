from langchain_community.document_loaders import TextLoader
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings,ChatCohere
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor,create_react_agent
from dotenv import load_dotenv
import os
load_dotenv()

# Steps we can do
# 1- Load the document.
# 2- Split the document.
# 3- Load the embeddings.
# 4- Load the vector db.
# 5- Set the retrieval chain.
# 6- Load the LLM model.



# Load the document
file_path="RAG/doc/wikipedia.txt"
vector_store_path="RAG/Db/recur_db"

print("===========Loading the document============")
loader=TextLoader(file_path=file_path)
doc=loader.load()

print("===========Split the document============")
# Split the doc in chunks
splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
chunk_doc=splitter.split_documents(documents=doc)

# load the embeddings
try:
    print("===========Loading the Embedding============")
    embedding=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ['cohere_api'])

    # load the vector db
    print("===========Load the vector db============")
    library=FAISS.load_local(folder_path=vector_store_path,embeddings=embedding,allow_dangerous_deserialization=True)

    # Set the retrieval chain
    print("===========Set the retrieval============")
    retiever=library.as_retriever(search_type="similarity_score_threshold",
        search_kwargs={"k":3,"score_threshold":0.4})
                                
except Exception as e:
    print(f"Embeddng Error is {e}")


# # load the llm
print("Loading LLM model")
llm=ChatCohere(cohere_api_key=os.environ['cohere_api'])

# Contextual question prompt
contextual_question_prompt=(
    "Given a chat history and the latest user question"
    "which might reference context in the history"
    "formulate the standalone question which can be understood"
    "with chat history donot answer the question, just"
    "reformulate it if need and otherwise return it as is."
)

print("Coustomize the prompt.")
costomize_q_prompt=ChatPromptTemplate.from_messages(
    [
    ("system",contextual_question_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
    ]
)


# Create a history aware retriever chain
print("Create a history aware retriever")
history_aware_retriever=create_history_aware_retriever(
    llm=llm,retriever=retiever,prompt=costomize_q_prompt
)


# # Set the Q&A system prompt
print("Set the system prompt..")
qa_system_prompt=(
    "you are a assistant for question answering task."
    "Use the following piece of retriever context to answe the question."
    "if you dont't know the answer, just saty i don't know."
    "Use three sentecen maximum and keep the answer concise.\n\n"
    "{context}"
)

print("Set the Q&A prompt")
qa_prompt=ChatPromptTemplate(
    [
        ("system",qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

# Question answering chain
print("Create a question answering chain")
question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

# Create the rag chain
print("Set the retrieval chain")
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

# pull the hub
prompt=hub.pull("hwchase17/react")

# make a tool
tool=[
    Tool(
        name="Vector Store Search",
        description="This tool can search the answer from vector store for you",
        func=lambda query,**karws:rag_chain.invoke({
            "input":query,
            "chat_history":karws.get("chat_history",[])
        })
    )
]

# make a agent
agent=create_react_agent(
    prompt=prompt,
    llm=llm,
    tools=tool,
    stop_sequence=True
)

# make a agent executor
agent_executor=AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tool,
    verbose=True
)

chat_history=[]
while True:
    query=input("Enter your query: ")
    if query=="q":
        break
    else:
        response=agent_executor.invoke({"input":query,"chat_history":chat_history})
        print(response['output'])

        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=response['output']))