#======================================Import PAckages======================================#
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings,ChatCohere
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_huggingface import ChatHuggingFace
import pandas as pd
from utils import (get_txt_files,load_document,create_vector_store,get_similar)
from dotenv import load_dotenv
import os
load_dotenv()
#====================================================================================================#

# Steps:
# 1- Read the csv file
# 2- Get the unique labels
# 3- Loop through the labels and call the get_txt_files function

df=pd.read_csv("projects/Q&A_System/bbc_data.csv")
labels = df['labels'].unique()
for label in labels:
    print("==========Label=======",label)
    get_txt_files(df,label)


# Load documents files
files_path="projects/Q&A_System/doc"
documents=load_document(files_path)

# Convert the text into chunks
print("===========Splitting the Documents into Chunks==============")
text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
doc_chunks=text_spliter.split_documents(documents=documents)

# Convert the text into embedding
try:
    embeddding=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ["cohere_api"])
    
    # Create a FAISS Vector Store
    print("===========Creating Vector Store==============")
    vector_store_path='projects/Q&A_System'
    library=create_vector_store("news_db",doc_chunks,embeddding,vector_store_path)

except Exception as e:
    print(f"This Error Occur: => {e}")



# Query
query="who threat to Apples iTunes  Users of Apples music jukebox iTunes?"
results=get_similar(query,library)

# Combine the query and relevant document
combine_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n\n"
    + "\n\n".join([doc.page_content for doc in results])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found, simply respond with 'I don't know'."
)

# Load the llm model
llm=ChatCohere(cohere_api_key=os.environ['cohere_api'])

# Set the human and system message
messages=(
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combine_input)
)

# get the response
try:
    response=llm.invoke(input=messages)
    print("===========Final Response============")
    print(response.content)
except Exception as e:
    print(f"Error is {e}")