# ---------------------------------------Importing Libraries---------------------------------------#
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_community.vectorstores.chroma import Chroma
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
co_api=os.environ['cohere_api']

# Vectore Store https://python.langchain.com/v0.2/docs/integrations/vectorstores/
#------------------------------------------------------------------------------------------------#

# Steps to run the code
# 1- Load the document
# 2- Split the document into chunks
# 3- Convert the chunks into embedding
# 4- Save the embedding into vector db
# ------------------------------------------------------------------------------------------------#

# Load the document
file_path = "RAG/doc/wikipedia.txt"
persist_directory=os.path.join("RAG/Db","wikipedia_db")

doc_loader=TextLoader(file_path=file_path)
doc=doc_loader.load()

print("\n=====Doc==========\n",doc)
print(f"=====Length: {len(doc)}")

# =============================Split the text into chunks===================================#
text_spliter=CharacterTextSplitter(chunk_size=100,chunk_overlap=0)
doc_chunks=text_spliter.split_documents(documents=doc)

print("============Document Chunks Information===============")
print(f"=====Total chunks : {len(doc_chunks)}")
print(f"=====First Chunk=====\n {doc_chunks[0].page_content}")

# =============================Text Into Embedding===================================#
try:
    embeddings_function = CohereEmbeddings(model="embed-english-light-v3.0",
                                        cohere_api_key="HMYlZTLMskrr4ClxQC8vxnz64LxuMN66SmMfWsne")
    
except Exception as e:
    print(f"This Error Occur: => {e}")

# =============================Create a chroma Vector Store======================# 
try:
    library=FAISS.from_documents(documents=doc_chunks,embedding=embeddings_function)
    # library=Chroma.from_documents(documents=doc_chunks,embedding=embeddings_function,persist_directory=persist_directory)
    
    print("===========Vectore Store Created=========")
except Exception as e:
    print(f"========Vector Db error ==> {e}")

#===============================================================================================#