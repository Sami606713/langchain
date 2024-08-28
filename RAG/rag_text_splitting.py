#======================================Import Packages====================================================#
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (HumanMessage,AIMessage,SystemMessage)
from langchain_core.prompts import ChatPromptTemplate 
from langchain.text_splitter import (CharacterTextSplitter,SentenceTransformersTokenTextSplitter,
                                     RecursiveCharacterTextSplitter,TextSplitter,TokenTextSplitter)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from  dotenv import load_dotenv
import os
load_dotenv()
#==========================================================================================================#
# Steps to run the code
# 1- Load the document
# 2- Split the document into chunks
# 3- Convert the chunks into embedding
# 4- Save the embedding into vector db

# fun for creating the vector store
def create_local_vector_store(documents,embedding,file_path,store_name):
    try:
        library=FAISS.from_documents(documents,embedding)
        if not os.path.exists(file_path):
            library.save_local(folder_path=file_path)
            print(f"========{store_name} Vectore Store created=======")
            
        else:
            return library
    except Exception as e:
        return f"Error Occur: {e}"

# vector db path
char_db_path=os.path.join("RAG/Db","char_db")
sent_db_path=os.path.join("RAG/Db","sent_db")
recur_db_path=os.path.join("RAG/Db","recur_db")
custom_text_db_path=os.path.join("RAG/Db","custom_splitter_db")
token_db_path=os.path.join("RAG/Db","token_db")

# Load the document
file_path = "RAG/doc/wikipedia.txt"
doc_loader=TextLoader(file_path=file_path)
document=doc_loader.load()

# +=========================Char Text  Splitter===============================#
# Split the text into chunks
print('============Char text Splitter==================')
char_spliter=CharacterTextSplitter(chunk_size=100,chunk_overlap=0)
chunk_text=char_spliter.split_documents(documents=document)
print("========Total Chunks in Char splitting======",len(chunk_text))
print("========First Chunk Data======",(chunk_text[0].page_content))

# Convert the text into embedding
try:
    embedding_fun=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ['cohere_api'])
except Exception as e:
    print(f"This Error Occur: => {e}")

# Create a FAISS Vector Store
char_db=create_local_vector_store(documents=chunk_text,embedding=embedding_fun,
                                  file_path=char_db_path,store_name="Char_db")

# +======================================================================#

# # +=========================Sentence Text  Splitter===============================#
print('============Sent text Splitter==================')
# sent_spliter=SentenceTransformersTokenTextSplitter(chunk_size=100,chunk_overlap=0)
# chunk_text=sent_spliter.split_documents(documents=document)
# print("========Total Chunks in Sentence splitting======",len(chunk_text))
# print("========First Chunk Data======",(chunk_text[0].page_content))

# # Create a FAISS Vector Store
# sent_db=create_local_vector_store(documents=chunk_text,embedding=embedding_fun,
#                                   file_path=sent_db_path,store_name="sent_db")
print('At that time sent text splitter through an error due to some internal issues')
# # +======================================================================#

# +=========================RecerssiveText  Splitter===============================#
print('============Recurrsive text Splitter==================')
rec_spliter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
chunk_text=rec_spliter.split_documents(documents=document)
print("========Total Chunks in recurrsive splitting======",len(chunk_text))
print("========First Chunk Data======",(chunk_text[0].page_content))

# Create a FAISS Vector Store
rec_db=create_local_vector_store(documents=chunk_text,embedding=embedding_fun,
                                  file_path=recur_db_path,store_name="rec_db")
# +======================================================================#

# +=========================Token Text Splitter===============================#
print('============Recurrsive text Splitter==================')
token_spliter=TokenTextSplitter(chunk_size=100,chunk_overlap=0)
chunk_text=token_spliter.split_documents(documents=document)
print("========Total Chunks in token Txt splitting======",len(chunk_text))
print("========First Chunk Data======",(chunk_text[0].page_content))

# Create a FAISS Vector Store
token_text_db=create_local_vector_store(documents=chunk_text,embedding=embedding_fun,
                                  file_path=token_db_path,store_name="token_text_db")
# +======================================================================#

# +=========================Text  Splitter===============================#
print('============Recurrsive text Splitter==================')
class CustomTextSplitter(TokenTextSplitter):
    def split_text(self, text: str):
        return text.split("\n\n")

custom_split=CustomTextSplitter(chunk_size=100,chunk_overlap=0)
chunk_text=custom_split.split_documents(documents=document)
print("========Total Chunks in text splitting======",len(chunk_text))
print("========First Chunk Data======",(chunk_text[0].page_content))

# Create a FAISS Vector Store
custom_db=create_local_vector_store(documents=chunk_text,embedding=embedding_fun,
                                  file_path=custom_text_db_path,store_name="custom_db")
# +======================================================================#


# =======================Query======================================#
def query_results(store_name,query):
    try:
        if store_name=="char_db":
            print("=========Char Db Search=======")
            results=char_db.similarity_search(query=query,k=2)
            for i,doc in enumerate(results):
                print(f"=====Char Document {i}=====")
                print(doc.page_content)

            return results
        elif store_name=="recur_db":
            print("=========recurrsive Db Search=======")
            results=rec_db.similarity_search(query=query,k=2)
            for i,doc in enumerate(results):
                print(f"=====Recurrsive Document {i}=====")
                print(doc.page_content)
            return results
        elif store_name=="token_db":
            print("=========Token Db Search=======")
            results=token_text_db.similarity_search(query=query,k=2)
            for i,doc in enumerate(results):
                print(f"=====Token Document {i}=====")
                print(doc.page_content)
            return results
        elif store_name=="custom_db":
            print("=========Custom Db Search=======")
            results=custom_db.similarity_search(query=query,k=2)
            for i,doc in enumerate(results):
                print(f"=====Custom Document {i}=====")
                print(doc.page_content)
            return results
        else:
            return "No Store Found"
    except Exception as e:
        print(f"Error===> {e}")

query="Korean currency code should be ?"
query_results(query=query,store_name="char_db")
query_results(query=query,store_name="recur_db")
query_results(query=query,store_name="token_db")
query_results(query=query,store_name="custom_db")

