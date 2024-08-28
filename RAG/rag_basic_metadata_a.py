#=================================Import Packages====================================================#
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (HumanMessage,AIMessage,SystemMessage)
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.text import TextLoader
from dotenv import load_dotenv
import os
load_dotenv()
#====================================================================================================#

#------------------------------------------------------------------------------------------------#

# Steps to run the code
# 1- Load the document
# 2- Split the document into chunks
# 3- Convert the chunks into embedding
# 4- Save the embedding into vector db
# ------------------------------------------------------------------------------------------------#
doc_dir="RAG/doc"

full_doc=[doc for doc in os.listdir(doc_dir) if doc.endswith(".txt")]

print("===========All Documents List==========",full_doc)

documents=[]
for file in full_doc:
    curr_doc=os.path.join(doc_dir,file)
    print("===========Current Document==========",curr_doc)
    # load the document
    text_loader=TextLoader(file_path=curr_doc)
    docs=text_loader.load()

    # Adding meta data to the document
    for doc in docs:
        doc.metadata={"source":file,"type":"text","name":"wikipedia"}
        documents.append(doc)
    
# check the meta data
print(documents[0].metadata)


# Convert the text into chunks
doc_spliter=CharacterTextSplitter(chunk_size=100,chunk_overlap=0)
doc_chunks=doc_spliter.split_documents(documents=documents)

print("===========Document Chunks Information===========")
print(f"=====Total Chunks: {len(doc_chunks)}")
print("=====First Chunk=====\n",doc_chunks[0].page_content)

# Convert the text into embedding
try:
    embeding_fun=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ["cohere_api"])
except Exception as e:
    print(f"This Error Occur: => {e}")

# Create a FAISS Vector Store
try:
    library=FAISS.from_documents(documents=doc_chunks,embedding=embeding_fun)
    library.save_local(folder_path=os.path.join("RAG/Db","wikipedia_db"))
    print("===========Vectore Store Created Successfully=========")
except Exception as e:
    print(f"========Vector Db error ==> {e}")

# =====================================Query===============================================#
try:
    query="In 1973, whon decided to develop codes for the representation of currencies"
    # Search the query
    results=library.similarity_search(query=query,k=3,threshold=0.5)

    for i,doc in enumerate(results):
        print(f"======Document====={i}")
        print(doc.page_content)
except Exception as e:
    print(f"========Query Error ==> {e}")