#======================================Import Packages====================================================#
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (HumanMessage,AIMessage,SystemMessage)
from langchain_core.prompts import ChatPromptTemplate 
from langchain.text_splitter import (RecursiveCharacterTextSplitter)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from  dotenv import load_dotenv
import os
load_dotenv()

def create_vector_store(store_name,doc,embedding,file_path):
    try:
        library=FAISS.from_documents(documents=doc,embedding=embedding)
        if not os.path.exists(file_path):
            library.save_local(folder_path=file_path)
            print(f"======{store_name} created successfully======")
        else:
            return library
    except Exception as e:
        return e
#==========================================================================================================#
# Embedding https://python.langchain.com/v0.2/docs/integrations/text_embedding/
# Steps to run the code
# 1- Load the document
# 2- Split the document into chunks
# 3- Convert the chunks into embedding
# 4- Save the embedding into vector db

# Load the document
file_path = "RAG/doc/wikipedia.txt"
paid_embed_path=os.path.join("RAG/Db","paid_emb_db")
free_embed_path=os.path.join("RAG/Db","free_emb_db")

#==========================Loading Data=======================================#
print("Loading Data")
doc_loader=TextLoader(file_path=file_path)
document=doc_loader.load()

#==========================Split the text into chunks=======================================#
print("Splitting Text")
splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
chunk_doc=splitter.split_documents(documents=document)

#==================================Text to embedding(Cohere)==============================================#
try:
    embedd_fun=CohereEmbeddings(model="embed-english-light-v3.0",
                                cohere_api_key=os.environ['cohere_api'])
    
    # Create Vector Store
    paid_embed=create_vector_store(store_name="Paid embedding",doc=chunk_doc,
                    embedding=embedd_fun,file_path=paid_embed_path)
except Exception as e:
    print(f"Error is {e}")

#==================================Text to embedding(hugging face)==============================================#
try:
    fake_embedding=FakeEmbeddings(size=384)
    # Create vectoer Store
    free_embed=create_vector_store(store_name="Free Fake embedding",doc=chunk_doc,
                    embedding=fake_embedding,file_path=free_embed_path)

except Exception as e:
    print(f"Error is {e}")

#==================================Query==============================================#
def query_results(query,store_name):
    if store_name=="Free Fake embedding":
        print("Query Search in Fake Embedding")
        results=free_embed.similarity_search(query=query,k=2)
        for i,doc in enumerate(results):
            print(f"=======Free Embed====={i}")
            print("=====Doc====\n",doc.page_content)

    elif store_name=="Paid embedding":
        print("Query Search in Paid Embedding")
        results=paid_embed.similarity_search(query=query,k=2)
        for i,doc in enumerate(results):
            print(f"=======Free Embed====={i}")
            print("=====Doc====\n",doc.page_content)

query="Korean currency code should be ?"
query_results(query=query,store_name="Paid embedding")
query_results(query=query,store_name="Free Fake embedding")