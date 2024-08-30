from langchain_community.document_loaders import TextLoader
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import os
load_dotenv()


# Load the document
file_path = "RAG/doc/wikipedia.txt"
vector_store_path="RAG/Db/recur_db"

loader=TextLoader(file_path=file_path)
documents=loader.load()

# Split the text into chunks
splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=0)
chunk_documents=splitter.split_documents(documents=documents)

# Convert the chunks into embedding
try:
    embedd_fun=CohereEmbeddings(model="embed-english-light-v3.0",
                                cohere_api_key=os.environ['cohere_api'])
    
    library=FAISS.load_local(folder_path=vector_store_path,embeddings=embedd_fun,allow_dangerous_deserialization=True) 
except Exception as e:
    print(f"Error is {e}")



# Query the vector store
def query_results(query,searh_type):
    try:
        if searh_type=="similarity":
            print("=========Similarity Search=========")
            results=library.similarity_search(query=query,k=2)
            for i,doc in enumerate(results):     
                print("=========",i,"=========")
                print("========MetaData========",doc.metadata)
                print("========Docs========\n",doc.page_content)
            return results
        elif searh_type=="similarity_score":
            print("=========Similarity Search With score=========")
            results=library.similarity_search_with_score(query=query,k=2)
            for i,doc in enumerate(results):
                print("=========",i,"=========")
                print("========Meta Data========",doc[0].metadata)
                print("========Score========",doc[1])
                print("========Docs========\n",doc[0].page_content)
            return results
        elif searh_type=="MMR":
            print("=========MMR=========")
            results=library.max_marginal_relevance_search(query=query,k=2,lambda_mult=0.5)
            for i,doc in enumerate(results):
                print("=========",i,"=========")
                print("========MetaData========",doc.metadata)
                print("========Docs========\n",doc.page_content)
            return results
    except Exception as e:
        print(f"Query Error is {e}")

query="Korean currency code should be ?"
results=query_results(query,searh_type="similarity")
results_score=query_results(query,searh_type="similarity_score")
mmr=query_results(query,searh_type="MMR")
