from langchain_community.document_loaders import TextLoader
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings,ChatCohere
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
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
def query_results(query):
    try:
        results=library.similarity_search(query=query,k=2)
        for i,doc in enumerate(results):     
            print("=========",i,"=========")
            print("========MetaData========",doc.metadata)
            print("========Docs========\n",doc.page_content)
        return results
    except Exception as e:
        print(f"Query Error is {e}")

query="KRW is the currency code of ?"
results=query_results(query)

# combine the query and relevent document
combine_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n\n"
    + "\n\n".join([doc.page_content for doc in results])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found, simply respond with 'I don't know'."
)

# ===================load the llm model======================#
try:
    # llm=HuggingFaceEndpoint(
    #     repo_id="HuggingFaceH4/zephyr-7b-beta",
    #     task="text-generation",
    #     max_new_tokens=512,
    #     do_sample=False,
    #     repetition_penalty=1.03,
    # )
    # chat_model = ChatHuggingFace(llm=llm)
    print("Loading Model....")
    model=ChatCohere(cohere_api_key=os.environ['cohere_api'])
    print("Model Loaded Successfully....")
except Exception as e:
    print(f"Hugging Face Model Error {e}")

# Make a prompt template
messages=[
    SystemMessage(content="you are a helpful assistant."),
    HumanMessage(content=combine_input)
]
# Use the llm model for generating result
result=model.invoke(messages)
print("============Generating Reesult===========")
print("========Full Content========")
print(result.content)