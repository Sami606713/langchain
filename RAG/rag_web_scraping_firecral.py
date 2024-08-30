#===================================Import Packages================================#
from langchain_community.document_loaders import FireCrawlLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
# fire_cral
#==========================================================================================#
vector_store_path=os.path.join("RAG","Db","web_db2")

print("===========Loading the document============")
url="https://apple.com/"
loader=FireCrawlLoader(api_key=os.environ['fire_cral'],mode='scrape',url=url)
docs=loader.load()

print("========loading data successfully=========")

print("Adding meta data to the document")
for doc in docs:
    for key,value  in doc.metadata.items():
        if isinstance(value,list):
            doc.metadata[key]=",".join(map(str,value))


print("===========Split the document============")
splitter=CharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
doc_chunks=splitter.split_documents(documents=docs)
print("===========document chunks information===========")
print(f"=====Total Chunks: {len(doc_chunks)}")
print("=====First Chunk=====\n",doc_chunks[0].page_content)

print("=================Load the embedding============")
try:
    print('Creating Embedding')
    embedding=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ['cohere_api'])
    
    print("===========Setting the vector db============")
    
    if not os.path.exists(vector_store_path):
        library=FAISS.from_documents(documents=doc_chunks,embedding=embedding)
        print('Saving vector db')
        library.save_local(folder_path=vector_store_path)
    else:
        print("==============Loading vector Db===========")
        library=FAISS.load_local(folder_path=vector_store_path,embeddings=embedding,allow_dangerous_deserialization=True)

except Exception as e:
    print(f"Embeddng Error is {e}")


try:
    print('Query the vector db')
    query="Apple new product"
    print(query)

    retriever=library.as_retriever(search_type="similarity",
                         search_kwargs={"k":2})
    
    results=retriever.invoke(query)

    for i,doc in enumerate(results):
        print(f"======Document{i+1}=====")
        # print(f"=========Meta Data=======\n{doc.metadata}")
        print("=======Content=======")
        print(doc.page_content)

except Exception as e:
    print(f"Query Error is {e}")