#===================================Import Packages================================#
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
#==========================================================================================#
vector_store_path=os.path.join("RAG","Db","web_db")

print("===========Loading the document============")
url="https://en.wikipedia.org/wiki/Python_(programming_language)"
loader=WebBaseLoader(web_path=url)
# content=loader.load()

# print(content)
print("===========Split the document============")
splitter=CharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
# doc_chunks=splitter.split_documents(documents=content)

print("=================Load the embedding============")
try:
    embedding=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ['cohere_api'])

    print("setting the vector db")
    if not os.path.exists(vector_store_path):
        # library=FAISS.from_documents(documents=doc_chunks,embedding=embedding)
        print('Saving vector db')
        # library.save_local(folder_path=vector_store_path)
    else:
        print("==============Loading vector Db===========")
        library=FAISS.load_local(folder_path=vector_store_path,
                                 embeddings=embedding,allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Embeddng Error is {e}")


try:
    print("Query the vector db")
    query="python was invinted by whom"
    print(query)
    retriever=library.as_retriever(search_type="similarity",
                         search_kwargs={"k":2})
    
    results=retriever.invoke(query)
    # print("Results=====>\n",results)

    for i,doc in enumerate(results):
        print(f"======Document{i+1}=====")
        print(f"=========Meta Data=======\n{doc.metadata}")
        print("=======Content=======")
        print(doc.page_content)
except Exception as e:
    print(f"Query Error is {e}")