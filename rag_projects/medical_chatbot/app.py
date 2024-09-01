# =================Import packages=====================#
from utils import (load_document,split_document,create_vector_store,
                    create_rag_chain,get_response,load_previous_chat)
from langchain_cohere import CohereEmbeddings,ChatCohere
import streamlit as st
import os
# ====================================================#
if "chat_history" not  in st.session_state:
    st.session_state['chat_history']=[]
# Set the page configuration
st.set_page_config(
    page_title="Medical Chat Bot",  # Title of the web page
    page_icon="🎥",  # Icon for the web page (you can use any emoji)
    layout="centered",  # Layout of the page (either 'centered' or 'wide')
    initial_sidebar_state="auto"  # Initial state of the sidebar ('auto', 'expanded', 'collapsed')
)

# Set the page title
st.title("🩺 Medical Chat Bot")

# load document
file_path = "rag_projects/medical_chatbot/data/Medical_book.pdf"
vector_store_path="rag_projects/medical_chatbot/db"

# ========Load Document========
# documents=load_document(file_path=file_path)


# ========Splitting Document========
# document_chunks=split_document(documents=documents,chunk_size=1000,chunk_overlap=50)



# =======Load Embedding==========
embedding=CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=os.environ['cohere_api'])
    
# ============Creating Vector Store============
document_chunks="This is a test document"

library=create_vector_store(vector_store_path=vector_store_path,doc=document_chunks,embedding=embedding)

# Set the retriever
retriever=library.as_retriever(search_type="similarity",
            search_kwargs={"k":3})


# ============Creating RAG Chain============
llm=ChatCohere(cohere_api_key=os.environ['cohere_api'])

rag_chain=create_rag_chain(llm=llm,retriever=retriever)

# get_response(rag_chain)

prompt=st.chat_input(placeholder="Enter you question? ")

with st.container(border=False):
    load_previous_chat()

if prompt:
    st.write(f'💬 You: {prompt}') 
        
    response = get_response(rag_chain, prompt)

    st.write(f'🤖 Bot: {response}')  