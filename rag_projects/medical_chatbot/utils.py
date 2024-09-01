# =================Import packages=====================#
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st
import os
# ====================================================#

def load_document(file_path):
    """
    This fun is responsible for loading the document
    
    Args:
        path (str): The path to the document
    return:
        document (str): The document content
    """
    try:
        loader=PyPDFLoader(file_path)

        document=loader.load()

        return document
    except Exception as e:
        return str(e)


def split_document(documents,chunk_size,chunk_overlap):
    """
    This fun is responsible for splitting the document into chunks
    
    Args:
        documents (list): The list of documents
    return:
        document_chunks (list): The list of document chunks
    """
    try:
        splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        document_chunks=splitter.split_documents(documents=documents)

        return document_chunks
    except Exception as e:
        return str(e)

def create_vector_store(vector_store_path,doc,embedding):
    """
    This fun is responsible for creating the vector store
    
    Args:
        vector_store_path (str): The path to the vector store
    return:
        library (object): The vector store object
    """
    try:
        if not os.path.exists(vector_store_path):
            # print("==========Creating Vector Store Folder=============")
            os.makedirs(vector_store_path)

            library=FAISS.from_documents(documents=doc,embedding=embedding)
            
            # print("==========Saving Vector Store=============")
            library.save_local(vector_store_path)
           
            return library
        else:
            # print("==========Loading Vector Store=============")
            library=FAISS.load_local(folder_path=vector_store_path,embeddings=embedding,
                                    allow_dangerous_deserialization=True)
            return library
    except Exception as e:
        return str(e)
    

def create_rag_chain(llm,retriever):
    try:
        # make a sytem message
        system_message=(
            "Given a chat history and the latest user question"
        "which might reference context in the history"
        "formulate the standalone question which can be understood"
        "with chat history donot answer the question, just"
        "reformulate it if need and otherwise return it as is."
        )

        # cousomize the prompt
        coustomize_system_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_message),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        # set the history aware retriever
        history_aware_retriever=create_history_aware_retriever(
            llm=llm,retriever=retriever,prompt=coustomize_system_prompt
        )

        # make a question_answer prompt
        q_a_prompt=(
            "you are a assistant for medial related question answering task."
            "Use the following piece of retriever context to answer the question."
            "if you dont't know the answer, just saty i don't know."
            "Use three sentecen maximum and keep the answer concise.\n\n"
            "{context}"
        )

        # coustoize the Q&A prompt
        final_qa_prompt=ChatPromptTemplate(
            [
                ("system",q_a_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        # create a document stuff
        question_answer_chain=create_stuff_documents_chain(llm,final_qa_prompt)

        # make the rag chain
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        return rag_chain
    except Exception as e:
        return str(e)

def get_response(rag_chain,query):
    """
    This fun is responsible for continuing the chat
    
    Args:
        rag_chain (object): The rag chain object
        query (str): The user query
    return:
        response (str): The response from the chatbot
    """
    try:
        st.session_state['chat_history'].append(HumanMessage(content=query))
        response=rag_chain.invoke({
            "input":query,'chat_history':st.session_state['chat_history']
        })
            
        st.session_state['chat_history'].append(SystemMessage(content=response['answer']))

        return response['answer']
            
    except Exception as e:
        print(str(e))


def load_previous_chat():
    for i in range(len(st.session_state['chat_history'])):
        if i%2==0:
            st.write(f'💬 You: {st.session_state['chat_history'][i].content}') 
        else:
            st.write(f'🤖 Bot: {st.session_state['chat_history'][i].content}')  
    