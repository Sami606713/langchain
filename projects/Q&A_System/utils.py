from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import pandas as pd
import os

# Fun that can make a txt for each label
def get_txt_files(df,lables):
    """
    1- This function takes the dataframe and the label as input
    2- It filters the dataframe based on the label
    3- It creates a txt file for each label
    """
    save_path=os.path.join('projects/Q&A_System',"doc")
    df=df[df['labels']==lables]['data']
    
    for i in df:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            if not os.path.exists(os.path.join(save_path,lables+".txt")):
                with(open(os.path.join(save_path,lables+".txt"),"a+")) as f:
                    f.write(i+"\n")

    print(f"Files for {lables} are created successfully")
    return None

def load_document(file_path):
    """
    1- This function loads the txt files
    2- It returns the list of txt files
    """
    documents=[]
    files=os.listdir(file_path)
    # print(files)
    for file in files:
        print(f"========{file}==========")
        curr_file_path=os.path.join(file_path,file)
        doc_loader= TextLoader(curr_file_path)
        docs=doc_loader.load()
        # Adding meta data to the document
        print(f"===={file} Adding Meta Data====")
        for doc in docs:
            doc.metadata={"source":file,"type":"text","name":file.split(".")[0]}
            documents.append(doc)

    return documents


# Create vector store
def create_vector_store(store_name,docs,embedding,file_path):
    vector_db_path=os.path.join(file_path,"Db")
    print("==========Creating Vector Store Folder=============")
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path)

    print("=====FIASS Object====")
    save_path=os.path.join(vector_db_path,store_name)
        
    if not os.path.exists(save_path):
        library = FAISS.from_documents(documents=docs, embedding=embedding)
        library.save_local(folder_path=save_path)
        print(f"==========={store_name} Store Created Successfully=========")
        
    else:
        print(f"========={store_name} Store Already Exist=========")
        
        library=FAISS.load_local(save_path,embeddings=embedding,allow_dangerous_deserialization=True)
    return library

def get_similar(query,vector_store):
    print('=======Getting Similar==========')
    results=vector_store.similarity_search(query=query,k=2)

    for i,doc in enumerate(results):
        print(f"======={i}=======")
        print(f"====MeataData=={doc.metadata}==")
        print(f"====Documents===\n{doc.page_content}")
    return results

# Get the query results
def make_rag_chain(llm,retriever):
    # Set the custom prompt
    print("Set the custom prompt")
    coustom_sys_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the history"
        "formulate the standalone question which can be understood"
        "with chat history donot answer the question, just"
        "reformulate it if need and otherwise return it as is."
    )

    # Customize the prompt
    coustomize_sys_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",coustom_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    # Make the history aware retrieve
    print("Set the history aware retriever")
    history_aware_retriever=create_history_aware_retriever(llm=llm,
                                                           retriever=retriever,
                                                           prompt= coustomize_sys_prompt)

    # make the question prompt
    print("Set the question prompt")
    ques_promt=(
        "you are a assistant for question answering task."
        "Use the following piece of retriever context to answe the question."
        "if you dont't know the answer, just saty i don't know."
        "Use three sentecen maximum and keep the answer concise.\n\n"
        "{context}"
    )
    # Coustomize the question prompt
    coustize_ques_prompt=ChatPromptTemplate(
        [
            ("system",ques_promt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    # Make a question chain
    print("Set the quesiton answer chain")
    question_answer_chain=create_stuff_documents_chain(llm,coustize_ques_prompt)

    # make the rag_chain
    print("Set the retriever chain")
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    return rag_chain


def continue_chat(rag_chain):
    try:
        chat_history=[]
        print("Start Chating With Ai and press 'q' to quit")

        while True:
            query=input("You: ")
            if query=="q":
                break
            
            response=rag_chain.invoke({"chat_history":chat_history,"input":query})
            print(f"AI: {response['answer']}")

            chat_history.append(HumanMessage(content=query))
            chat_history.append(SystemMessage(content=response['answer']))

    except Exception as e:
        print("Error in continue chat is ",e)                                 