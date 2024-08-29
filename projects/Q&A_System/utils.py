from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
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