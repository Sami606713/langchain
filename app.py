#------------------------------------------------Import Packages---------------------------------------------#
from langchain_google_genai import GoogleGenerativeAI
import streamlit as st
from utils import load_mode_get_respose
import os
#-----------------------------------------------------------------------------------------------------------#

st.set_page_config(
    page_title="QnA Chat Bot",  
    page_icon="🤖",            
    layout="centered",          
    initial_sidebar_state="auto"  
)

st.title("QnA Chat Bot")

question=st.text_input("Enter Question:\n")
resposne=load_mode_get_respose(question=question)
if st.button('Reponse'):
    st.write_stream(resposne)


