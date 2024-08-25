 #------------------------------------------------Import Packages---------------------------------------------#
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
#-----------------------------------------------------------------------------------------------------------#
load_dotenv()



# make a fun to load the genai model and get resposne
def load_mode_get_respose(question:str):
    llm=GoogleGenerativeAI(google_api_key=os.environ['google_api'],
                           model='gemini-1.5-pro',temperature=0.4)
    
    respose=llm(question)
    yield respose