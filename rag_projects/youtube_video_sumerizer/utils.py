from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()  



def get_transcript(url):
    """
    This fun is responsible for getting the url and return the video detail.

    input:str:url

    return list:video:data
    """
    # print("===========Loading the youtube transcript============")
    try:
        loader=YoutubeLoader.from_youtube_url(youtube_url=url,
                                            add_video_info=True,
                                            language=["en", "id"],
                                            translation="en")
        content=loader.load()
        return content
    except Exception as e:
        st.error(f"{e}")

def load_llm():
    """
    Load the language model.

    Returns:
        ChatCohere: An instance of the ChatCohere model.
    """
    try:
        model=ChatCohere(cohere_api_key=os.environ['cohere_api'])
        return model
    except Exception as e:
        st.error(f"{e}")

def summerize_content(transcript:str):
     """
    Summarize the provided transcript using the language model.

    Input:
        transcript (str): The transcript text of the video.

    Returns:
        str: The summarized content of the video.
    """
    try:
        system_message = """
        You are an intelligent assistant specialized in summarizing YouTube videos. 
        Your task is to provide a concise and coherent summary of the video content based on the provided transcript. 

        Focus on the following guidelines:
        1. Extract the main ideas and key points discussed in the video.
        2. Capture the overall structure and flow of the content.
        3. Avoid including unnecessary details, repetitions, or filler content.
        4. Maintain the context and meaning of the original content.
        5. The summary should be clear and informative, making it easy for someone to understand the video’s content without watching it.

        The transcript may contain different sections or topics. Please ensure the summary reflects any transitions between these topics.

        Provide the summary in form of bullet points.
        Summary should be in maximum 250-300 words.
        """
        messages=[
                ("system",system_message),
                ("human","{transcript}")
            ]

        prompt_template=ChatPromptTemplate.from_messages(messages)

        # load the llm
        print("Loading the llm")
        llm=load_llm()

        # setting  up the chain
        print("====Settting up the chain")
        chain= prompt_template|llm 

        print("===getting response====")
        

        for chunk in chain.stream({"transcript":transcript}):
            # generate the response
            yield chunk.content
     
    except Exception as e:
        st.error(f"{e}")


# "source":"yF9kGESAi3M"
# "title":"LangChain Master Class For Beginners 2024 [+20 Examples, LangChain V0.2]"
# "description":"Unknown"
# "view_count":91191
# "thumbnail_url":"https://i.ytimg.com/vi/yF9kGESAi3M/hq720.jpg"
# "publish_date":"2024-06-22 00:00:00"
# "length":11870
# "author":"codewithbrandon"
