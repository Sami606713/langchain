import streamlit as st
from utils import get_transcript,summerize_content


# Set the page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",  # Title of the web page
    page_icon="🎥",  # Icon for the web page (you can use any emoji)
    layout="centered",  # Layout of the page (either 'centered' or 'wide')
    initial_sidebar_state="auto"  # Initial state of the sidebar ('auto', 'expanded', 'collapsed')
)

# Set the title of the app with an emoji
st.title("🎬 YouTube Video Summarizer")

# add a input field
yt_url=st.text_input("Enter youtube video link: ")
# https://www.youtube.com/watch?v=yF9kGESAi3M&t=9028s
if  yt_url:
    with st.spinner("Getting video transcript..."):
        response=get_transcript(yt_url)

        content=response[0].page_content

        meta_data=response[0].metadata

    if st.button("Summerize:"):
        with st.spinner("Summerizing..."):
            st.image(meta_data['thumbnail_url'])
            response=summerize_content(content)

            st.header("Summary")
            st.write(response)