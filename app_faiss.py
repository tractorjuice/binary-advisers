import os
import re
import uuid
import openai
import promptlayer
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from streamlit_player import st_player

#MODEL = "gpt-3"
#MODEL = "gpt-3.5-turbo"
#MODEL = "gpt-3.5-turbo-0613"
#MODEL = "gpt-3.5-turbo-16k"
MODEL = "gpt-3.5-turbo-16k-0613"
#MODEL = "gpt-4"
#MODEL = "gpt-4-0613"
#MODEL = "gpt-4-32k-0613"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Remove HTML from sources
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_markdown(text):
    # Remove headers (e.g., # Header)
    text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
    # Remove bold/italic (e.g., **bold**, *italic*)
    text = re.sub(r'\*.*\*', '', text)
    # Remove links (e.g., [text](url))
    text = re.sub(r'\[.*\]\(.*\)', '', text)
    # Remove lists (e.g., - item)
    text = re.sub(r'- .*$', '', text, flags=re.MULTILINE)
    return text

def clean_text(text):
    text = remove_html_tags(text)
    text = remove_markdown(text)
    return text

st.set_page_config(page_title="Chat with Binary Advisors")
st.title("Chat with Binary Advisors")
st.sidebar.markdown("# Query Videos using AI")
st.sidebar.markdown("Current Version: 1.0.0")
st.sidebar.divider()
# Check if the user has provided an API key, otherwise default to the secret
user_openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", placeholder="sk-...", type="password")

if user_openai_api_key:
    # If the user has provided an API key, use it
    # Swap out openai for promptlayer
    promptlayer.api_key = st.secrets["PROMPTLAYER"]
    openai = promptlayer.openai
    openai.api_key = user_openai_api_key
else:
    st.warning("Please enter your OpenAI API key", icon="⚠️")

system_template="""
    As a chatbot, analyze the provided videos on AI and offer insights and recommendations.
    Suggestions:
    Discuss the key insights derived from the videos
    Provide recommendations based on the analysis
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that "I don't know", don't try to make up an answer.
    ----------------
    {summaries}"""
prompt_messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
    ]
prompt = ChatPromptTemplate.from_messages(prompt_messages)

# Get datastore
DATASTORE = "datastore"

if user_openai_api_key:
    if os.path.exists(DATASTORE):
        vector_store = FAISS.load_local(
            DATASTORE,
            OpenAIEmbeddings()
        )
    else:
        st.write(f"Missing files. Upload index.faiss and index.pkl files to {DATA_STORE_DIR} directory first")

    chain_type_kwargs = {"prompt": prompt}
    llm = PromptLayerChatOpenAI(
        model_name=MODEL,
        temperature=0,
        max_tokens=1000,
        pl_tags=["binary-advisors-chat", st.session_state.session_id],
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 50}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] in ["user", "assistant"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if user_openai_api_key:
    if query := st.chat_input("What question do you have for the videos?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
    
        with st.spinner():
            with st.chat_message("assistant"):
                response = chain(query)
                st.markdown(response['answer'])
                st.divider()
                
                source_documents = response['source_documents']
                for index, document in enumerate(source_documents):
                    if 'source' in document.metadata:
                        source_details = document.metadata['source']
                        with st.expander(f"Source {index + 1}: {document.metadata['source']}"):
                            st.write(f"Source {index + 1}: {document.metadata['source']}\n")
                            st.write(f"Video title: {document.metadata['title']}")
                            st.write(f"Video author: {document.metadata['author']}")
                            st.write(f"Source video: https://youtu.be/{document.metadata['source_video']}?t={int(document.metadata['start_time'])}")
                            st.write(f"Start Time: {document.metadata['start_time']}")
                            
                        cleaned_content = clean_text(document.page_content)
                        st.write(f"Content: {cleaned_content}\n")
                        video_id = f"Source video: https://youtu.be/{document.metadata['source_video']}?t={int(document.metadata['start_time'])}"
                        key = f"video_{index}"
                        st_player(video_id, height=150, key=key)
    
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
