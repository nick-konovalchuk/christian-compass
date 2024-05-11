import warnings

import streamlit as st
from dotenv import load_dotenv

from src.chat_rag_agent import ChatRagAgent
from src.utils import render_chat_history, get_render_assistant_message

warnings.filterwarnings("ignore")

load_dotenv()

st.set_page_config(
    page_icon="images/logo2.png",
    initial_sidebar_state="collapsed"

)


@st.cache_resource(show_spinner=False)
def get_chat_rag_agent():
    return ChatRagAgent()


def calc_progress_perc():
    return min(round(st.session_state["ctx_len"] / chat_rag_agent.n_ctx * 100), 100)


def pbar_callback():
    pbar.progress(calc_progress_perc(), "Chat history limit")


with st.spinner("Engine loading"):
    chat_rag_agent = get_chat_rag_agent()

if "messages" not in st.session_state or st.sidebar.button("Clear chat history"):
    st.session_state["input_blocked"] = False
    st.session_state["messages"] = []
    st.session_state["ctx_len"] = 0
    st.title("Christian compass")
    st.markdown("What theological questions you have?")

pbar = st.sidebar.progress(calc_progress_perc(), "Chat history limit")

user_message = st.chat_input(disabled=st.session_state["input_blocked"])
if user_message:
    if not st.session_state["input_blocked"]:
        (message_generator, n_tokens), sources = chat_rag_agent.chat(
            st.session_state["messages"],
            user_message
        )
        st.session_state["ctx_len"] = n_tokens
        st.session_state["messages"].append(
            {
                "role": "user",
                "content": user_message
            }
        )
    render_chat_history()
    if not st.session_state["input_blocked"]:
        pbar_callback()
        message = get_render_assistant_message(message_generator, sources, pbar_callback)
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": message
            }
        )
if st.session_state["ctx_len"] >= chat_rag_agent.n_ctx:
    st.session_state["input_blocked"] = True
    st.info("Chat history limit reached")
