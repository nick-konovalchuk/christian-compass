import streamlit as st

from src.chat_engine.llama_cpp_chat_engine import LlamaCPPChatEngine
from src.utils import render_chat_history, get_render_assistant_message


@st.cache_resource(show_spinner=False)
def get_chat_rag_agent():
    return LlamaCPPChatEngine("Phi-3-mini-4k-instruct-q4.gguf")


def pbar_callback():
    pbar.progress(min(st.session_state["ctx_len"] / chat_rag_agent.n_ctx, 1), "Chat history limit")


chat_rag_agent = get_chat_rag_agent()

if "messages" not in st.session_state or st.sidebar.button("Clear chat history"):
    st.session_state["input_blocked"] = False
    st.session_state["messages"] = []
    st.session_state["ctx_len"] = 0

pbar = st.sidebar.progress(
    min(st.session_state["ctx_len"] / chat_rag_agent.n_ctx, 1), "Chat history limit"
)

user_message = st.chat_input(disabled=st.session_state["input_blocked"])
if user_message:
    message_generator, n_tokens = chat_rag_agent.chat(st.session_state["messages"], user_message)
    st.session_state["ctx_len"] += n_tokens
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": user_message
        }
    )
    render_chat_history()
    pbar_callback()

    message, finish_reason = get_render_assistant_message(message_generator, pbar_callback)
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": message
        }
    )
if st.session_state["ctx_len"] >= chat_rag_agent.n_ctx:
    st.session_state["input_blocked"] = True
    user_message = st.chat_input(disabled=st.session_state["input_blocked"])
    st.info("Chat history limit reached")

