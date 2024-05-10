import streamlit as st

from src.chat_engine.llama_cpp_chat_engine import LlamaCPPChatEngine
from src.utils import render_chat_history, get_render_assistant_message


@st.cache_resource(show_spinner=False)
def get_chat_engine():
    return LlamaCPPChatEngine("Phi-3-mini-4k-instruct-q4.gguf")


def pbar_callback():
    pbar.progress(st.session_state["ctx_len"] / chat_engine.n_ctx, "Chat history limit")


chat_engine = get_chat_engine()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["ctx_len"] = 0
pbar = st.sidebar.progress(st.session_state["ctx_len"] / chat_engine.n_ctx, "Chat history limit")

user_message = st.chat_input()
if user_message:
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": user_message
        }
    )
    render_chat_history()
    message_generator, n_tokens = chat_engine.chat(st.session_state["messages"], None)
    st.session_state["ctx_len"] = n_tokens / chat_engine.n_ctx
    pbar_callback()

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": get_render_assistant_message(message_generator, pbar_callback)
        }
    )
