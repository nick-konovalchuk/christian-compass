import streamlit as st

from src.const import CHAT_AVATARS


def render_chat_history():
    for message in st.session_state["messages"]:
        with st.chat_message(message['role'], avatar=CHAT_AVATARS[message['role']]):
            st.write(message['content'])


def get_render_assistant_message(message_generator, sources, callback):
    assistant_message = []

    def gen_patched():
        for chunk in message_generator:
            st.session_state["ctx_len"] += 1
            callback()
            text = chunk['choices'][0]["text"]
            assistant_message.append(text)
            yield text
    with st.chat_message('assistant', avatar=CHAT_AVATARS['assistant']):
        st.write_stream(gen_patched())
        for source in sources:
            st.write(source)
        st.caption("AI can make mistakes. Please, fact check the answers")
    return "".join(assistant_message)
