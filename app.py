import streamlit as st
from llama_cpp import Llama


@st.cache_resource
def get_model():
    return Llama(
        model_path="./Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx=4096,
        n_threads=8,
        stream=True,
        verbose=False
    )


st.title("Christian Compass")

llm = get_model()

user_message = st.chat_input()

if user_message:
    with st.chat_message("user", avatar='👤'):
        st.write(user_message)
    with st.chat_message("ai", avatar='🧭'):
        output = llm.create_chat_completion(
            [
                {"role": "user", "content": user_message}
            ],
            stop=["<|end|>"],
            stream=True
        )
        st.write_stream(item['choices'][0]['delta']['content'] for item in output if
                        'content' in item['choices'][0]['delta'])
        st.caption("AI can make mistakes. Please, check the references")
