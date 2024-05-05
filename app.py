import streamlit as st
from llama_cpp import Llama

llm = Llama(
    model_path="./Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
    n_ctx=4096,
    n_threads=8,
    stream=True
)

user_message = st.chat_input()

if user_message:
    with st.chat_message("ai"):
        output = llm.create_completion(
            f"<|user|>\n{user_message}<|end|>\n<|assistant|>",
            max_tokens=256,  # Generate up to 256 tokens
            stop=["<|end|>"],
            stream=True
        )
        st.write_stream(item['choices'][0]['text'] for item in output)

