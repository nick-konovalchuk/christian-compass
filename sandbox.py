# from llama_cpp import Llama, llama_chat_format
#
# llm = Llama(
#     model_path="./Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
#     n_ctx=0,
#     n_threads=8,
#     # stream=True
# )
#
# formatter = llama_chat_format.Jinja2ChatFormatter(
#     template=llm.metadata['tokenizer.chat_template'],
#     bos_token=llm._model.token_get_text(int(llm.metadata['tokenizer.ggml.bos_token_id'])),
#     eos_token=llm._model.token_get_text(int(llm.metadata['tokenizer.ggml.eos_token_id'])),
#     stop_token_ids=llm.metadata['tokenizer.ggml.eos_token_id']
# )
#
# text = formatter(messages=[
#     {"role": "user", "content": "Hello"},
#     # {"role": "assistant", "content": "world"},
#     # {"role": "user", "content": "exactly!"},
#
# ]).prompt
#
# tokens = llm.tokenizer().encode(text)
#
# output = llm.create_completion(
#     tokens,
#     max_tokens=256,
#     stop=["<|end|>"],
#     # stream=True
# )
#
# print(output)
# print(1)

# from langchain_community.llms import LlamaCpp
# from langchain_community.chat_models import ChatOllama
#
# llm = LlamaCpp(
#     model_path="./Phi-3-mini-4k-instruct-q4.gguf",
#     n_threads=8,
#     # stream=True
# )
#
# a = llm.invoke("Hello")
#
# print(1)

from src.chat_rag_agent import ChatRagAgent
import warnings

# warnings.filterwarnings("ignore")
#
#
# from dotenv import load_dotenv
#
# load_dotenv()
#
# ChatRagAgent()

import streamlit as st

st.title('Sandbox')