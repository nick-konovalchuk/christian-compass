# Christian Compass
This is a Christian Apologetic chatbot prototype.  
It runs fully on CPU.  
It's built from following components:
- Streamlit interface
- llama-cpp-python for LLM inference
- Jina's vectorizer and reranker with HuggingFace's Sentence Transformers
- Weaviate and it's free sandbox (you can also run it locally)

## Download the model
You can download the models following way
```bash
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-q4.gguf --local-dir . --local-dir-use-symlinks False
```
