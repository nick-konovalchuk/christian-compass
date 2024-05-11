import weaviate
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.chat_engine.llama_cpp_chat_engine import LlamaCPPChatEngine


class ChatRagAgent:
    def __init__(self):
        self._chat_engine = LlamaCPPChatEngine("Phi-3-mini-4k-instruct-q4.gguf")
        self.n_ctx = self._chat_engine.n_ctx
        self._vectorizer = SentenceTransformer(
            "jinaai/jina-embeddings-v2-base-en",
            trust_remote_code=True
        )
        self._reranker = CrossEncoder(
            "jinaai/jina-reranker-v1-turbo-en",
            trust_remote_code=True
        )

        # self._collection = weaviate.connect_to_local(port=8001).collections.get("Collection")

    def chat(self, messages, user_message):
        # embedding = self._vectorizer(user_message).tolist()
        # docs = self._collection.query.near_vector(
        #     near_vector=embedding,
        #     limit=10
        # )
        # docs = self._reranker.rank(user_message, docs, top_k=2)

        return self._chat_engine.chat(messages, user_message)

