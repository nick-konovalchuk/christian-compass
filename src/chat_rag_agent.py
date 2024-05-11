import os

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
            trust_remote_code=True,
        )

        self._collection = weaviate.connect_to_wcs(
            cluster_url=os.getenv("WCS_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_KEY")),
        ).collections.get("Collection")

    def chat(self, messages, user_message):
        embedding = self._vectorizer.encode(user_message).tolist()
        docs = self._collection.query.near_vector(
            near_vector=embedding,
            limit=10
        )
        ranks = self._reranker.rank(
            user_message,
            [i.properties['answer'] for i in docs.objects],
            top_k=2,
            apply_softmax=True
        )
        context = [
            f"""\
            Question: {docs.objects[rank['corpus_id']].properties['question']}
            Answer: {docs.objects[rank['corpus_id']].properties['answer']}
            """
            for rank in ranks if rank["score"] > 0.2
        ]

        sources = [
            docs.objects[rank['corpus_id']].properties['link']
            for rank in ranks if rank["score"] > 0.2
        ]
        return self._chat_engine.chat(messages, user_message, context), sources
