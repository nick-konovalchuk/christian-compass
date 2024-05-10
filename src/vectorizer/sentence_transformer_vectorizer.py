from sentence_transformers import SentenceTransformer

from src.vectorizer.vectorizer import Vectorizer


class SentenceTransformerVectorizer(Vectorizer):
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def encode(self, doc):
        return self.model.encode(doc)
