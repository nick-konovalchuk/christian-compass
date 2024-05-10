from InstructorEmbedding import INSTRUCTOR

from src.vectorizer.vectorizer import Vectorizer


class InstructorVectorizer(Vectorizer):
    def __init__(self):
        self.model = INSTRUCTOR('hkunlp/instructor-xl')
        self.instruction = "Represent theology sentence:"

    def encode(self, doc):
        return self.model.encode([self.instruction, doc])
