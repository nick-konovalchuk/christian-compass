from abc import ABC, abstractmethod


class Vectorizer(ABC):
    @abstractmethod
    def encode(self, doc):
        pass
