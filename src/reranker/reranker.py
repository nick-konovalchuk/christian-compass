from abc import ABC, abstractmethod


class Reranker(ABC):
    @abstractmethod
    def rank(self, query, docs):
        pass
