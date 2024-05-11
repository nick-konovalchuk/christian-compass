from abc import ABC, abstractmethod


class ChatEngine(ABC):
    @abstractmethod
    def chat(self, messages):
        pass
