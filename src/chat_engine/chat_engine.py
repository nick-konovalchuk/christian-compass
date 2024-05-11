from abc import ABC, abstractmethod


class ChatEngine(ABC):
    @abstractmethod
    def chat(self, messages, user_message, docs):
        pass
