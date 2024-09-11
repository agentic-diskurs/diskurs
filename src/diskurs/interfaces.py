from typing import Protocol

from entities import Conversation


class ConversationDispatcher(Protocol):
    def subscribe(self, topic: str, participant: "ConversationParticipant") -> None:
        """Subscribe a participant to a specific topic."""
        pass

    def unsubscribe(self, topic: str, participant: "ConversationParticipant") -> None:
        """Unsubscribe a participant from a specific topic."""
        pass

    def publish(self, topic: str, conversation: Conversation) -> None:
        """Dispatch a conversation to all participants subscribed to the topic."""
        pass


class ConversationParticipant(Protocol):
    dispatcher: "ConversationDispatcher"

    def register_dispatcher(self, dispatcher: ConversationDispatcher) -> None:
        pass

    def process_conversation(self, conversation: Conversation) -> None:
        """Actively participate in the conversation by processing and possibly responding."""
        pass

class LLMClient(Protocol):
    @classmethod
    def create(cls, api_key: str, model: str) -> "LLMClient":
        pass