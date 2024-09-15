# interfaces.py
from abc import ABC, abstractmethod
from typing import Optional, Callable
from typing import Protocol, Self

from entities import Conversation
from entities import ToolDescription, ChatMessage
from prompt import Prompt


class LLMClient(ABC):
    @classmethod
    @abstractmethod
    def create(cls, **kwargs) -> Self:
        pass

    @abstractmethod
    def generate(self, conversation: Conversation, tools: Optional[list[ToolDescription]] = None) -> Conversation:
        pass


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


class Agent(ConversationParticipant, ABC):
    @classmethod
    @abstractmethod
    def create(
        cls,
        name: str,
        prompt: Prompt,
        llm_client: LLMClient,
        **kwargs,
    ):
        pass

    @abstractmethod
    def invoke(self, conversation: Conversation | str) -> Conversation:
        """Run the agent on a conversation."""
        pass

    @abstractmethod
    def prepare_conversation(self, conversation: Conversation | str) -> Conversation:
        """Prepare the conversation for reasoning, ensuring it is valid."""
        pass
