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

    def run(self, participant: "ConversationParticipant", question: str) -> dict:
        """Finish the conversation."""
        pass

    def finalize(self, conversation: dict) -> None:
        """Finalize a conversation by adding metadata or other final touches."""
        pass


class ConversationParticipant(ABC):
    @property
    def topics(self) -> list[str]:
        return self._topics

    @topics.setter
    def topics(self, value: list[str]) -> None:
        self._topics = value

    def register_dispatcher(self, dispatcher: ConversationDispatcher) -> None:
        self.dispatcher = dispatcher

    @abstractmethod
    def process_conversation(self, conversation: Conversation) -> None:
        """Actively participate in the conversation by processing and possibly responding."""
        pass


class ConversationFinalizer(ABC):

    @abstractmethod
    def can_finalize(self, conversation: Conversation) -> bool:
        """Check if a conversation can be finalized."""
        pass

    @abstractmethod
    def finalize(self, conversation: Conversation) -> dict:
        """Format the final answer as a dict."""
        pass


class Agent(ABC):
    name: str
    prompt: Prompt

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
