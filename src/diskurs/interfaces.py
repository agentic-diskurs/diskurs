from typing import Optional
from typing import Protocol, Self

from entities import Conversation
from entities import ToolDescription
from prompt import Prompt


class LLMClient(Protocol):
    @classmethod
    def create(cls, **kwargs) -> Self: ...

    def generate(self, conversation: Conversation, tools: Optional[list[ToolDescription]] = None) -> Conversation: ...


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


class ConversationParticipant(Protocol):
    @property
    def topics(self) -> list[str]: ...

    def process_conversation(self, conversation: Conversation) -> None:
        """Actively participate in the conversation by processing and possibly responding."""
        ...

    def register_dispatcher(self, dispatcher: ConversationDispatcher) -> None: ...


class Agent(Protocol):

    @classmethod
    def create(cls, name: str, prompt: Prompt, llm_client: LLMClient, **kwargs) -> Self: ...

    def invoke(self, conversation: Conversation | str) -> Conversation:
        """Run the agent on a conversation."""
        ...


class Conductor(Protocol):
    @classmethod
    def create(cls, name: str, prompt: Prompt, llm_client: LLMClient, **kwargs) -> Self: ...

    def update_longterm_memory(self, conversation: Conversation, overwrite: bool = False) -> Conversation: ...
