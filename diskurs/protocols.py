from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Union, Self, TypeVar, Protocol, Type, Optional, Any

from diskurs.entities import ToolDescription, ChatMessage, LongtermMemory, PromptArgument, MessageType, DiskursInput


class LongtermMemoryHandler(Protocol):
    def can_finalize(self, longterm_memory: Any) -> bool:
        pass


class PromptValidator(Protocol):

    @classmethod
    def validate_dataclass(
        cls,
        parsed_response: dict[str, Any],
        user_prompt_argument: Type[dataclass],
        strict: bool = False,
    ) -> dataclass:
        pass

    @classmethod
    def validate_json(cls, llm_response: str) -> dict:
        pass


class Prompt(Protocol):
    """Protocol for rendering templates in prompts."""

    def render_system_template(self, name: str, prompt_args: PromptArgument) -> ChatMessage: ...

    def render_user_template(self, name: str, prompt_args: PromptArgument) -> ChatMessage: ...

    def parse_user_prompt(self, llm_response: str, message_type: MessageType) -> PromptArgument | ChatMessage: ...

    def create_system_prompt_argument(self, **prompt_args: dict) -> PromptArgument: ...

    def create_user_prompt_argument(self, **prompt_args: dict) -> PromptArgument: ...


class MultistepPromptProtocol(Prompt):

    def is_final(self, user_prompt_argument: PromptArgument) -> bool: ...

    def is_valid(self, user_prompt_argument: PromptArgument) -> bool: ...


class ConductorPromptProtocol(Prompt):

    def init_longterm_memory(self) -> Any: ...

    def can_finalize(self, longterm_memory: Any) -> Any: ...

    def finalize(self, longterm_memory: Any) -> Any: ...


class Conversation(Protocol):
    @property
    @abstractmethod
    def chat(self) -> List[ChatMessage]:
        """Provides a deep copy of the chat messages to ensure immutability."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> ChatMessage:
        """Retrieves the system prompt of the conversation."""
        pass

    @property
    @abstractmethod
    def user_prompt(self) -> ChatMessage:
        """Retrieves the user prompt of the conversation."""
        pass

    @property
    @abstractmethod
    def system_prompt_argument(self) -> PromptArgument:
        """Returns the system prompt arguments."""
        pass

    @property
    @abstractmethod
    def user_prompt_argument(self) -> PromptArgument:
        """Returns the user prompt arguments."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, str]:
        """Provides a deep copy of the metadata dictionary to ensure immutability."""
        pass

    @property
    @abstractmethod
    def last_message(self) -> ChatMessage:
        """Retrieves the last message in the conversation."""
        pass

    @abstractmethod
    def get_agent_longterm_memory(self, agent_name: str) -> Optional[LongtermMemory]:
        """Provides a deep copy of the long-term memory dictionary for the specified agent."""
        pass

    @abstractmethod
    def update_agent_longterm_memory(self, agent_name: str, longterm_memory: LongtermMemory) -> Self:
        """Updates the long-term memory for a specific agent and returns a new Conversation instance."""
        pass

    @abstractmethod
    def update(
        self,
        chat: Optional[List[ChatMessage]] = None,
        system_prompt_argument: Optional[PromptArgument] = None,
        user_prompt_argument: Optional[PromptArgument] = None,
        system_prompt: Optional[ChatMessage] = None,
        user_prompt: Optional[Union[ChatMessage, List[ChatMessage]]] = None,
        longterm_memory: Optional[Dict[str, LongtermMemory]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Self:
        """Returns a new instance of Conversation with updated fields, preserving immutability."""
        pass

    @abstractmethod
    def append(
        self,
        message: Union[str, ChatMessage, List[ChatMessage]],
        role: Optional[str] = "",
        name: Optional[str] = "",
    ) -> Self:
        """Appends a new chat message and returns a new instance of Conversation."""
        pass

    @abstractmethod
    def render_chat(self) -> List[ChatMessage]:
        """Returns the complete chat with the system prompt prepended and the user prompt appended."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Checks if the chat is empty."""
        pass

    @abstractmethod
    def has_pending_tool_call(self) -> bool:
        """Checks if there is a pending tool call."""
        pass


class LLMClient(Protocol):
    @classmethod
    def create(cls, **kwargs) -> Self: ...

    def generate(self, conversation: Conversation, tools: Optional[list[ToolDescription]] = None) -> Conversation: ...


class ConversationParticipant(Protocol):
    @property
    def topics(self) -> list[str]: ...

    def process_conversation(self, conversation: Conversation | str) -> None:
        """Actively participate in the conversation by processing and possibly responding."""
        ...

    def register_dispatcher(self, dispatcher: "ConversationDispatcher") -> None: ...

    def start_conversation(self, question: DiskursInput) -> None: ...


class ConversationDispatcher(Protocol):
    def subscribe(self, topic: str, participant: ConversationParticipant) -> None:
        """Subscribe a participant to a specific topic."""
        pass

    def unsubscribe(self, topic: str, participant: ConversationParticipant) -> None:
        """Unsubscribe a participant from a specific topic."""
        pass

    def publish(self, topic: str, conversation: Conversation) -> None:
        """Dispatch a conversation to all participants subscribed to the topic."""
        pass

    def run(self, participant: ConversationParticipant, question: dict) -> dict:
        """Finish the conversation."""
        pass

    def finalize(self, response: dict) -> None:
        """Finalize a conversation by adding metadata or other final touches."""
        pass


GenericAgent = TypeVar("GenericAgent")


class Agent(Protocol[GenericAgent]):

    @classmethod
    def create(cls, name: str, prompt: Prompt, llm_client: LLMClient, **kwargs): ...

    def invoke(self, conversation: Conversation | str) -> Conversation:
        """Run the agent on a conversation."""
        ...


class Conductor(Protocol):
    @classmethod
    def create(cls, name: str, prompt: MultistepPromptProtocol, llm_client: LLMClient, **kwargs) -> Self: ...

    def update_longterm_memory(self, conversation: Conversation, overwrite: bool = False) -> Conversation: ...
