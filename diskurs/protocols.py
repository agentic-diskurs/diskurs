from dataclasses import dataclass
from typing import List, Dict, Union, Self, TypeVar, Protocol, Type, Optional, Any, Generic

from diskurs.entities import ToolDescription, ChatMessage, LongtermMemory, PromptArgument, MessageType


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

    def fail(self, longterm_memory: Any) -> Any: ...


SystemPromptArgT = TypeVar("SystemPromptArgT")
UserPromptArgT = TypeVar("UserPromptArgT")


class Conversation(Protocol[SystemPromptArgT, UserPromptArgT]):

    @property
    def chat(self) -> List[ChatMessage]:
        """
        Provides a deep copy of the chat messages to ensure immutability.

        :return: A deep copy of the list of chat messages.
        """
        ...

    @property
    def system_prompt(self) -> Optional[ChatMessage]:
        """
        Retrieves the system prompt of the conversation.

        :return: The system prompt message.
        """
        ...

    @property
    def user_prompt(self) -> Optional[ChatMessage]:
        """
        Retrieves the user prompt of the conversation.

        :return: The user prompt message.
        """
        ...

    @property
    def system_prompt_argument(self) -> Optional[SystemPromptArgT]:
        """
        Retrieves the system prompt arguments.

        :return: The system prompt arguments.
        """
        ...

    @property
    def user_prompt_argument(self) -> Optional[UserPromptArgT]:
        """
        Retrieves the user prompt arguments.

        :return: The user prompt arguments.
        """
        ...

    @property
    def metadata(self) -> Dict[str, str]:
        """
        Provides a deep copy of the metadata dictionary to ensure immutability.

        :return: A deep copy of the metadata dictionary.
        """
        ...

    @property
    def last_message(self) -> ChatMessage:
        """
        Retrieves the last message in the conversation.

        :return: The last message in the chat.
        """
        ...

    @property
    def active_agent(self) -> str:
        """
        Retrieves the name of the active agent.

        :return: The name of the active agent.
        """
        ...

    @property
    def conversation_id(self) -> str:
        """
        Retrieves the conversation ID.

        :return: The conversation ID.
        """
        ...

    def get_agent_longterm_memory(self, agent_name: str) -> Optional[LongtermMemory]:
        """
        Provides a deep copy of the long-term memory for the specified agent.

        :param agent_name: The name of the agent.
        :return: A deep copy of the agent's long-term memory.
        """
        ...

    def update_agent_longterm_memory(self, agent_name: str, longterm_memory: LongtermMemory) -> "Conversation":
        """
        Updates the long-term memory for a specific agent.

        :param agent_name: The name of the agent.
        :param longterm_memory: The new long-term memory for the agent.
        :return: A new instance of the Conversation with updated long-term memory.
        """
        ...

    def update_prompt_argument_with_longterm_memory(self, conductor_name: str) -> "Conversation":
        """
        Updates the prompt arguments with the long-term memory of the conductor agent.

        :param conductor_name: The name of the conductor agent.
        :return: A new instance of the Conversation with updated prompt arguments.
        """
        ...

    def update(
        self,
        chat: Optional[List[ChatMessage]] = None,
        system_prompt_argument: Optional[SystemPromptArgT] = None,
        user_prompt_argument: Optional[UserPromptArgT] = None,
        system_prompt: Optional[ChatMessage] = None,
        user_prompt: Optional[Union[ChatMessage, List[ChatMessage]]] = None,
        longterm_memory: Optional[Dict[str, LongtermMemory]] = None,
        metadata: Optional[Dict[str, str]] = None,
        active_agent: Optional[str] = None,
    ) -> "Conversation":
        """
        Returns a new instance of Conversation with updated fields, preserving immutability.

        :return: A new instance of the Conversation class with updated fields.
        """
        ...

    def append(
        self,
        message: Union[ChatMessage, List[ChatMessage], str],
        role: Optional[str] = "",
        name: Optional[str] = "",
    ) -> "Conversation":
        """
        Appends a new chat message and returns a new instance of Conversation.

        :param message: The message to be added.
        :param role: The role of the message sender.
        :param name: The name of the message sender.
        :return: A new instance of Conversation with the appended message.
        """
        ...

    def render_chat(self, message_type: MessageType = MessageType.CONVERSATION) -> List[ChatMessage]:
        """
        Returns the complete chat with the system prompt prepended and the user prompt appended.

        :return: A list representing the full chat.
        """
        ...

    def is_empty(self) -> bool:
        """
        Checks if the chat is empty.

        :return: True if the chat is empty, False otherwise.
        """
        ...

    def has_pending_tool_call(self) -> bool:
        """
        Checks if there is a pending tool call in the conversation.

        :return: True if there is a pending tool call, False otherwise.
        """
        ...

    def has_pending_tool_response(self) -> bool:
        """
        Checks if there is a pending tool response in the conversation.

        :return: True if there is a pending tool response, False otherwise.
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any], agents: List[Any]) -> "Conversation":
        """
        Creates a Conversation instance from a dictionary.

        :param data: The data dictionary.
        :param agents: A list of agent instances.
        :return: A new instance of Conversation.
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Conversation instance to a dictionary.

        :return: A dictionary representation of the Conversation.
        """
        ...


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

    def start_conversation(self, conversation: Conversation, user_query) -> None: ...


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

    def run(self, participant: ConversationParticipant, conversation: dict, user_query: str) -> dict:
        """Finish the conversation."""
        pass

    def finalize(self, response: dict) -> None:
        """Finalize a conversation by adding metadata or other final touches."""
        pass


class ConversationStore(Protocol):
    def persist(self, conversation: Conversation) -> None: ...

    def fetch(self, conversation_id: str) -> Conversation: ...

    def delete(self, conversation_id: str) -> None: ...

    def exists(self, conversation_id: str) -> bool: ...


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
