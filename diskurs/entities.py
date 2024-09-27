import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar, Any, Callable


class Role(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"
    USER = "user"

    def __str__(self):
        """
        Override the default string representation to return the enum's value.
        """
        return self.value


class MessageType(Enum):
    ROUTING = "routing"
    CONVERSATION = "conversation"

    def __str__(self):
        """
        Override the default string representation to return the enum's value.
        """
        return self.value


GenericPrompt = TypeVar("GenericPrompt", bound="Prompt")
GenericUserPromptArg = TypeVar("GenericUserPromptArg", bound="PromptArgument")
GenericSystemPromptArg = TypeVar("GenericSystemPromptArg", bound="PromptArgument")

GenericConductorLongtermMemory = TypeVar("GenericConductorLongtermMemory", bound="ConductorLongtermMemory")

# TODO: Implement to string method for entities i.e. __format__


@dataclass
class ToolCall:
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    tool_call_id: str
    function_name: str
    result: Any


@dataclass
class ChatMessage:
    role: Role
    content: Optional[str] = ""
    name: Optional[str] = ""
    tool_call_id: Optional[str] = ""
    tool_calls: Optional[list[ToolCall]] = None
    type: MessageType = MessageType.CONVERSATION

    def to_dict(self) -> dict:
        """
        Render the ChatMessage into a JSON serializable dictionary, suitable for submission to an API endpoint.
        Automatically converts Role enum into a string.

        :return: A dictionary with role as a string and other message attributes.
        """
        return {
            "role": str(self.role),
            "content": self.content,
            "name": self.name,
        }


class Conversation:
    """
    Represents a conversation between a user and an agent, containing messages
    and prompts. This class is immutable; any modifications result in new instances.
    """

    # TODO:  We need to handle ...prompt_arguments in a better way i.e. reset them reliably after each agent.
    def __init__(
        self,
        system_prompt: Optional[ChatMessage] = None,
        user_prompt: Optional[ChatMessage] = None,
        system_prompt_argument: Optional[GenericSystemPromptArg] = None,
        user_prompt_argument: Optional[GenericUserPromptArg] = None,
        chat=None,
        longterm_memory: Optional[dict[str, "LongtermMemory"]] = None,
        metadata: Optional[dict[str, str]] = None,
    ):
        """
        Initializes a new immutable instance of the Conversation class.

        :param system_prompt: The initial system prompt message to set the context for the conversation.
        :param user_prompt: The final message from the user to conclude the conversation.
        :param system_prompt_argument: system_prompt_arguments i.e. placeholders for the system prompt.
        :param user_prompt_argument: user_prompt_arguments i.e. placeholders for the user prompt.
        :param chat: A list of ChatMessage objects representing the conversation history.
        :param longterm_memory: A dictionary of long-term memories for agents. The key is the agent name
               and the value is the LongTermMemory object. Use this to store information between turns,
               especially for conductor agents.
        :param metadata: A dictionary of metadata associated with the conversation (default is None).
        """
        if chat is None:
            self._chat = []
        else:
            self._chat = copy.deepcopy(chat)

        self._system_prompt = copy.deepcopy(system_prompt) if system_prompt else None
        self._user_prompt = copy.deepcopy(user_prompt) if user_prompt else None
        self._user_prompt_argument = copy.deepcopy(user_prompt_argument) if user_prompt_argument else None
        self._system_prompt_argument = copy.deepcopy(system_prompt_argument) if system_prompt_argument else None
        self._longterm_memory = copy.deepcopy(longterm_memory) or {}
        self._metadata = copy.deepcopy(metadata) or {}

    @property
    def chat(self) -> list[ChatMessage]:
        """
        Provides a deep copy of the chat messages to ensure immutability.

        :return: A deep copy of the list of chat messages.
        """
        return copy.deepcopy(self._chat)

    @property
    def system_prompt(self) -> ChatMessage:
        """
        Retrieves the system prompt of the conversation.

        :return: The system prompt message.
        """
        return self._system_prompt

    @property
    def user_prompt(self) -> ChatMessage:
        """
        Retrieves the user prompt of the conversation.

        :return: The user prompt generated by applying the prompt arguments to the user prompt template.
        """
        return self._user_prompt

    @property
    def system_prompt_argument(self) -> GenericSystemPromptArg:
        return copy.deepcopy(self._system_prompt_argument)

    @property
    def user_prompt_argument(self) -> GenericUserPromptArg:
        return copy.deepcopy(self._user_prompt_argument)

    @property
    def metadata(self) -> dict[str, str]:
        """
        Provides a deep copy of the metadata dictionary to ensure immutability.

        :return: A deep copy of the metadata dictionary.
        """
        return copy.deepcopy(self._metadata)

    @property
    def last_message(self) -> ChatMessage:
        """
        Retrieves the last message in the conversation i.e. the most chat

        :return: The last message in the chat.
        """
        if not self.is_empty():
            return copy.deepcopy(self._chat[-1])
        else:
            raise ValueError("The chat is empty.")

    def get_agent_longterm_memory(self, agent_name: str) -> "LongtermMemory":
        """
        Provides a deep copy of the longterm memory dictionary to ensure immutability.

        :return: A deep copy of the longterm memory dictionary.
        """
        return copy.deepcopy(self._longterm_memory.get(agent_name))

    def update_agent_longterm_memory(self, agent_name: str, longterm_memory: "LongtermMemory") -> "Conversation":
        """
        Updates the longterm memory for a specific agent in the conversation.

        :param agent_name: The name of the agent whose longterm memory is to be updated.
        :param longterm_memory: The updated longterm memory for the agent.
        :return: A new instance of the Conversation class with the updated longterm memory.
        """
        updated_longterm_memory = copy.deepcopy(self._longterm_memory)
        updated_longterm_memory[agent_name] = longterm_memory

        return self.update(longterm_memory=updated_longterm_memory)

    def update(
        self,
        chat: Optional[list[ChatMessage]] = None,
        system_prompt_argument: Optional[GenericSystemPromptArg] = None,
        user_prompt_argument: Optional[GenericUserPromptArg] = None,
        system_prompt: Optional[ChatMessage] = None,
        user_prompt: Optional[ChatMessage | list[ChatMessage]] = None,
        longterm_memory: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> "Conversation":
        """
        Returns a new instance of Conversation with updated fields, preserving immutability.

        :param chat: An updated list of chat messages (default is None).
        :param system_prompt: An updated system prompt (default is None).
        :param user_prompt: An updated user prompt (default is None). If we compute multiple tool responses in one turn,
            we can pass a list of ChatMessages.
        :param system_prompt_argument: system_prompt_arguments i.e. placeholders for the system prompt (default is None).
        :param user_prompt_argument:d user_prompt_arguments i.e. placeholders for the user prompt (default is None).
        :param longterm_memory: Updated long-term memory for the conversation (default is None).
        :param metadata: Updated metadata for the conversation (default is None).
        :return: A new instance of the Conversation class with updated fields.
        """
        return Conversation(
            chat=chat or self._chat,
            system_prompt=(system_prompt or self._system_prompt),
            user_prompt=user_prompt or self._user_prompt,
            system_prompt_argument=system_prompt_argument or self._system_prompt_argument,
            user_prompt_argument=user_prompt_argument or self._user_prompt_argument,
            longterm_memory=longterm_memory or self._longterm_memory,
            metadata=metadata or self._metadata,
        )

    def append(
        self,
        message: ChatMessage | list[ChatMessage],
        role: Optional[Role] = "",
        name: Optional[str] = "",
    ) -> "Conversation":
        """
        Appends a new chat message and returns a new instance of Conversation.

        :param message: The ChatMessage object to be added to the conversation, alternatively a string can be provided.
        :param role: Only needed if message is str, the role (system, user, assistant)
        :param name: Only needed if message is str, name of the agent
        :return: A new instance of Conversation with the appended message.
        """
        if isinstance(message, str):
            new_message = [ChatMessage(content=message, role=role, name=name)]
        elif isinstance(message, ChatMessage):
            new_message = [message]
        elif isinstance(message, list) and all(isinstance(m, ChatMessage) for m in message):
            new_message = message
        else:
            raise ValueError("Invalid message type. Must be a string, ChatMessage or a list of ChatMessages")

        new_chat = self.chat + new_message

        return self.update(chat=new_chat)

    def render_chat(self, message_type: MessageType = MessageType.CONVERSATION) -> list[ChatMessage]:
        """
        Returns the complete chat with the system prompt prepended and the user prompt appended.

        :return: A list representing the full chat, including the system and user prompts.
        """
        if message_type == MessageType.CONVERSATION:
            chat = [message for message in self.chat if message.type == message_type]
        else:
            chat = self.chat

        return [self.system_prompt] + chat + [self.user_prompt]

    def is_empty(self) -> bool:
        """
        Checks if the chat is empty.

        :return: True if the chat is empty, False otherwise.
        """
        return len(self._chat) == 0

    def has_pending_tool_call(self):
        if not self.is_empty():
            last_message = self._chat[-1]
            return last_message.role == Role.ASSISTANT and last_message.tool_calls
        else:
            return False

    def __setattr__(self, key, value):
        """
        Overrides attribute setting to prevent modifications to existing attributes.

        :param key: The name of the attribute to set.
        :param value: The value to assign to the attribute.
        :raises AttributeError: If an attempt is made to modify an existing attribute.
        """
        if key in self.__dict__:
            raise AttributeError(f"{key} is immutable and cannot be changed")
        super().__setattr__(key, value)


@dataclass
class PromptArgument:
    pass


@dataclass
class LongtermMemory:
    user_query: str


@dataclass
class ToolDescription:
    name: str
    description: str
    arguments: dict[str, dict[str, str]]

    @classmethod
    def from_function(cls, function: Callable):
        """
        Create a ToolDescription instance from a decorated function.
        Assumes the function has been annotated with the @tool decorator.
        """
        return cls(
            name=function.name,
            description=function.description,
            arguments=function.args,
        )
