import copy
from dataclasses import dataclass, field, fields
from typing import Optional, TypeVar, Any

from enum import Enum


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


GenericUserPromptArg = TypeVar("GenericUserPromptArg", bound="PromptArgument")
GenericSystemPromptArg = TypeVar("GenericSystemPromptArg", bound="PromptArgument")

# TODO: Implement to string method for entities


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

    def __init__(
        self,
        system_prompt: ChatMessage,
        user_prompt: ChatMessage,
        system_prompt_argument: Optional[GenericSystemPromptArg] = None,
        user_prompt_argument: Optional[GenericUserPromptArg] = None,
        chat=None,
        metadata: Optional[dict[str, str]] = None,
    ):
        """
        Initializes a new immutable instance of the Conversation class.

        :param chat: A list of ChatMessage objects representing the conversation history.
        :param system_prompt: The initial system prompt message to set the context for the conversation.
        :param user_prompt: The final message from the user to conclude the conversation.
        :param metadata: A dictionary of metadata associated with the conversation (default is None).
        """
        if chat is None:
            self._chat = []
        else:
            self._chat = copy.deepcopy(chat)

        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._user_prompt_argument = (
            copy.deepcopy(user_prompt_argument) if user_prompt_argument else None
        )
        self._system_prompt_argument = (
            copy.deepcopy(system_prompt_argument) if system_prompt_argument else None
        )
        self._metadata = copy.deepcopy(metadata or {})

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

        :return: The final user message in the conversation.
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

    def update(
        self,
        chat: Optional[list[ChatMessage]] = None,
        system_prompt_argument: Optional[GenericSystemPromptArg] = None,
        user_prompt_argument: Optional[GenericUserPromptArg] = None,
        system_prompt: Optional[ChatMessage] = None,
        user_prompt: Optional[ChatMessage | list[ChatMessage]] = None,
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
        :param metadata: Updated metadata for the conversation (default is None).
        :return: A new instance of the Conversation class with updated fields.
        """
        return Conversation(
            chat=chat or self._chat,
            system_prompt=(system_prompt or self._system_prompt),
            user_prompt=user_prompt or self._user_prompt,
            system_prompt_argument=system_prompt_argument
            or self._system_prompt_argument,
            user_prompt_argument=user_prompt_argument or self._user_prompt_argument,
            metadata=metadata or self._metadata,
        )

    def append(
        self,
        message: str | ChatMessage | list[ChatMessage],
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
        elif isinstance(message, list) and all(
            isinstance(m, ChatMessage) for m in message
        ):
            new_message = message
        else:
            raise ValueError(
                "Invalid message type. Must be a string, ChatMessage or a list of ChatMessages"
            )

        new_chat = self.chat + new_message

        return self.update(chat=new_chat)

    def render_chat(self) -> list[ChatMessage]:
        """
        Returns the complete chat with the system prompt prepended and the user prompt appended.

        :return: A list representing the full chat, including the system and user prompts.
        """
        return [self.system_prompt] + self.chat + [self.user_prompt]

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


if __name__ == "__main__":
    # Step 1: Create ChatMessages using the Role enum
    system_message = ChatMessage(
        role=Role.SYSTEM, content="Welcome to the conversation."
    )
    user_message_1 = ChatMessage(
        role=Role.USER, content="Hello, I need help with my account."
    )
    assistant_message = ChatMessage(
        role=Role.ASSISTANT, content="Sure, I can help with that."
    )
    user_message_2 = ChatMessage(role=Role.USER, content="I forgot my password.")

    # Step 2: Create an initial Conversation
    conversation = Conversation(
        chat=[user_message_1, assistant_message],
        system_prompt=system_message,
        user_prompt=user_message_2,
    )

    # Step 3: Append a new message to the conversation
    tool_message = ChatMessage(
        role=Role.TOOL, content="Here is a link to reset your password."
    )
    updated_conversation = conversation.append(tool_message)

    # Step 4: Render the final conversation
    final_chat = updated_conversation.render_chat()

    # Step 5: Print the conversation (formatted for OpenAI)
    print("Conversation for OpenAI:")
    for msg in final_chat:
        print(msg.to_dict())

    # Step 6: Print the full internal conversation representation
    print("\nInternal Conversation Object:")
    print(updated_conversation)
