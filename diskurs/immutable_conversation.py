import copy
from dataclasses import fields, replace
from typing import TypeVar, Optional, Any

from diskurs import LongtermMemory
from diskurs.entities import ChatMessage, Role, MessageType
from diskurs.protocols import Conversation
from diskurs.registry import register_conversation

GenericPrompt = TypeVar("GenericPrompt", bound="Prompt")
GenericUserPromptArg = TypeVar("GenericUserPromptArg", bound="PromptArgument")
GenericSystemPromptArg = TypeVar("GenericSystemPromptArg", bound="PromptArgument")


@register_conversation("immutable_conversation")
class ImmutableConversation(Conversation):
    """
    Represents a conversation between a user and an agent, containing messages
    and prompts. This class is immutable; any modifications result in new instances.
    """

    def __init__(
        self,
        system_prompt: Optional[ChatMessage] = None,
        user_prompt: Optional[ChatMessage] = None,
        system_prompt_argument: Optional[GenericSystemPromptArg] = None,
        user_prompt_argument: Optional[GenericUserPromptArg] = None,
        chat=None,
        longterm_memory: Optional[dict[str, "LongtermMemory"]] = None,
        metadata: Optional[dict[str, str]] = None,
        active_agent: str = "",
        conversation_id="",
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
        self._active_agent = active_agent
        self._conversation_id = conversation_id

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    @conversation_id.setter
    def conversation_id(self, value):
        self._conversation_id = value

    @property
    def active_agent(self):
        return self._active_agent

    @active_agent.setter
    def active_agent(self, value):
        self._active_agent = value

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

    def update_agent_longterm_memory(
        self, agent_name: str, longterm_memory: "LongtermMemory"
    ) -> "ImmutableConversation":
        """
        Updates the longterm memory for a specific agent in the conversation.

        :param agent_name: The name of the agent whose longterm memory is to be updated.
        :param longterm_memory: The updated longterm memory for the agent.
        :return: A new instance of the Conversation class with the updated longterm memory.
        """
        updated_longterm_memory = copy.deepcopy(self._longterm_memory)
        updated_longterm_memory[agent_name] = longterm_memory

        return self.update(longterm_memory=updated_longterm_memory)

    def update_prompt_argument_with_longterm_memory(self, conductor_name: str) -> "ImmutableConversation":
        """
        Updates the prompt arguments with the longterm memory of the conductor agent, by copying the values from the
        longterm memory to the prompt arguments, where the properties match (both are dataclasses)

        :param conductor_name: The name of the conductor agent.
        :return: A new instance of the Conversation class with the updated prompt arguments.
        """
        longterm_memory = self.get_agent_longterm_memory(conductor_name)
        prompt_argument = self.user_prompt_argument

        updated_fields = {}

        for field in fields(prompt_argument):
            if hasattr(longterm_memory, field.name):
                longterm_value = getattr(longterm_memory, field.name)
                if longterm_value:
                    updated_fields[field.name] = longterm_value

        updated_user_prompt_argument = replace(prompt_argument, **updated_fields)

        return self.update(user_prompt_argument=updated_user_prompt_argument)

    def update(
        self,
        chat: Optional[list[ChatMessage]] = None,
        system_prompt_argument: Optional[GenericSystemPromptArg] = None,
        user_prompt_argument: Optional[GenericUserPromptArg] = None,
        system_prompt: Optional[ChatMessage] = None,
        user_prompt: Optional[ChatMessage | list[ChatMessage]] = None,
        longterm_memory: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, str]] = None,
        active_agent: Optional[str] = None,
    ) -> "ImmutableConversation":
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
        :param active_agent: The name of the agent that is currently mutating the conversation,
            used for de-serialization (default is None).
        :return: A new instance of the Conversation class with updated fields.
        """
        return ImmutableConversation(
            system_prompt=(system_prompt or self._system_prompt),
            user_prompt=user_prompt or self._user_prompt,
            system_prompt_argument=system_prompt_argument or self._system_prompt_argument,
            user_prompt_argument=user_prompt_argument or self._user_prompt_argument,
            chat=chat or self._chat,
            longterm_memory=longterm_memory or self._longterm_memory,
            metadata=metadata or self._metadata,
            active_agent=active_agent or self.active_agent,
        )

    def append(
        self,
        message: ChatMessage | list[ChatMessage],
        role: Optional[Role] = "",
        name: Optional[str] = "",
    ) -> "ImmutableConversation":
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

    def has_pending_tool_response(self) -> bool:
        user_prompt = self.user_prompt if isinstance(self.user_prompt, list) else [self.user_prompt]
        if not any(user_prompt):
            return False
        else:
            return any([msg.role == Role.TOOL for msg in user_prompt])

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

    @classmethod
    def from_dict(cls, data: dict[str, Any], agents: list):
        active_agent = next(agent for agent in agents if agent.name == data["active_agent"])

        system_prompt_argument_class = active_agent.prompt.system_prompt_argument
        system_prompt_argument = (
            system_prompt_argument_class.from_dict(data["system_prompt_argument"])
            if data["system_prompt_argument"]
            else None
        )
        system_prompt = ChatMessage.from_dict(data["system_prompt"]) if data["system_prompt"] else None

        user_prompt_argument = active_agent.prompt.user_prompt_argument
        user_prompt_argument = (
            user_prompt_argument.from_dict(data["user_prompt_argument"]) if data["user_prompt_argument"] else None
        )
        user_prompt = ChatMessage.from_dict(data["user_prompt"]) if data["user_prompt"] else None

        ltm_cls_map = {
            conductor_name: [agent for agent in agents if agent.name == conductor_name][0].prompt.longterm_memory
            for conductor_name in data["longterm_memory"].keys()
        }
        longterm_memory = {k: ltm_cls_map[k].from_dict(v) for k, v in data.get("longterm_memory", {}).items()}

        return cls(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            system_prompt_argument=system_prompt_argument,
            user_prompt_argument=user_prompt_argument,
            chat=[ChatMessage.from_dict(msg) for msg in data.get("chat", [])],
            longterm_memory=longterm_memory,
            metadata=data.get("metadata", {}),
            active_agent=data["active_agent"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": (self.system_prompt.to_dict() if self.system_prompt else None),
            "user_prompt": self.user_prompt.to_dict() if self.user_prompt else None,
            "system_prompt_argument": (self.system_prompt_argument.to_dict() if self.system_prompt_argument else None),
            "user_prompt_argument": (self.user_prompt_argument.to_dict() if self.user_prompt_argument else None),
            "chat": [msg.to_dict() for msg in self.chat],
            "longterm_memory": {k: v.to_dict() for k, v in self._longterm_memory.items()},
            "metadata": self.metadata,
            "active_agent": self.active_agent,
        }
