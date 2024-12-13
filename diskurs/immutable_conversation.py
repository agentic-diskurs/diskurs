import copy
from dataclasses import fields, replace
from typing import TypeVar, Optional, Any

from diskurs import LongtermMemory
from diskurs.entities import ChatMessage, Role, MessageType, ResultHolder
from diskurs.protocols import Conversation
from diskurs.registry import register_conversation

GenericPrompt = TypeVar("GenericPrompt", bound="Prompt")
GenericUserPromptArg = TypeVar("GenericUserPromptArg", bound="PromptArgument")
GenericSystemPromptArg = TypeVar("GenericSystemPromptArg", bound="PromptArgument")


@register_conversation("immutable_conversation")
class ImmutableConversation(Conversation):

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
        final_result: Optional[dict[str, Any]] = None,
    ):
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
        self._final_result = final_result or ResultHolder()

    @property
    def final_result(self) -> dict[str, Any]:
        """
        Retrieves the final result of the conversation.

        The final result is a dictionary containing the data representing the outcome
        of the conversation. It is typically generated by the conductor agent when the
        conversation reaches a final state.

        :return: The final result of the conversation if available, otherwise None.
        """
        return self._final_result.result

    @final_result.setter
    def final_result(self, value: dict[str, Any]) -> None:
        """
        Takes the value, unpacks it and assigns its items to the final_result attribute.
        This ensures that the final_result maintains the same reference, thereby being the
        only attribute of the Conversation class that is explicitly not immutable.

        :param value: A dictionary containing the final result of the conversation.
        """
        self._final_result.result = value

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
        return copy.deepcopy(self._chat)

    @property
    def system_prompt(self) -> ChatMessage:
        return self._system_prompt

    @property
    def user_prompt(self) -> ChatMessage:
        return self._user_prompt

    @property
    def system_prompt_argument(self) -> GenericSystemPromptArg:
        return copy.deepcopy(self._system_prompt_argument)

    @property
    def user_prompt_argument(self) -> GenericUserPromptArg:
        return copy.deepcopy(self._user_prompt_argument)

    @property
    def metadata(self) -> dict[str, str]:
        return copy.deepcopy(self._metadata)

    @property
    def last_message(self) -> ChatMessage:
        if not self.is_empty():
            return copy.deepcopy(self._chat[-1])
        else:
            raise ValueError("The chat is empty.")

    def get_agent_longterm_memory(self, agent_name: str) -> "LongtermMemory":
        return copy.deepcopy(self._longterm_memory.get(agent_name))

    def update_agent_longterm_memory(
        self, agent_name: str, longterm_memory: "LongtermMemory"
    ) -> "ImmutableConversation":
        updated_longterm_memory = copy.deepcopy(self._longterm_memory)
        updated_longterm_memory[agent_name] = longterm_memory

        return self.update(longterm_memory=updated_longterm_memory)

    @staticmethod
    def update_prompt_argument(source_values, target_values):
        updated_fields = {}
        for field in fields(target_values):
            if hasattr(source_values, field.name):
                source_value = getattr(source_values, field.name)
                if source_value:
                    updated_fields[field.name] = source_value
        updated_user_prompt_argument = replace(target_values, **updated_fields)
        return updated_user_prompt_argument

    def update_prompt_argument_with_longterm_memory(self, conductor_name: str) -> "ImmutableConversation":
        longterm_memory = self.get_agent_longterm_memory(conductor_name)
        updated_prompt_argument = self.update_prompt_argument(
            source_values=longterm_memory, target_values=self.user_prompt_argument
        )

        return self.update(user_prompt_argument=updated_prompt_argument)

    def update_prompt_argument_with_previous_agent(
        self, previous_agent_prompt_argument: GenericUserPromptArg
    ) -> "ImmutableConversation":
        updated_prompt_argument = self.update_prompt_argument(
            source_values=previous_agent_prompt_argument, target_values=self.user_prompt_argument
        )

        return self.update(user_prompt_argument=updated_prompt_argument)

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
        conversation_id: Optional[str] = None,
    ) -> "ImmutableConversation":
        return ImmutableConversation(
            system_prompt=(system_prompt or self._system_prompt),
            user_prompt=user_prompt or self._user_prompt,
            system_prompt_argument=system_prompt_argument or self._system_prompt_argument,
            user_prompt_argument=user_prompt_argument or self._user_prompt_argument,
            chat=chat or self._chat,
            longterm_memory=longterm_memory or self._longterm_memory,
            metadata=metadata or self._metadata,
            active_agent=active_agent or self.active_agent,
            final_result=self._final_result,
            conversation_id=conversation_id or self._conversation_id,
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
        if message_type == MessageType.CONVERSATION:
            chat = [message for message in self.chat if message.type == message_type]
        else:
            chat = self.chat

        return [self.system_prompt] + chat + [self.user_prompt]

    def is_empty(self) -> bool:
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
