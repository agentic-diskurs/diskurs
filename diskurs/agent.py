import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional

from typing_extensions import TypeVar

from diskurs.entities import ChatMessage, MessageType, PromptArgument, Role
from diskurs.logger_setup import get_logger
from diskurs.protocols import Agent, Conversation, ConversationDispatcher, ConversationParticipant, LLMClient, Prompt

logger = logging.getLogger(__name__)

PromptType = TypeVar("PromptType", bound=Prompt)

# TODO: implement conditional rendering i.e. for each agent, only the information relevant to it is shown


def is_previous_agent_conductor(conversation):
    if conversation.is_empty():
        return False
    else:
        return conversation.last_message.type == MessageType.CONDUCTOR


def get_last_conductor_name(chat: list[ChatMessage]) -> Optional[str]:
    for message in reversed(chat):
        if message.type == MessageType.CONDUCTOR:
            return message.name
    return None


def has_conductor_been_called(conversation):
    return any(message.type == MessageType.CONDUCTOR for message in conversation.chat)


class BaseAgent(ABC, Agent, ConversationParticipant, Generic[PromptType]):
    def __init__(
        self,
        name: str,
        prompt: PromptType,
        llm_client: LLMClient,
        topics: Optional[list[str]] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        max_trials: int = 5,
    ):
        self.dispatcher = dispatcher
        self.prompt = prompt
        self.name = name
        self._topics = topics or []
        self.max_trials = max_trials
        self.llm_client = llm_client
        self.logger = get_logger(f"diskurs.agent.{self.name}")

        self.logger.info(f"Initializing agent {self.name}")

    @property
    def topics(self) -> list[str]:
        return self._topics

    @topics.setter
    def topics(self, value):
        self._topics = value

    @abstractmethod
    async def invoke(
        self, conversation: Conversation | str, message_type=MessageType.CONVERSATION, reset_prompt=True
    ) -> Conversation:
        pass

    def register_dispatcher(self, dispatcher: ConversationDispatcher) -> None:
        self.dispatcher = dispatcher

        self.logger.debug(f"Registered dispatcher {dispatcher} for agent {self.name}")

    @abstractmethod
    async def process_conversation(self, conversation: Conversation) -> None:
        """
        Receives a conversation from the dispatcher, i.e. message bus, processes it and finally publishes
        a deep copy of the resulting conversation back to the dispatcher.

        :param conversation: The conversation object to process.
        """
        pass

    def return_fail_validation_message(self, response):
        return response.append(
            ChatMessage(
                role=Role.USER,
                content=f"No valid answer found after {self.max_trials} trials.",
                name=self.name,
            )
        )

    async def handle_tool_call(self, conversation, response):
        """
        This method handles a tool call by executing the tool and updating the conversation
        with the tool response. It is not implemented in the base class and should be implemented
        in subclasses that require it

        :param conversation: The conversation object to add the result to.
        :param response: The response object containing the tool call to handle.
        :return: The updated conversation object with the tool response.
        """
        return conversation

    async def generate_validated_response(
        self,
        conversation: Conversation,
        message_type: MessageType = MessageType.CONVERSATION,
    ) -> Conversation:
        """
        Generates a validated response for the given conversation.

        This method attempts to generate a valid response for the conversation by
        interacting with the LLM client and validating the response. It performs
        multiple trials if necessary, and handles tool calls and corrective messages.

        :param conversation: The conversation object to generate a response for.
        :param message_type: The type of message to render the user prompt as, defaults to MessageType.CONVERSATION.
        :return: The updated conversation object with the validated response.
        """
        response = None

        for max_trials in range(self.max_trials):
            self.logger.debug(f"Generating validated response trial {max_trials + 1} for Agent {self.name}")

            response = await self.llm_client.generate(
                conversation=conversation, tools=getattr(self, "tools", None), message_type=message_type
            )

            if response.has_pending_tool_call():
                conversation = await self.handle_tool_call(conversation, response)
            else:

                parsed_response = self.prompt.parse_user_prompt(
                    self.name,
                    llm_response=response.last_message.content,
                    old_user_prompt_argument=response.user_prompt_argument,
                    message_type=message_type,
                )

                if isinstance(parsed_response, PromptArgument):
                    self.logger.debug(f"Valid response found for Agent {self.name}")
                    return response.update(
                        user_prompt_argument=parsed_response,
                        user_prompt=self.prompt.render_user_template(name=self.name, prompt_args=parsed_response),
                    )
                elif isinstance(parsed_response, ChatMessage):
                    self.logger.debug(f"Invalid response, created corrective message for Agent {self.name}")
                    conversation = response.update(user_prompt=parsed_response)
                else:
                    self.logger.error(f"Failed to parse response from LLM model: {parsed_response}")
                    raise ValueError(f"Failed to parse response from LLM model: {parsed_response}")

        return self.return_fail_validation_message(response or conversation)
