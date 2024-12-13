import logging
from abc import abstractmethod, ABC
from typing import Optional, Generic

from typing_extensions import TypeVar

from diskurs.entities import ChatMessage, PromptArgument, MessageType, Role
from diskurs.logger_setup import get_logger
from diskurs.protocols import (
    ConversationDispatcher,
    LLMClient,
    Agent,
    ConversationParticipant,
    Conversation,
    Prompt,
)

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
    async def invoke(self, conversation: Conversation | str) -> Conversation:
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

    def prepare_conversation(
        self,
        conversation: Conversation,
        system_prompt_argument: PromptArgument,
        user_prompt_argument: PromptArgument,
        message_type: MessageType = MessageType.CONVERSATION,
    ) -> Conversation:
        """
        Ensures the conversation is in a valid state by creating a new set of prompts
        and prompt_variables for system and user, as well creating a fresh copy of the conversation.

        :param conversation: A conversation object, possible passed from another agent
            or a string to start a new conversation.
        :param system_prompt_argument: The system prompt argument to use for the system prompt.
        :param user_prompt_argument: The user prompt argument to use for the user prompt.
        :param message_type: The type of message to render the user prompt as.
        :return: A deep copy of the conversation, in a valid state for this agent
        """
        self.logger.debug(f"Preparing conversation for agent {self.name}")
        system_prompt = self.prompt.render_system_template(self.name, prompt_args=system_prompt_argument)
        user_prompt = self.prompt.render_user_template(
            name=self.name, prompt_args=user_prompt_argument, message_type=message_type
        )

        return conversation.update(
            system_prompt_argument=system_prompt_argument,
            user_prompt_argument=user_prompt_argument,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            active_agent=self.name,
        )

    def return_fail_validation_message(self, response):
        return response.append(
            ChatMessage(
                role=Role.USER,
                content=f"No valid answer found after {self.max_trials} trials.",
                name=self.name,
            )
        )

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

            response = await self.llm_client.generate(conversation, getattr(self, "tools", None))

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
