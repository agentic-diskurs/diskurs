import logging
from abc import abstractmethod, ABC
from dataclasses import is_dataclass
from typing import Optional, Self

from typing_extensions import TypeVar

from diskurs.entities import ChatMessage, PromptArgument, MessageType, Conversation, Role
from diskurs.protocols import ConversationDispatcher, LLMClient, Agent, ConversationParticipant

logger = logging.getLogger(__name__)

Prompt = TypeVar("Prompt")

# TODO: ensure maximum context length is 8192 tokens, if exceeds, truncate from left

# TODO: implement conditional rendering i.e. for each agent, only the information relevant to it is shown


class BaseAgent(ABC, Agent, ConversationParticipant):
    def __init__(
        self,
        name: str,
        prompt: Prompt,
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

    @property
    def topics(self) -> list[str]:
        return self._topics

    @topics.setter
    def topics(self, value):
        self._topics = value

    @abstractmethod
    def invoke(self, conversation: Conversation | str) -> Conversation:
        """
        Runs the agent on a conversation, performing reasoning steps until the user prompt is final,
        meaning all the conditions, as specified in the prompt's is_final function, are met.
        If the conversation is a string i.e. starting a new conversation, the agent will prepare
        the conversation by setting the user prompt argument's content to this string.

        :param conversation: The conversation object to run the agent on. If a string is provided, the agent will
            start a new conversation with the string as the user query's content.
        :return: the updated conversation object after the agent has finished reasoning. Contains
            the chat history, with all the system and user messages, as well as the final answer.
        """
        pass

    def register_dispatcher(self, dispatcher: ConversationDispatcher) -> None:
        self.dispatcher = dispatcher

    @abstractmethod
    def process_conversation(self, conversation: Conversation) -> None:
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

        system_prompt = self.prompt.render_system_template(self.name, prompt_args=system_prompt_argument)
        user_prompt = self.prompt.render_user_template(
            name=self.name, prompt_args=user_prompt_argument, message_type=message_type
        )

        return conversation.update(
            system_prompt_argument=system_prompt_argument,
            user_prompt_argument=user_prompt_argument,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def return_fail_validation_message(self, response):
        return response.append(
            ChatMessage(
                role=Role.USER, content=f"No valid answer found after {self.max_trials} trials.", name=self.name
            )
        )

    def generate_validated_response(
        self, conversation: Conversation, message_type: MessageType = MessageType.CONVERSATION
    ) -> Conversation:
        response = None

        for max_trials in range(self.max_trials):

            response = self.llm_client.generate(conversation, getattr(self, "tools", None))

            parsed_response = self.prompt.parse_user_prompt(response.last_message.content, message_type=message_type)

            if isinstance(parsed_response, PromptArgument):
                return response.update(
                    user_prompt_argument=parsed_response,
                    user_prompt=self.prompt.render_user_template(name=self.name, prompt_args=parsed_response),
                )
            elif isinstance(parsed_response, ChatMessage):
                conversation = response.update(user_prompt=parsed_response)
            else:
                raise ValueError(f"Failed to parse response from LLM model: {parsed_response}")

        return self.return_fail_validation_message(response or conversation)
