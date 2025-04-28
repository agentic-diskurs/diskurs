import logging
from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, List, Any

from typing_extensions import TypeVar

from diskurs.entities import ChatMessage, MessageType, PromptArgument, Role, ToolDescription
from diskurs.logger_setup import get_logger
from diskurs.protocols import (
    Agent,
    Conversation,
    ConversationDispatcher,
    ConversationFinalizer,
    ConversationParticipant,
    LLMClient,
    Prompt,
    ToolExecutor,
)
from diskurs.tools import generate_tool_descriptions
from diskurs.utils import get_fields_as_dict

logger = logging.getLogger(__name__)

PromptType = TypeVar("PromptType", bound=Prompt)

# TODO: implement conditional rendering i.e. for each agent, only the information relevant to it is shown


class BaseAgent(ABC, Agent, ConversationParticipant, Generic[PromptType]):
    def __init__(
        self,
        name: str,
        prompt: PromptType,
        llm_client: LLMClient,
        topics: Optional[list[str]] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        max_trials: int = 5,
        tools: Optional[list[ToolDescription]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        locked_fields: Optional[dict[str, Any]] = None,
        init_prompt_arguments_with_longterm_memory: bool = True,
    ):
        self.dispatcher = dispatcher
        self.prompt = prompt
        self.name = name
        self._topics = topics or []
        self.max_trials = max_trials
        self.llm_client = llm_client
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.locked_fields = locked_fields
        self.init_prompt_arguments_with_longterm_memory = init_prompt_arguments_with_longterm_memory
        self.logger = get_logger(f"diskurs.agent.{self.name}")

        self.logger.info(f"Initializing agent {self.name}")

    @property
    def topics(self) -> list[str]:
        return self._topics

    @topics.setter
    def topics(self, value):
        self._topics = value

    def register_tools(self, tools: list[Callable] | Callable) -> None:
        """
        Register tools with the agent.

        :param tools: A list of callables or a single callable to register as tools
        """
        self.tools = generate_tool_descriptions(self.tools, tools, self.logger, self.name)

    def register_dispatcher(self, dispatcher: ConversationDispatcher) -> None:
        """
        Register a dispatcher with the agent.

        :param dispatcher: The dispatcher to register
        """
        self.dispatcher = dispatcher
        self.logger.debug(f"Registered dispatcher {dispatcher} for agent {self.name}")

    @abstractmethod
    async def invoke(
        self, conversation: Conversation | str, message_type=MessageType.CONVERSATION, reset_prompt=True
    ) -> Conversation:
        pass

    @abstractmethod
    async def process_conversation(self, conversation: Conversation) -> None:
        """
        Receives a conversation from the dispatcher, i.e. message bus, processes it and finally publishes
        a deep copy of the resulting conversation back to the dispatcher.

        :param conversation: The conversation object to process.
        """
        pass

    def return_fail_validation_message(self, response):
        """
        Returns a fail validation message when max trials are exhausted.

        :param response: The conversation to add the fail message to
        :return: The updated conversation with the fail message
        """
        return response.append(
            ChatMessage(
                role=Role.USER,
                content=f"No valid answer found after {self.max_trials} trials.",
                name=self.name,
            )
        )

    async def compute_tool_response(self, response: Conversation) -> list[ChatMessage]:
        """
        Executes the tool calls in the response and returns the tool responses.

        :param response: The conversation object containing the tool calls to execute.
        :return: One or more ChatMessage objects containing the tool responses.
        """
        if not self.tool_executor:
            self.logger.warning("Tool executor not set, cannot compute tool response")
            return []

        self.logger.debug("Computing tool response for response")

        tool_responses = []
        for tool in response.last_message.tool_calls:
            try:
                tool_call_result = await self.tool_executor.execute_tool(tool, response.metadata)
                tool_responses.append(
                    ChatMessage(
                        role=Role.TOOL,
                        tool_call_id=tool_call_result.tool_call_id,
                        content=tool_call_result.result,
                        name=self.name,
                    )
                )
            except Exception as e:
                # Convert the exception to a PromptValidationError with an instructive message
                error_message = f"Tool '{tool.function_name}' execution failed: {str(e)}"
                self.logger.error(error_message, exc_info=True)

                # Return a formatted error message as a tool response
                # This maintains the expected message sequence (tool call -> tool response)
                tool_responses.append(
                    ChatMessage(
                        role=Role.TOOL,
                        tool_call_id=tool.tool_call_id,
                        content=f"ERROR: {error_message}. Please correct your input and try again.",
                        name=self.name,
                    )
                )
        return tool_responses

    async def handle_tool_call(self, conversation, response):
        """
        This method handles a tool call by executing the tool and updating the conversation
        with the tool response.

        :param conversation: The conversation object to add the result to.
        :param response: The response object containing the tool call to handle.
        :return: The updated conversation object with the tool response.
        """
        tool_responses = await self.compute_tool_response(response)
        conversation = response.update(user_prompt=tool_responses)
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
                    llm_response=response.last_message.content or "",
                    old_prompt_argument=response.prompt_argument or self.prompt.create_prompt_argument(),
                    message_type=message_type,
                )

                if isinstance(parsed_response, PromptArgument):
                    self.logger.debug(f"Valid response found for Agent {self.name}")
                    return response.update(
                        prompt_argument=parsed_response,
                        user_prompt=self.prompt.render_user_template(name=self.name, prompt_args=parsed_response),
                    )
                elif isinstance(parsed_response, ChatMessage):
                    self.logger.debug(f"Invalid response, created corrective message for Agent {self.name}")
                    conversation = response.update(user_prompt=parsed_response)
                else:
                    self.logger.error(f"Failed to parse response from LLM model: {parsed_response}")
                    raise ValueError(f"Failed to parse response from LLM model: {parsed_response}")

        return self.return_fail_validation_message(response or conversation)


class FinalizerMixin(ConversationFinalizer):
    """
    A mixin for agent classes that need finalizer capabilities.
    This provides common functionality for finalizing conversations.
    """

    def __init__(self, final_properties: List[str], **kwargs):
        """
        Initialize the finalizer mixin.

        :param final_properties: The properties to extract from the prompt argument for the final result
        """
        self.final_properties = final_properties
        # Note: Do not call super().__init__ here, this is a mixin

    async def finalize_conversation(self, conversation: Conversation) -> None:
        """
        Finalize the conversation by invoking the agent and extracting final properties.

        :param conversation: The conversation to finalize
        """
        self.logger.info(f"Finalize conversation on agent: {self.name}")
        conversation = await self.invoke(conversation)

        await conversation.maybe_persist()

        # Extract final properties from the prompt argument
        conversation.final_result = get_fields_as_dict(conversation.prompt_argument, self.final_properties)

    async def process_conversation(self, conversation: Conversation) -> None:
        """
        Process the conversation by finalizing it.

        :param conversation: The conversation to process
        """
        self.logger.info(f"Finalizing conversation on agent: {self.name}")
        return await self.finalize_conversation(conversation)
