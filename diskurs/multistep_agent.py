import logging
from typing import Optional, Callable, Self

from diskurs.agent import BaseAgent
from diskurs.entities import ToolDescription, Conversation, ChatMessage, Role, MessageType, PromptArgument
from diskurs.protocols import LLMClient, ConversationDispatcher, MultistepPromptProtocol
from diskurs.registry import register_agent
from diskurs.tools import ToolExecutor

logger = logging.getLogger(__name__)


@register_agent("multistep")
class MultiStepAgent(BaseAgent):
    # TODO: fix mess with TypeVars
    def __init__(
        self,
        name: str,
        prompt: MultistepPromptProtocol,
        llm_client: LLMClient,
        topics: Optional[list[str]] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        tool_executor: Optional[ToolExecutor] = None,
        tools: Optional[list[ToolDescription]] = None,
        max_reasoning_steps: int = 5,
        max_trials: int = 5,
        init_prompt_arguments_with_longterm_memory: bool = True,
    ):
        super().__init__(name, prompt, llm_client, topics, dispatcher, max_trials)
        self.tool_executor = tool_executor
        self.tools = tools or []
        self.max_reasoning_steps = max_reasoning_steps
        self.init_prompt_arguments_with_longterm_memory = init_prompt_arguments_with_longterm_memory

    @classmethod
    def create(
        cls,
        name: str,
        prompt: MultistepPromptProtocol,
        llm_client: LLMClient,
        **kwargs,
    ) -> Self:
        dispatcher = kwargs.get("dispatcher", None)
        tool_executor = kwargs.get("tool_executor", None)
        tools = kwargs.get("tools", None)
        max_reasoning_steps = kwargs.get("max_reasoning_steps", 5)
        max_trials = kwargs.get("max_trials", 5)
        topics = kwargs.get("topics", [])
        init_prompt_arguments_with_longterm_memory = kwargs.get("init_prompt_arguments_with_longterm_memory", True)

        return cls(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            dispatcher=dispatcher,
            tool_executor=tool_executor,
            tools=tools,
            max_reasoning_steps=max_reasoning_steps,
            max_trials=max_trials,
            topics=topics,
            init_prompt_arguments_with_longterm_memory=init_prompt_arguments_with_longterm_memory,
        )

    def get_conductor_name(self) -> str:
        # TODO: somewhat hacky, but should work for now
        return self.topics[0]

    def register_tools(self, tools: list[Callable] | Callable) -> None:
        if callable(tools):
            tools = [tools]

        new_tools = [ToolDescription.from_function(fun) for fun in tools]

        if self.tools and set([tool.name for tool in new_tools]) & set(tool.name for tool in self.tools):
            raise ValueError("Tool names must be unique")
        else:
            self.tools = self.tools + new_tools

    def compute_tool_response(self, response: Conversation) -> list[ChatMessage]:
        """
        Executes the tool calls in the response and returns the tool responses.

        :param response: The conversation object containing the tool calls to execute.
        :return: One or more ChatMessage objects containing the tool responses.
        """
        tool_responses = []
        for tool in response.last_message.tool_calls:
            tool_call_result = self.tool_executor.execute_tool(tool, response.metadata)
            tool_responses.append(
                ChatMessage(
                    role=Role.TOOL,
                    tool_call_id=tool_call_result.tool_call_id,
                    content=tool_call_result.result,
                    name=self.name,
                )
            )
        return tool_responses

    def generate_validated_response(
        self, conversation: Conversation, message_type: MessageType = MessageType.CONVERSATION
    ) -> Conversation:
        response = None

        for max_trials in range(self.max_trials):

            response = self.llm_client.generate(conversation, getattr(self, "tools", None))

            if response.has_pending_tool_call():
                tool_responses = self.compute_tool_response(response)
                conversation = response.update(user_prompt=tool_responses)
            else:

                parsed_response = self.prompt.parse_user_prompt(
                    response.last_message.content, message_type=message_type
                )

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

    def invoke(self, conversation: Conversation) -> Conversation:
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
        conversation = self.prepare_conversation(
            conversation,
            system_prompt_argument=self.prompt.create_system_prompt_argument(),
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
        )
        if self.init_prompt_arguments_with_longterm_memory:
            conversation.update_prompt_argument_with_longterm_memory(conductor_name=self.get_conductor_name())

        for reasoning_step in range(self.max_reasoning_steps):
            conversation = self.generate_validated_response(conversation)

            if (
                self.prompt.is_final(conversation.user_prompt_argument)
                and not conversation.has_pending_tool_response()
            ):
                break

        return conversation.update()

    def process_conversation(self, conversation: Conversation) -> None:
        """
        Receives a conversation from the dispatcher, i.e. message bus, processes it and finally publishes
        a deep copy of the resulting conversation back to the dispatcher.

        :param conversation: The conversation object to process.
        """
        logger.info(f"Agent: {self.name}")
        conversation = self.invoke(conversation)
        self.dispatcher.publish(topic=self.get_conductor_name(), conversation=conversation)
