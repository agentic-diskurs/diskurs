from typing import Optional, Callable, Self

from diskurs.agent import BaseAgent
from diskurs.entities import (
    ToolDescription,
    ChatMessage,
    Role,
    MessageType,
    PromptArgument,
)
from diskurs.protocols import (
    LLMClient,
    ConversationDispatcher,
    MultistepPrompt,
    Conversation,
    ToolExecutor,
)
from diskurs.registry import register_agent


@register_agent("multistep")
class MultiStepAgent(BaseAgent[MultistepPrompt]):
    def __init__(
        self,
        name: str,
        prompt: MultistepPrompt,
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
        prompt: MultistepPrompt,
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
        """
        Registers one or more tools with the executor.

        This method allows the registration of a single tool or a list of tools
        that can be executed by the executor. Each tool is a callable that can
        be invoked with specific arguments.

        :param tools: A single callable or a list of callables representing the tools to be registered.
        """
        self.logger.info(f"Registering tools for agent {self.name}: {[tool.name for tool in tools]}")
        if callable(tools):
            tools = [tools]

        new_tools = [ToolDescription.from_function(fun) for fun in tools]

        if self.tools and set([tool.name for tool in new_tools]) & set(tool.name for tool in self.tools):
            self.logger.error(
                f"Tool names must be unique, found: {set([tool.name for tool in new_tools]) & set(tool.name for tool in self.tools)}"
            )
            raise ValueError("Tool names must be unique")
        else:
            self.tools = self.tools + new_tools

    def compute_tool_response(self, response: Conversation) -> list[ChatMessage]:
        """
        Executes the tool calls in the response and returns the tool responses.

        :param response: The conversation object containing the tool calls to execute.
        :return: One or more ChatMessage objects containing the tool responses.
        """
        self.logger.debug("Computing tool response for response")

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
        self,
        conversation: Conversation,
        message_type: MessageType = MessageType.CONVERSATION,
    ) -> Conversation:
        response = None

        for max_trials in range(self.max_trials):
            self.logger.debug(f"Generating validated response trial {max_trials + 1} for Agent {self.name}")

            response = self.llm_client.generate(conversation, getattr(self, "tools", None))

            if response.has_pending_tool_call():
                tool_responses = self.compute_tool_response(response)
                conversation = response.update(user_prompt=tool_responses)
            else:

                parsed_response = self.prompt.parse_user_prompt(
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

    def invoke(self, conversation: Conversation) -> Conversation:

        self.logger.debug(f"Invoke called on agent {self.name}")

        conversation = self.prepare_conversation(
            conversation,
            system_prompt_argument=self.prompt.create_system_prompt_argument(),
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
        )
        if self.init_prompt_arguments_with_longterm_memory:
            conversation = conversation.update_prompt_argument_with_longterm_memory(
                conductor_name=self.get_conductor_name()
            )
            conversation = conversation.update(
                user_prompt=self.prompt.render_user_template(
                    name=self.name, prompt_args=conversation.user_prompt_argument
                )
            )

        for reasoning_step in range(self.max_reasoning_steps):
            self.logger.debug(f"Reasoning step {reasoning_step + 1} for Agent {self.name}")
            conversation = self.generate_validated_response(conversation)

            if (
                self.prompt.is_final(conversation.user_prompt_argument)
                and not conversation.has_pending_tool_response()
            ):
                self.logger.debug(f"Final response found for Agent {self.name}")
                break

        return conversation.update()

    def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = self.invoke(conversation)
        self.dispatcher.publish(topic=self.get_conductor_name(), conversation=conversation)
