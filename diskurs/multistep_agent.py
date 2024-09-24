from typing import Optional, Callable, Self

from agent import BaseAgent
from entities import ToolDescription, Conversation, ChatMessage, Role
from protocols import LLMClient, ConversationDispatcher, MultistepPromptProtocol
from registry import register_agent
from tools import ToolExecutor


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
    ):
        super().__init__(name, prompt, llm_client, topics, dispatcher, max_trials)
        self.tool_executor = tool_executor
        self.tools = tools or []
        self.max_reasoning_steps = max_reasoning_steps

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
            tool_call_result = self.tool_executor.execute_tool(tool)
            tool_responses.append(
                ChatMessage(
                    role=Role.TOOL,
                    tool_call_id=tool_call_result.tool_call_id,
                    content=tool_call_result.result,
                )
            )
        return tool_responses

    def perform_reasoning(self, conversation: Conversation) -> Conversation:
        """
        Performs a single reasoning step on the conversation, generating a response from the LLM model,
        updating the conversation with the response, generating the next user prompt and returning
        the updated conversation.
        :param conversation: The conversation object to perform reasoning on.
        :return: Updated conversation object after reasoning step.
        """
        response = self.llm_client.generate(conversation, self.tools)

        if response.has_pending_tool_call():
            tool_responses = self.compute_tool_response(response)
            return response.update(user_prompt=tool_responses)
        else:
            return self.generate_validated_response(response)

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

        for reasoning_step in range(self.max_reasoning_steps):
            conversation = self.perform_reasoning(conversation)

            if self.prompt.is_final(conversation.user_prompt_argument):
                self.max_reasoning_steps = 0
                break

        return conversation.update()

    def process_conversation(self, conversation: Conversation) -> None:
        """
        Receives a conversation from the dispatcher, i.e. message bus, processes it and finally publishes
        a deep copy of the resulting conversation back to the dispatcher.

        :param conversation: The conversation object to process.
        """
        conversation = self.invoke(conversation)
        self.dispatcher.publish(topic=self.get_conductor_name(), conversation=conversation)
