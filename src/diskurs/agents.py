import logging
from abc import ABC
from typing import Optional, Callable, Self

from entities import Conversation, ChatMessage, Role, PromptArgument, ToolDescription
from interfaces import ConversationDispatcher, LLMClient, Agent, ConversationParticipant, ConversationFinalizer
from prompt import Prompt
from registry import register_agent
from test_tools import tool_executor
from tools import ToolExecutor

logger = logging.getLogger(__name__)


@register_agent("multistep")
class MultiStepAgent(Agent, ConversationParticipant):

    def __init__(
        self,
        name: str,
        prompt: Prompt,
        llm_client: LLMClient,
        topics: Optional[list[str]] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        tool_executor: Optional[ToolExecutor] = None,
        tools: Optional[list[ToolDescription]] = None,
        max_reasoning_steps: int = 5,
    ):
        self.name = name
        self.prompt = prompt
        self.llm_client = llm_client
        self.topics = topics
        self.dispatcher = dispatcher
        self.tool_executor = tool_executor
        self.tools = tools or []
        self.max_reasoning_steps = max_reasoning_steps
        self._topics = []

    @classmethod
    def create(
        cls,
        name: str,
        prompt: Prompt,
        llm_client: LLMClient,
        **kwargs,
    ) -> Self:
        dispatcher = kwargs.get("dispatcher", None)
        tool_executor = kwargs.get("tool_executor", None)
        tools = kwargs.get("tools", None)
        max_reasoning_steps = kwargs.get("max_reasoning_steps", 5)

        return cls(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            dispatcher=dispatcher,
            tool_executor=tool_executor,
            tools=tools,
            max_reasoning_steps=max_reasoning_steps,
        )

    def register_tools(self, tools: list[Callable] | Callable) -> None:
        if callable(tools):
            tools = [tools]

        new_tools = [ToolDescription.from_function(fun) for fun in tools]

        if self.tools and set([tool.name for tool in new_tools]) & set(tool.name for tool in self.tools):
            raise ValueError("Tool names must be unique")
        else:
            self.tools = self.tools + new_tools

    def prepare_conversation(self, conversation: Conversation | str) -> Conversation:
        """
        Ensures the conversation is in a valid state by creating a new set of prompts
        and prompt_variables for system and user, as well creating a fresh copy of the conversation.

        :param conversation: A conversation object, possible passed from another agent
            or a string to start a new conversation.
        :return: A deep copy of the conversation, in a valid state for this agent
        """
        system_prompt_argument = self.prompt.system_prompt_argument()
        user_prompt_argument = self.prompt.user_prompt_argument()

        if isinstance(conversation, str):
            user_prompt_argument.content = conversation

        system_prompt = self.prompt.render_system_template(self.name, system_prompt_argument)
        user_prompt = self.prompt.render_user_template(name=self.name, prompt_args=user_prompt_argument)

        if isinstance(conversation, str):
            return Conversation(
                system_prompt_argument=system_prompt_argument,
                user_prompt_argument=user_prompt_argument,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            return conversation.update(
                chat=conversation.chat,
                system_prompt_argument=system_prompt_argument,
                user_prompt_argument=user_prompt_argument,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

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

    def create_next_user_prompt(self, response: Conversation) -> Conversation:
        parsed_response = self.prompt.parse_prompt(response.last_message.content)
        if isinstance(parsed_response, PromptArgument):
            return response.update(
                user_prompt_argument=parsed_response,
                user_prompt=self.prompt.render_user_template(name=self.name, prompt_args=parsed_response),
            )
        elif isinstance(parsed_response, ChatMessage):
            return response.update(user_prompt=parsed_response)
        else:
            raise ValueError(f"Failed to parse response from LLM model: {parsed_response}")

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
            return self.create_next_user_prompt(response)

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
        conversation = self.prepare_conversation(conversation)
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
        self.dispatcher.publish(topic=self.name, conversation=conversation)


@register_agent("conductor")
class ConductorAgent(Agent, ConversationParticipant, ConversationFinalizer):
    def __init__(
        self,
        name: str,
        prompt: Prompt,
        llm_client: LLMClient,
        agent_descriptions: dict[str, str],
        dispatcher: Optional[ConversationDispatcher] = None,
        max_reasoning_steps: int = 5,
        conductor_memory_fields: Optional[list[str]] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.llm_client = llm_client
        self.agent_descriptions = agent_descriptions
        self.dispatcher = dispatcher
        self.max_reasoning_steps = max_reasoning_steps
        self.conductor_memory_fields = conductor_memory_fields or {}
        self._topics = []

    @classmethod
    def create(
        cls,
        name: str,
        prompt: Prompt,
        llm_client: LLMClient,
        **kwargs,
    ) -> Self:
        agent_descriptions = kwargs.get("agent_descriptions", {})
        dispatcher = kwargs.get("dispatcher", None)
        max_reasoning_steps = kwargs.get("max_reasoning_steps", 5)
        conductor_memory_fields = kwargs.get("conductor_memory_fields", [])

        return cls(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            agent_descriptions=agent_descriptions,
            dispatcher=dispatcher,
            max_reasoning_steps=max_reasoning_steps,
            conductor_memory_fields=conductor_memory_fields,
        )

    def set_conductor_memory_fields(self, conductor_memory_fields: list[str]) -> None:
        self.conductor_memory_fields = conductor_memory_fields

    def invoke(self, conversation: Conversation | str) -> Conversation:
        conversation = self.prepare_conversation(conversation)
        for reasoning_step in range(self.max_reasoning_steps):
            conversation = self.perform_reasoning(conversation)

            if self.prompt.is_final(conversation.user_prompt_argument):
                self.max_reasoning_steps = 0
                break

        return conversation.update()

    def prepare_conversation(self, conversation: Conversation | str) -> Conversation:
        """
        Ensures the conversation is in a valid state by creating a new set of prompts
        and prompt_variables for system and user, as well creating a fresh copy of the conversation.

        :param conversation: A conversation object, possible passed from another agent
            or a string to start a new conversation.
        :return: A deep copy of the conversation, in a valid state for this agent
        """
        system_prompt_argument = self.prompt.system_prompt_argument(agent_descriptions=self.agent_descriptions)
        user_prompt_argument = self.prompt.user_prompt_argument()

        if isinstance(conversation, str):
            user_prompt_argument.content = conversation
            metadata = {"conductor_memory": {field_name: None for field_name in self.conductor_memory_fields}}

        system_prompt = self.prompt.render_system_template(name=self.name, prompt_args=system_prompt_argument)
        user_prompt = self.prompt.render_user_template(name=self.name, prompt_args=user_prompt_argument)

        if isinstance(conversation, str):
            return Conversation(
                system_prompt_argument=system_prompt_argument,
                user_prompt_argument=user_prompt_argument,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            return conversation.update(
                chat=conversation.chat,
                system_prompt_argument=system_prompt_argument,
                user_prompt_argument=user_prompt_argument,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

    def can_finalize(self, conversation: Conversation) -> bool:
        pass

    def finalize(self, conversation: Conversation) -> dict:
        pass

    def process_conversation(self, conversation: Conversation) -> None:
        conversation = self.invoke(conversation)

        if self.can_finalize(conversation):
            final_response = self.finalize(conversation)
            self.dispatcher.finalize(final_response)
        else:
            self.dispatcher.publish(topic=self.name, conversation=conversation)
