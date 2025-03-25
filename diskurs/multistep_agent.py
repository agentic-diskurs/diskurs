import asyncio
from typing import Callable, Optional, Self

from diskurs.agent import BaseAgent, get_last_conductor_name, is_previous_agent_conductor
from diskurs.entities import ChatMessage, MessageType, Role, ToolDescription
from diskurs.protocols import (
    Conversation,
    ConversationDispatcher,
    ConversationFinalizer,
    ConversationResponder,
    LLMClient,
    MultistepPrompt,
    ToolExecutor,
)
from diskurs.registry import register_agent
from diskurs.tools import generate_tool_descriptions
from diskurs.utils import get_fields_as_dict


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
        init_prompt_arguments_with_previous_agent: bool = True,
    ):
        super().__init__(name, prompt, llm_client, topics, dispatcher, max_trials)
        self.tool_executor = tool_executor
        self.tools = tools or []
        self.max_reasoning_steps = max_reasoning_steps
        self.init_prompt_arguments_with_longterm_memory = init_prompt_arguments_with_longterm_memory
        self.init_prompt_arguments_with_previous_agent = init_prompt_arguments_with_previous_agent

    @classmethod
    def create(
        cls,
        name: str,
        prompt: MultistepPrompt,
        llm_client: LLMClient,
        **kwargs,
    ) -> Self:

        return cls(name=name, prompt=prompt, llm_client=llm_client, **kwargs)

    def register_tools(self, tools: list[Callable] | Callable) -> None:
        self.tools = generate_tool_descriptions(self.tools, tools, self.logger, self.name)

    async def compute_tool_response(self, response: Conversation) -> list[ChatMessage]:
        """
        Executes the tool calls in the response and returns the tool responses.

        :param response: The conversation object containing the tool calls to execute.
        :return: One or more ChatMessage objects containing the tool responses.
        """
        self.logger.debug("Computing tool response for response")

        tool_responses = []
        for tool in response.last_message.tool_calls:
            tool_call_result = await self.tool_executor.execute_tool(tool, response.metadata)
            tool_responses.append(
                ChatMessage(
                    role=Role.TOOL,
                    tool_call_id=tool_call_result.tool_call_id,
                    content=tool_call_result.result,
                    name=self.name,
                )
            )
        return tool_responses

    async def handle_tool_call(self, conversation, response):
        tool_responses = await self.compute_tool_response(response)
        conversation = response.update(user_prompt=tool_responses)
        return conversation

    async def prepare_invoke(self, conversation):
        self.logger.debug(f"Invoke called on agent {self.name}")
        previous_user_prompt_augment = conversation.user_prompt_argument
        conversation = self.prepare_conversation(
            conversation,
            system_prompt_argument=self.prompt.create_system_prompt_argument(),
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
        )
        if self.init_prompt_arguments_with_longterm_memory:
            conversation = conversation.update_prompt_argument_with_longterm_memory(
                conductor_name=get_last_conductor_name(conversation.chat)
            )
            conversation = conversation.update(
                user_prompt=self.prompt.render_user_template(
                    name=self.name, prompt_args=conversation.user_prompt_argument
                )
            )
        if not is_previous_agent_conductor(conversation) and self.init_prompt_arguments_with_previous_agent:
            conversation = conversation.update_prompt_argument_with_previous_agent(previous_user_prompt_augment)
        return conversation

    async def invoke(self, conversation: Conversation, message_type=MessageType.CONVERSATION) -> Conversation:

        conversation = await self.prepare_invoke(conversation)

        for reasoning_step in range(self.max_reasoning_steps):
            self.logger.debug(f"Reasoning step {reasoning_step + 1} for Agent {self.name}")
            conversation = await self.generate_validated_response(conversation=conversation, message_type=message_type)

            if (
                self.prompt.is_final(conversation.user_prompt_argument)
                and not conversation.has_pending_tool_response()
            ):
                self.logger.debug(f"Final response found for Agent {self.name}")
                break

        return conversation.update()

    async def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke(conversation)

        for topic in self.topics:
            await self.dispatcher.publish(topic=topic, conversation=conversation)


@register_agent("multistep_finalizer")
class MultistepAgentFinalizer(MultiStepAgent, ConversationFinalizer):
    def __init__(self, **kwargs):
        final_properties = kwargs.pop("final_properties")

        super().__init__(**kwargs)

        self.final_properties = final_properties

    async def finalize_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke(conversation)

        await conversation.maybe_persist()

        conversation.final_result = get_fields_as_dict(conversation.user_prompt_argument, self.final_properties)


@register_agent("multistep_predicate")
class MultistepAgentPredicate(MultiStepAgent, ConversationResponder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def respond(self, conversation: Conversation) -> Conversation:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke(conversation, message_type=MessageType.CONDUCTOR)
        return conversation


@register_agent("parallel_multistep")
class ParallelMulstistepAgent(MultiStepAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.invoke_on_final: bool = kwargs.pop("invoke_on_final")
        self._branch_conversation: Callable[[Conversation], list[Conversation]] = kwargs.pop("branch_conversations")
        self._join_conversations: Callable[[list[Conversation]], Conversation] = kwargs.pop("join_conversations")

    async def branch_conversations(self, conversation: Conversation) -> list[Conversation]:
        return self._branch_conversation(conversation)

    async def join_conversations(self, conversations: list[Conversation]) -> Conversation:
        return self._join_conversations(conversations)

    async def invoke_parallel(self, conversation: Conversation, message_type=MessageType.CONVERSATION) -> Conversation:
        """
        Parallelize work, by first invoking the main conversation and then branching out the conversation to multiple
        invoke calls. The first invoke is meant to identify parallelizable work, we then call branch_conversations to
        split the work into multiple conversations and then invoke each conversation in parallel.

        :param conversation: The conversation object to process.
        :param message_type: The message type to use when generating responses.
        :return: The conversation object with the final result.
        """
        conversation = await self.invoke(conversation, message_type=message_type)
        conversations = await self.branch_conversations(conversation)

        results = await asyncio.gather(
            *(self.invoke(conversation, message_type=message_type) for conversation in conversations)
        )
        result = await self.join_conversations(results)

        if self.invoke_on_final:
            return await self.invoke(conversation, message_type=message_type)
        else:
            return result
