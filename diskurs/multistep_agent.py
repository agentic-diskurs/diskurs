import asyncio
from pathlib import Path
from typing import Callable, Optional, Self

from diskurs.agent import BaseAgent, FinalizerMixin
from diskurs.entities import MessageType, ToolDescription
from diskurs.protocols import (
    Conversation,
    ConversationDispatcher,
    ConversationResponder,
    LLMClient,
    MultistepPrompt,
    ToolExecutor,
)
from diskurs.registry import register_agent
from diskurs.utils import load_module_from_path


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
        super().__init__(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            topics=topics,
            dispatcher=dispatcher,
            max_trials=max_trials,
            tools=tools,
            tool_executor=tool_executor,
            init_prompt_arguments_with_longterm_memory=init_prompt_arguments_with_longterm_memory,
        )
        self.max_reasoning_steps = max_reasoning_steps

    @classmethod
    def create(
        cls,
        name: str,
        prompt: MultistepPrompt,
        llm_client: LLMClient,
        **kwargs,
    ) -> Self:
        return cls(name=name, prompt=prompt, llm_client=llm_client, **kwargs)

    async def invoke(
        self, conversation: Conversation, message_type=MessageType.CONVERSATION, reset_prompt=True
    ) -> Conversation:
        """
        Executes the reasoning steps to process the conversation.

        :param conversation: The conversation object to process
        :param message_type: The type of message to use
        :param reset_prompt: Whether to reset the prompt with fresh prompts and prompt arguments before invoking
        :return: The updated conversation after processing
        """
        self.logger.debug(f"Invoke called on agent {self.name}")

        conversation = self.prompt.initialize_prompt(
            agent_name=self.name,
            conversation=conversation,
            locked_fields=self.locked_fields,
            init_from_longterm_memory=self.init_prompt_arguments_with_longterm_memory,
            reset_prompt=reset_prompt,
        )

        for reasoning_step in range(self.max_reasoning_steps):
            self.logger.debug(f"Reasoning step {reasoning_step + 1} for Agent {self.name}")
            conversation = await self.generate_validated_response(conversation=conversation, message_type=message_type)

            if self.prompt.is_final(conversation.prompt_argument) and not conversation.has_pending_tool_response():
                self.logger.debug(f"Final response found for Agent {self.name}")
                break

        if conversation.prompt_argument:
            conversation = conversation.update_longterm_memory(conversation.prompt_argument)

        return conversation

    async def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke(conversation)

        for topic in self.topics:
            await self.dispatcher.publish(topic=topic, conversation=conversation)


@register_agent("multistep_finalizer")
class MultistepAgentFinalizer(MultiStepAgent, FinalizerMixin):
    def __init__(self, **kwargs):
        final_properties = kwargs.pop("final_properties")
        MultiStepAgent.__init__(self, **kwargs)
        FinalizerMixin.__init__(self, final_properties=final_properties)


@register_agent("multistep_predicate")
class MultistepAgentPredicate(MultiStepAgent, ConversationResponder):
    async def respond(self, conversation: Conversation) -> Conversation:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke(conversation, message_type=MessageType.CONDUCTOR)
        return conversation


@register_agent("parallel_multistep")
class ParallelMulstistepAgent(MultiStepAgent):
    def __init__(self, **kwargs):
        """
        Parallelize work, by first invoking the main conversation and then branching out the conversation to multiple
        :param invoke_on_final: whether we want to call the LLM again on the final result after applying the join function
        """
        self.invoke_on_final: bool = kwargs.pop("invoke_on_final")
        self._branch_conversation: Callable[[Conversation], list[Conversation]] = kwargs.pop("branch_conversation")
        self._join_conversations: Callable[[list[Conversation]], Conversation] = kwargs.pop("join_conversations")

        super().__init__(**kwargs)

    @classmethod
    def create(
        cls,
        name: str,
        prompt: MultistepPrompt,
        llm_client: LLMClient,
        **kwargs,
    ) -> Self:
        location = kwargs.pop("location", Path(__file__).parent)
        module_name: str = kwargs.get("code_filename", "parallelize.py")
        module = load_module_from_path(location / module_name)

        branch_conversation_name = kwargs.pop("branch_conversation_name", "branch_conversation")
        join_conversation_name = kwargs.pop("join_conversations_name", "join_conversations")

        branch_conversation: Callable = getattr(module, branch_conversation_name)
        join_conversation: Callable = getattr(module, join_conversation_name)

        return cls(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            branch_conversation=branch_conversation,
            join_conversations=join_conversation,
            **kwargs,
        )

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
            *(self.invoke(conv, message_type=message_type, reset_prompt=False) for conv in conversations)
        )
        result = await self.join_conversations(results)

        if self.invoke_on_final:
            return await self.invoke(result, message_type=message_type)
        else:
            return result

    async def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke_parallel(conversation)

        for topic in self.topics:
            await self.dispatcher.publish(topic=topic, conversation=conversation)
