from typing import Optional, Self

from diskurs import Conversation, ToolExecutor, register_agent
from diskurs.agent import BaseAgent, FinalizerMixin
from diskurs.entities import MessageType, ToolDescription
from diskurs.protocols import (
    ConversationDispatcher,
    HeuristicPrompt,
    LLMClient,
)


@register_agent("heuristic")
class HeuristicAgent(BaseAgent[HeuristicPrompt]):
    def __init__(
        self,
        name: str,
        prompt: HeuristicPrompt,
        llm_client: Optional[LLMClient] = None,
        topics: Optional[list[str]] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        tools: Optional[list[ToolDescription]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        max_trials: int = 5,
        init_prompt_arguments_with_longterm_memory: bool = True,
        render_prompt: bool = True,
        final_properties: Optional[list[str]] = None,
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
        self.render_prompt = render_prompt
        self.final_properties = final_properties

    @classmethod
    def create(cls, name: str, prompt: HeuristicPrompt, **kwargs) -> Self:
        return cls(name=name, prompt=prompt, **kwargs)

    async def invoke(
        self, conversation: Conversation, message_type=MessageType.CONVERSATION, reset_prompt=True
    ) -> Conversation:
        self.logger.debug(f"Invoke called on agent {self.name}")

        conversation = self.prompt.initialize_prompt(
            agent_name=self.name,
            conversation=conversation,
            locked_fields=self.locked_fields,
            init_from_longterm_memory=self.init_prompt_arguments_with_longterm_memory,
            reset_prompt=reset_prompt,
            message_type=message_type,
            render_system_prompt=False,  # Heuristic agents don't use system prompts
        )

        conversation = await self.prompt.heuristic_sequence(
            conversation=conversation,
            call_tool=self.tool_executor.call_tool if self.tool_executor else None,
            llm_client=self.llm_client,
        )

        if self.render_prompt:
            conversation = conversation.append(
                name=self.name,
                message=self.prompt.render_user_template(
                    self.name,
                    prompt_args=conversation.prompt_argument,
                    message_type=MessageType.CONVERSATION,
                ),
            )

        # Update the global longterm memory with output fields from the prompt argument
        if conversation.prompt_argument:
            conversation = conversation.update_longterm_memory(conversation.prompt_argument)

        return conversation

    async def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke(conversation)

        for topic in self.topics:
            await self.dispatcher.publish(topic=topic, conversation=conversation)


@register_agent("heuristic_finalizer")
class HeuristicAgentFinalizer(HeuristicAgent, FinalizerMixin):
    def __init__(self, **kwargs):
        final_properties = kwargs.pop("final_properties")
        HeuristicAgent.__init__(self, **kwargs)
        FinalizerMixin.__init__(self, final_properties=final_properties)
