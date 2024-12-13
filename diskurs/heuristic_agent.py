from typing import Optional, Self

from diskurs import register_agent, Conversation, ToolExecutor, Agent, PromptArgument
from diskurs.agent import is_previous_agent_conductor, get_last_conductor_name, has_conductor_been_called
from diskurs.entities import MessageType, ToolDescription
from diskurs.logger_setup import get_logger
from diskurs.protocols import (
    ConversationParticipant,
    HeuristicPrompt,
    ConversationDispatcher,
    ConversationFinalizer,
)
from diskurs.utils import get_fields_as_dict


@register_agent("heuristic")
class HeuristicAgent(Agent, ConversationParticipant):
    def __init__(
        self,
        name: str,
        prompt: HeuristicPrompt,
        topics: Optional[list[str]] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        tools: Optional[list[ToolDescription]] = None,
        tool_executor: Optional[ToolExecutor] = None,
        init_prompt_arguments_with_longterm_memory: bool = True,
        init_prompt_arguments_with_previous_agent: bool = True,
        render_prompt: bool = True,
        final_properties: Optional[list[str]] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.topics = topics or []
        self.dispatcher = dispatcher
        self.tool_executor = tool_executor
        self.tools = tools or []
        self.init_prompt_arguments_with_longterm_memory = init_prompt_arguments_with_longterm_memory
        self.init_prompt_arguments_with_previous_agent = init_prompt_arguments_with_previous_agent
        self.render_prompt = render_prompt
        self.final_properties = final_properties
        self.logger = get_logger(f"diskurs.agent.{self.name}")

    @classmethod
    def create(cls, name: str, prompt: HeuristicPrompt, **kwargs) -> Self:
        return cls(name=name, prompt=prompt, **kwargs)

    def register_dispatcher(self, dispatcher: ConversationDispatcher) -> None:
        self.dispatcher = dispatcher

        self.logger.debug(f"Registered dispatcher {dispatcher} for agent {self.name}")

    def prepare_conversation(self, conversation: Conversation, user_prompt_argument: PromptArgument) -> Conversation:
        self.logger.debug(f"Preparing conversation for agent {self.name}")
        return conversation.update(user_prompt_argument=user_prompt_argument, active_agent=self.name)

    async def invoke(self, conversation: Conversation) -> Conversation:
        self.logger.debug(f"Invoke called on agent {self.name}")

        previous_user_prompt_augment = conversation.user_prompt_argument

        conversation = self.prepare_conversation(
            conversation=conversation,
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
        )
        if has_conductor_been_called(conversation) and self.init_prompt_arguments_with_longterm_memory:
            conversation = conversation.update_prompt_argument_with_longterm_memory(
                conductor_name=get_last_conductor_name(conversation.chat)
            )
        if self.tool_executor:
            call_tool = self.tool_executor.call_tool
        else:
            call_tool = None

        if not is_previous_agent_conductor(conversation) and self.init_prompt_arguments_with_previous_agent:
            conversation = conversation.update_prompt_argument_with_previous_agent(previous_user_prompt_augment)

        conversation = await self.prompt.heuristic_sequence(conversation=conversation, call_tool=call_tool)

        if self.render_prompt:
            conversation = conversation.append(
                name=self.name,
                message=self.prompt.render_user_template(
                    self.name,
                    prompt_args=conversation.user_prompt_argument,
                    message_type=MessageType.CONVERSATION,
                ),
            )

        return conversation

    async def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke(conversation)

        for topic in self.topics:
            await self.dispatcher.publish(topic=topic, conversation=conversation)


@register_agent("heuristic_finalizer")
class HeuristicAgentFinalizer(HeuristicAgent, ConversationFinalizer):
    def __init__(self, **kwargs):
        final_properties = kwargs.pop("final_properties")

        super().__init__(**kwargs)

        self.final_properties = final_properties

    async def finalize_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = await self.invoke(conversation)
        conversation.final_result = get_fields_as_dict(conversation.user_prompt_argument, self.final_properties)
