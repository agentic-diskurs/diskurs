from typing import Optional, Self

from diskurs import register_agent, Conversation, ToolExecutor, Agent, PromptArgument
from diskurs.entities import MessageType
from diskurs.logger_setup import get_logger
from diskurs.protocols import (
    ConversationParticipant,
    HeuristicPrompt,
    ConversationDispatcher,
)


@register_agent("heuristic")
class HeuristicAgent(Agent, ConversationParticipant):
    def __init__(
        self,
        name: str,
        prompt: HeuristicPrompt,
        topics: Optional[list[str]] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        tool_executor: Optional[ToolExecutor] = None,
        init_prompt_arguments_with_longterm_memory: bool = True,
        render_prompt: bool = True,
    ):
        self.name = name
        self.prompt = prompt
        self.topics = topics or []
        self.dispatcher = dispatcher
        self.tool_executor = tool_executor
        self.init_prompt_arguments_with_longterm_memory = init_prompt_arguments_with_longterm_memory
        self.render_prompt = render_prompt
        self.logger = get_logger(f"diskurs.agent.{self.name}")

    @classmethod
    def create(cls, name: str, prompt: HeuristicPrompt, **kwargs) -> Self:
        tool_executor = kwargs.get("tool_executor", None)
        topics = kwargs.get("topics", [])
        dispatcher = kwargs.get("dispatcher", None)
        render_prompt = kwargs.get("render_prompt", True)

        return cls(
            name=name,
            prompt=prompt,
            topics=topics,
            dispatcher=dispatcher,
            tool_executor=tool_executor,
            render_prompt=render_prompt,
        )

    def get_conductor_name(self) -> str:
        # TODO: somewhat hacky, but should work for now
        return self.topics[0]

    def register_dispatcher(self, dispatcher: ConversationDispatcher) -> None:
        self.dispatcher = dispatcher

        self.logger.debug(f"Registered dispatcher {dispatcher} for agent {self.name}")

    def prepare_conversation(self, conversation: Conversation, user_prompt_argument: PromptArgument) -> Conversation:
        self.logger.debug(f"Preparing conversation for agent {self.name}")
        return conversation.update(user_prompt_argument=user_prompt_argument, active_agent=self.name)

    def invoke(self, conversation: Conversation | str) -> Conversation:
        self.logger.debug(f"Invoke called on agent {self.name}")

        conversation = self.prepare_conversation(
            conversation=conversation,
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
        )

        if self.init_prompt_arguments_with_longterm_memory:
            conversation = conversation.update_prompt_argument_with_longterm_memory(
                conductor_name=self.get_conductor_name()
            )
        if self.tool_executor:
            call_tool = self.tool_executor.call_tool
        else:
            call_tool = None

        conversation = self.prompt.heuristic_sequence(conversation, call_tool=call_tool)

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

    def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on agent: {self.name}")
        conversation = self.invoke(conversation)
        self.dispatcher.publish(topic=self.get_conductor_name(), conversation=conversation)
