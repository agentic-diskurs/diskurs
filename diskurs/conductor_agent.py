import json
import logging
from dataclasses import fields
from typing import Optional, Self, Any

from diskurs.agent import BaseAgent, is_previous_agent_conductor
from diskurs.entities import (
    MessageType,
    LongtermMemory,
    PromptArgument,
)
from diskurs.protocols import (
    LLMClient,
    ConversationDispatcher,
    ConductorPrompt,
    Conversation,
    ConductorAgent as ConductorAgentProtocol,
)
from diskurs.registry import register_agent

logger = logging.getLogger(__name__)


def validate_finalization(can_finalize_name, finalizer_name, prompt, supervisor):
    if not (finalizer_name is None and supervisor is None):
        assert (finalizer_name is None) != (
            supervisor is None
        ), "Either finalizer_name or supervisor must be set, but not both"
        delattr(prompt, "_finalize")

    if can_finalize_name:
        delattr(prompt, "_can_finalize")


@register_agent("conductor")
class ConductorAgent(BaseAgent[ConductorPrompt], ConductorAgentProtocol):
    def __init__(
        self,
        name: str,
        prompt: ConductorPrompt,
        llm_client: LLMClient,
        topics: list[str],
        agent_descriptions: dict[str, str],
        finalizer_name: Optional[str] = None,
        supervisor: Optional[str] = None,
        can_finalize_name: Optional[str] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        max_trials: int = 5,
        max_dispatches: int = 50,
    ):

        validate_finalization(can_finalize_name, finalizer_name, prompt, supervisor=supervisor)

        super().__init__(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            topics=topics,
            dispatcher=dispatcher,
            max_trials=max_trials,
        )
        self.agent_descriptions = agent_descriptions
        self.finalizer_name = finalizer_name
        self.supervisor = supervisor
        self.can_finalize_name = can_finalize_name
        self.max_dispatches = max_dispatches
        self.n_dispatches = 0

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs,
    ) -> Self:
        prompt = kwargs.get("prompt")
        llm_client = kwargs.get("llm_client")
        agent_descriptions = kwargs.get("agent_descriptions")
        finalizer_name = kwargs.get("finalizer_name", None)
        can_finalize_name = kwargs.get("can_finalize_name", None)
        supervisor = kwargs.get("supervisor", None)
        dispatcher = kwargs.get("dispatcher", None)
        max_trials = kwargs.get("max_trials", 5)
        max_dispatches = kwargs.get("max_dispatches", 50)
        topics = kwargs.get("topics", [])

        return cls(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            dispatcher=dispatcher,
            agent_descriptions=agent_descriptions,
            finalizer_name=finalizer_name,
            can_finalize_name=can_finalize_name,
            supervisor=supervisor,
            max_trials=max_trials,
            max_dispatches=max_dispatches,
            topics=topics,
        )

    @staticmethod
    def update_longterm_memory(
        source: LongtermMemory | PromptArgument,
        target: LongtermMemory,
        overwrite: bool,
    ) -> LongtermMemory:
        common_fields = {field.name for field in fields(target)}.intersection({field.name for field in fields(source)})
        for field in common_fields:
            if overwrite or not getattr(target, field):
                setattr(
                    target,
                    field,
                    getattr(source, field),
                )
        return target

    def create_or_update_longterm_memory(self, conversation: Conversation, overwrite: bool = False) -> Conversation:
        longterm_memory = conversation.get_agent_longterm_memory(self.name) or self.prompt.init_longterm_memory()

        source = (
            conversation.get_agent_longterm_memory(conversation.last_message.name)
            if is_previous_agent_conductor(conversation)
            else conversation.user_prompt_argument
        )

        if source:
            longterm_memory = self.update_longterm_memory(source, longterm_memory, overwrite)
        else:
            self.logger.warning(
                f"No suitable user prompt argument nor long-term memory found in conversation {conversation}"
            )

        return conversation.update_agent_longterm_memory(agent_name=self.name, longterm_memory=longterm_memory)

    async def invoke(self, conversation: Conversation, message_type=MessageType.CONVERSATION) -> Conversation:
        self.logger.debug(f"Invoke called on conductor agent {self.name}")

        conversation = self.prepare_conversation(
            conversation,
            system_prompt_argument=self.prompt.create_system_prompt_argument(
                agent_descriptions=self.agent_descriptions
            ),
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
            message_type=MessageType.CONDUCTOR,
        )

        # TODO: try to unify with multistep agent
        for reasoning_step in range(self.max_trials):
            self.logger.debug(f"Reasoning step {reasoning_step + 1} for Agent {self.name}")
            conversation = await self.generate_validated_response(conversation, message_type=MessageType.CONDUCTOR)

            if (
                self.prompt.is_final(conversation.user_prompt_argument)
                and not conversation.has_pending_tool_response()
            ):
                self.logger.debug(f"Final response found for Agent {self.name}")
                break

        return conversation.update()

    async def can_finalize(self, conversation: Conversation) -> bool:
        if self.can_finalize_name:
            conversation = await self.dispatcher.request_response(self.can_finalize_name, conversation)
            return conversation.user_prompt_argument.can_finalize
        else:
            return self.prompt.can_finalize(conversation.get_agent_longterm_memory(self.name))

    async def finalize(self, conversation: Conversation) -> None:
        self.logger.debug(f"Finalize conversation on conductor agent {self.name}")

        if self.supervisor:
            await self.dispatcher.publish(topic=self.supervisor, conversation=conversation)
        elif self.finalizer_name:
            await self.dispatcher.publish_final(topic=self.finalizer_name, conversation=conversation)
        else:
            conversation.final_result = self.prompt.finalize(conversation.get_agent_longterm_memory(self.name))

    def fail(self, conversation: Conversation) -> dict[str, Any]:
        self.logger.debug(f"End conversation with fail on conductor agent {self.name}")
        return self.prompt.fail(conversation.get_agent_longterm_memory(self.name))

    async def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on conductor agent: {self.name}")
        self.n_dispatches += 1
        conversation = self.create_or_update_longterm_memory(conversation)

        if conversation.get_agent_longterm_memory(self.name) and await self.can_finalize(conversation):
            await self.finalize(conversation=conversation)
            self.n_dispatches = 0

        elif self.max_dispatches <= self.n_dispatches:
            formatted_response = self.fail(conversation=conversation)
            self.n_dispatches = 0
            conversation.final_result = formatted_response
        else:
            conversation = await self.invoke(conversation)
            next_agent = json.loads(conversation.last_message.content).get("next_agent")
            await self.dispatcher.publish(topic=next_agent, conversation=conversation)
