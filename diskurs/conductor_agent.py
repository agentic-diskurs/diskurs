import json
import logging
from dataclasses import fields
from typing import Optional, Self, Any

from diskurs.agent import BaseAgent
from diskurs.entities import (
    MessageType,
)
from diskurs.protocols import LLMClient, ConversationDispatcher, ConductorPromptProtocol, Conversation
from diskurs.registry import register_agent

logger = logging.getLogger(__name__)


@register_agent("conductor")
class ConductorAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        prompt: ConductorPromptProtocol,
        llm_client: LLMClient,
        topics: list[str],
        agent_descriptions: dict[str, str],
        finalizer_name: str,
        dispatcher: Optional[ConversationDispatcher] = None,
        max_trials: int = 5,
        max_dispatches: int = 50,
    ):
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
        finalizer_name = kwargs.get("finalizer_name", "")
        dispatcher = kwargs.get("dispatcher", None)
        max_trials = kwargs.get("max_trials", 5)
        max_dispatches = kwargs.get("max_dispatches", 50)
        topics = kwargs.get("topics", [])

        # TODO: think if the json description of the agents should be used directly from the prompt, or if we
        #  want to use the option here too (but we'll always need json output from the conductor anyways)

        return cls(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            dispatcher=dispatcher,
            agent_descriptions=agent_descriptions,
            finalizer_name=finalizer_name,
            max_trials=max_trials,
            max_dispatches=max_dispatches,
            topics=topics,
        )

    def update_longterm_memory(self, conversation: Conversation, overwrite: bool = False) -> Conversation:
        longterm_memory = conversation.get_agent_longterm_memory(self.name) or self.prompt.init_longterm_memory()

        # TODO: do not update memory if last agent was a conductor agent -> important: when called by start conversation it will be a conductor longterm memory
        if last_agents_user_prompt_arguments := conversation.user_prompt_argument:
            common_fields = {field.name for field in fields(longterm_memory)}.intersection(
                {field.name for field in fields(last_agents_user_prompt_arguments)}
            )
            for field in common_fields:
                if overwrite or not getattr(longterm_memory, field):
                    setattr(
                        longterm_memory,
                        field,
                        getattr(last_agents_user_prompt_arguments, field),
                    )

        return conversation.update_agent_longterm_memory(agent_name=self.name, longterm_memory=longterm_memory)

    @staticmethod
    def is_conversation_start(conversation: Conversation) -> bool:
        return (not conversation.user_prompt) and (not conversation.system_prompt) and conversation.is_empty()

    def invoke(self, conversation: Conversation) -> Conversation:

        conversation = self.prepare_conversation(
            conversation,
            system_prompt_argument=self.prompt.create_system_prompt_argument(
                agent_descriptions=self.agent_descriptions
            ),
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
        )

        return self.generate_validated_response(conversation, message_type=MessageType.ROUTING)

    def finalize(self, conversation: Conversation) -> dict[str, Any]:
        return self.prompt.finalize(conversation.get_agent_longterm_memory(self.name))

    def fail(self, conversation: Conversation) -> dict[str, Any]:
        return self.prompt.fail(conversation.get_agent_longterm_memory(self.name))

    def process_conversation(self, conversation: Conversation) -> None:
        logger.info(f"Agent: {self.name}")
        self.n_dispatches += 1
        conversation = self.update_longterm_memory(conversation)

        if conversation.get_agent_longterm_memory(self.name) and self.prompt.can_finalize(
            conversation.get_agent_longterm_memory(self.name)
        ):
            formated_response = self.finalize(conversation=conversation)
            self.n_dispatches = 0
            self.dispatcher.finalize(response=formated_response)

        elif self.max_dispatches <= self.n_dispatches:
            formatted_response = self.fail(conversation=conversation)
            self.n_dispatches = 0
            self.dispatcher.finalize(response=formatted_response)

        else:
            conversation = self.invoke(conversation)
            next_agent = json.loads(conversation.last_message.content).get("next_agent")
            self.dispatcher.publish(topic=next_agent, conversation=conversation)

    def start_conversation(self, conversation: Conversation, user_query: str) -> None:

        conversation = conversation.update_agent_longterm_memory(
            agent_name=self.name,
            longterm_memory=self.prompt.init_longterm_memory(user_query=user_query),
        )

        self.process_conversation(conversation)
