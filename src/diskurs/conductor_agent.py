import json
from dataclasses import fields
from typing import Optional, Self

from agent import BaseAgent
from entities import Conversation
from interfaces import LLMClient, ConversationDispatcher, ConductorPrompt

from registry import register_agent


@register_agent("conductor")
class ConductorAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        prompt: ConductorPrompt,
        llm_client: LLMClient,
        topics: list[str],
        agent_descriptions: dict[str, str],
        finalizer_name: str,
        dispatcher: Optional[ConversationDispatcher] = None,
        max_reasoning_steps: int = 5,
        max_trials: int = 5,
    ):
        super().__init__(name, prompt, llm_client, topics, dispatcher, max_trials)
        self.agent_descriptions = agent_descriptions
        self.max_reasoning_steps = max_reasoning_steps
        self._topics = []
        self.finalizer_name = finalizer_name

    @classmethod
    def create(
        cls,
        name: str,
        prompt: ConductorPrompt,
        llm_client: LLMClient,
        **kwargs,
    ) -> Self:
        agent_descriptions = kwargs.get("agent_descriptions", {})
        finalizer_name = kwargs.get("finalizer_name", "")
        dispatcher = kwargs.get("dispatcher", None)
        max_reasoning_steps = kwargs.get("max_reasoning_steps", 5)
        max_trials = kwargs.get("max_trials", 5)
        topics = kwargs.get("topics", [])

        return cls(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            dispatcher=dispatcher,
            agent_descriptions=agent_descriptions,
            finalizer_name=finalizer_name,
            max_reasoning_steps=max_reasoning_steps,
            max_trials=max_trials,
            topics=topics,
        )

    def update_longterm_memory(self, conversation: Conversation, overwrite: bool = False) -> Conversation:
        longterm_memory = conversation.get_agent_longterm_memory(self.name)
        longterm_memory = longterm_memory or self.prompt.init_longterm_memory()
        # TODO: allow for custom mapping of prompt args to longterm memory

        last_agents_user_prompt_arguments = conversation.user_prompt_argument
        if longterm_memory:
            common_fields = {field.name for field in fields(longterm_memory)}.intersection(
                {field.name for field in fields(last_agents_user_prompt_arguments)}
            )

            for field in common_fields:
                if overwrite or not getattr(longterm_memory, field):
                    setattr(longterm_memory, field, getattr(last_agents_user_prompt_arguments, field))

        return conversation.update_agent_longterm_memory(agent_name=self.name, longterm_memory=longterm_memory)

    def invoke(self, conversation: Conversation | str) -> Conversation:
        conversation = self.prepare_conversation(
            conversation,
            system_prompt_argument=self.prompt.create_system_prompt_argument(
                agent_descriptions=self.agent_descriptions
            ),
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
        )
        conversation = self.update_longterm_memory(conversation)

        return self.generate_validated_response(conversation)

    def process_conversation(self, conversation: Conversation | str) -> None:

        if not isinstance(conversation, str) and conversation.get_agent_longterm_memory(self.name):
            if self.prompt.can_finalize(conversation.get_agent_longterm_memory(self.name)):
                self.dispatcher.publish(topic=self.finalizer_name, conversation=conversation)
        else:
            conversation = self.invoke(conversation)
            next_agent = json.loads(conversation.last_message.content).get("next_agent")
            self.dispatcher.publish(topic=next_agent, conversation=conversation)
