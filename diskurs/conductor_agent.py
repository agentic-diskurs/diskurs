import json
import logging
from dataclasses import asdict, fields
from typing import Any, List, Optional, Self

from diskurs.agent import BaseAgent, is_previous_agent_conductor
from diskurs.entities import ChatMessage, LongtermMemory, MessageType, PromptArgument, Role, RoutingRule
from diskurs.protocols import ConductorAgent as ConductorAgentProtocol
from diskurs.protocols import ConductorPrompt, Conversation, ConversationDispatcher, LLMClient
from diskurs.registry import register_agent

logger = logging.getLogger(__name__)


def validate_finalization(finalizer_name, prompt, supervisor):
    if not (finalizer_name is None and supervisor is None):
        assert (finalizer_name is None) != (
            supervisor is None
        ), "Either finalizer_name or supervisor must be set, but not both"
        delattr(prompt, "_finalize")


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
        rules: Optional[List[RoutingRule]] = None,
        fallback_to_llm: bool = True,
    ):

        validate_finalization(finalizer_name, prompt, supervisor=supervisor)

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

        self.rules = rules or []
        self.fallback_to_llm = fallback_to_llm

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

        rules = kwargs.get("rules", [])
        fallback_to_llm = kwargs.get("fallback_to_llm", True)

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
            rules=rules,
            fallback_to_llm=fallback_to_llm,
        )

    def evaluate_rules(self, conversation: Conversation) -> Optional[str]:
        if not self.rules:
            return None

        for rule in self.rules:
            self.logger.debug(f"Evaluating rule: {rule.name}")
            try:
                if rule.condition(conversation):
                    self.logger.info(f"Rule '{rule.name}' matched, routing to agent: {rule.target_agent}")
                    return rule.target_agent
            except Exception as e:
                self.logger.error(f"Error evaluating rule '{rule.name}': {e}")

        return None

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

    async def add_routing_message_to_chat(self, conversation, next_agent):
        updated_prompt = conversation.user_prompt_argument
        updated_prompt.next_agent = next_agent
        conversation = conversation.update(user_prompt_argument=updated_prompt)
        # Add JSON message to conversation to ensure a consistent chat history
        prompt_dict = asdict(updated_prompt)
        content = json.dumps(prompt_dict)
        return conversation.append(
            ChatMessage(role=Role.ASSISTANT, content=content, name=self.name, type=MessageType.CONDUCTOR)
        )

    async def invoke(self, conversation: Conversation, message_type=MessageType.CONDUCTOR) -> Conversation:
        self.logger.debug(f"Invoke called on conductor agent {self.name}")

        conversation = self.prepare_conversation(
            conversation,
            system_prompt_argument=self.prompt.create_system_prompt_argument(
                agent_descriptions=self.agent_descriptions
            ),
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
            message_type=MessageType.CONDUCTOR,
        )

        if next_agent := self.evaluate_rules(conversation):
            # Directly update the next_agent field in the prompt argument
            conversation = await self.add_routing_message_to_chat(conversation, next_agent)

        elif self.fallback_to_llm and self.llm_client:
            self.logger.debug("No matching rule found, falling back to LLM routing")

            for reasoning_step in range(self.max_trials):
                self.logger.debug(f"LLM routing step {reasoning_step + 1}")
                conversation = await self.generate_validated_response(conversation, message_type=MessageType.CONDUCTOR)

                if (
                    self.prompt.is_final(conversation.user_prompt_argument)
                    and not conversation.has_pending_tool_response()
                    and hasattr(conversation.user_prompt_argument, "next_agent")
                ):
                    self.logger.debug("Final response found in LLM routing")
                    break
        else:
            self.logger.error("No matching rule found and no LLM client available")

        return conversation

    async def can_finalize(self, conversation: Conversation) -> bool:
        if self.can_finalize_name:
            conversation = await self.dispatcher.request_response(self.can_finalize_name, conversation)
            return conversation.user_prompt_argument.can_finalize or self.prompt.can_finalize(
                conversation.get_agent_longterm_memory(self.name)
            )
        else:
            return self.prompt.can_finalize(conversation.get_agent_longterm_memory(self.name))

    async def finalize(self, conversation: Conversation) -> None:
        self.logger.debug(f"Finalize conversation on conductor agent {self.name}")

        if self.supervisor or self.finalizer_name:
            conversation = self.prepare_conversation(
                conversation,
                system_prompt_argument=self.prompt.create_system_prompt_argument(
                    agent_descriptions=self.agent_descriptions
                ),
                user_prompt_argument=self.prompt.create_user_prompt_argument(),
                message_type=MessageType.CONDUCTOR,
            )

        if self.supervisor:
            conversation = await self.add_routing_message_to_chat(conversation, self.supervisor)
            await self.dispatcher.publish(topic=self.supervisor, conversation=conversation)
        elif self.finalizer_name:
            conversation = await self.add_routing_message_to_chat(conversation, self.finalizer_name)
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
            # Get the next agent from user_prompt_argument and route if available
            if next_agent := conversation.user_prompt_argument.next_agent:
                await self.dispatcher.publish(topic=next_agent, conversation=conversation)
            else:
                self.logger.error("No next agent specified for routing. This should not happen after invoke().")
                conversation.final_result = self.fail(conversation)
