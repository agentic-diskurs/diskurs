import json
import logging
from dataclasses import asdict
from typing import Any, List, Optional, Self

from diskurs.agent import BaseAgent
from diskurs.entities import ChatMessage, MessageType, Role, RoutingRule
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
        locked_fields: Optional[dict[str, Any]],
        finalizer_name: Optional[str] = None,
        supervisor: Optional[str] = None,
        can_finalize_name: Optional[str] = None,
        dispatcher: Optional[ConversationDispatcher] = None,
        max_trials: int = 5,
        max_dispatches: int = 50,
        rules: Optional[List[RoutingRule]] = None,
        fallback_to_llm: bool = True,
        tools: Optional[list[Any]] = None,
        tool_executor: Optional[Any] = None,
        init_prompt_arguments_with_longterm_memory: bool = True,
    ):
        validate_finalization(finalizer_name, prompt, supervisor=supervisor)

        super().__init__(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            topics=topics,
            dispatcher=dispatcher,
            max_trials=max_trials,
            tools=tools,
            tool_executor=tool_executor,
            locked_fields=locked_fields,
            init_prompt_arguments_with_longterm_memory=init_prompt_arguments_with_longterm_memory,
        )
        self.agent_descriptions = locked_fields
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
        prompt = kwargs["prompt"]
        llm_client = kwargs["llm_client"]
        locked_fields = kwargs.get("locked_fields", {})
        finalizer_name = kwargs.get("finalizer_name", None)
        can_finalize_name = kwargs.get("can_finalize_name", None)
        supervisor = kwargs.get("supervisor", None)
        dispatcher = kwargs.get("dispatcher", None)
        max_trials = kwargs.get("max_trials", 5)
        max_dispatches = kwargs.get("max_dispatches", 50)
        topics = kwargs.get("topics", [])
        rules = kwargs.get("rules", [])
        fallback_to_llm = kwargs.get("fallback_to_llm", True)

        # BaseAgent parameters
        tools = kwargs.get("tools", None)
        tool_executor = kwargs.get("tool_executor", None)
        init_prompt_arguments_with_longterm_memory = kwargs.get("init_prompt_arguments_with_longterm_memory", True)

        return cls(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            dispatcher=dispatcher,
            locked_fields=locked_fields,
            finalizer_name=finalizer_name,
            can_finalize_name=can_finalize_name,
            supervisor=supervisor,
            max_trials=max_trials,
            max_dispatches=max_dispatches,
            topics=topics,
            rules=rules,
            fallback_to_llm=fallback_to_llm,
            tools=tools,
            tool_executor=tool_executor,
            init_prompt_arguments_with_longterm_memory=init_prompt_arguments_with_longterm_memory,
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

    async def add_routing_message_to_chat(self, conversation, next_agent):
        if updated_prompt := conversation.prompt_argument:
            updated_prompt.next_agent = next_agent
            conversation = conversation.update(prompt_argument=updated_prompt)
            # Add JSON message to conversation to ensure a consistent chat history
            prompt_dict = asdict(updated_prompt)
            content = json.dumps(prompt_dict)
            return conversation.append(
                ChatMessage(role=Role.ASSISTANT, content=content, name=self.name, type=MessageType.CONDUCTOR)
            )
        else:
            self.logger.error(
                f"Prompt argument is None for conversation {conversation}. Cannot add routing message to chat."
            )
            raise ValueError("Prompt argument is None")

    async def invoke(
        self, conversation: Conversation, message_type=MessageType.CONVERSATION, reset_prompt=True
    ) -> Conversation:
        self.logger.debug(f"Invoke called on conductor agent {self.name}")

        conversation = self.prompt.initialize_prompt(
            agent_name=self.name,
            conversation=conversation,
            locked_fields=self.locked_fields,
            init_from_longterm_memory=self.init_prompt_arguments_with_longterm_memory,
            reset_prompt=reset_prompt,
            message_type=MessageType.CONDUCTOR,  # Explicitly set message type for conductor
            render_system_prompt=True,
        )

        if next_agent := self.evaluate_rules(conversation):
            conversation = await self.add_routing_message_to_chat(conversation, next_agent)

        elif self.fallback_to_llm and self.llm_client:
            self.logger.debug("No matching rule found, falling back to LLM routing")

            for reasoning_step in range(self.max_trials):
                self.logger.debug(f"LLM routing step {reasoning_step + 1}")
                conversation = await self.generate_validated_response(conversation, message_type=MessageType.CONDUCTOR)

                if (
                    conversation.prompt_argument is not None
                    and self.prompt.is_final(conversation.prompt_argument)
                    and not conversation.has_pending_tool_response()
                    and hasattr(conversation.prompt_argument, "next_agent")
                ):
                    self.logger.debug("Final response found in LLM routing")
                    break
        else:
            self.logger.error("No matching rule found and no LLM client available")

        # Update global longterm memory with output fields from the prompt argument
        if conversation.prompt_argument:
            conversation = conversation.update_longterm_memory(conversation.prompt_argument)
        return conversation

    async def can_finalize(self, conversation: Conversation) -> bool:
        if can_finalize_name := self.can_finalize_name:
            conversation = self.prompt.initialize_prompt(
                agent_name=self.name,
                conversation=conversation,
                locked_fields=self.locked_fields,
                init_from_longterm_memory=self.init_prompt_arguments_with_longterm_memory,
            )

            conversation = await self.add_routing_message_to_chat(conversation, can_finalize_name)
            conversation = await self.dispatcher.request_response(self.can_finalize_name, conversation)
            return conversation.prompt_argument.can_finalize or self.prompt.can_finalize(conversation.longterm_memory)
        else:
            return self.prompt.can_finalize(conversation.longterm_memory)

    async def finalize(self, conversation: Conversation) -> None:
        self.logger.debug(f"Finalize conversation on conductor agent {self.name}")

        if self.supervisor or self.finalizer_name:
            conversation = self.prompt.initialize_prompt(
                agent_name=self.name,
                conversation=conversation,
                locked_fields=self.locked_fields,
                init_from_longterm_memory=self.init_prompt_arguments_with_longterm_memory,
            )

        if self.supervisor:
            conversation = await self.add_routing_message_to_chat(conversation, self.supervisor)
            await self.dispatcher.publish(topic=self.supervisor, conversation=conversation)
        elif self.finalizer_name:
            conversation = await self.add_routing_message_to_chat(conversation, self.finalizer_name)
            await self.dispatcher.publish_final(topic=self.finalizer_name, conversation=conversation)
        else:
            conversation.final_result = self.prompt.finalize(conversation.longterm_memory)

    def fail(self, conversation: Conversation) -> dict[str, Any]:
        self.logger.debug(f"End conversation with fail on conductor agent {self.name}")
        return self.prompt.fail(conversation.longterm_memory)

    async def process_conversation(self, conversation: Conversation) -> None:
        self.logger.info(f"Process conversation on conductor agent: {self.name}")
        self.n_dispatches += 1

        if conversation.longterm_memory and await self.can_finalize(conversation):
            await self.finalize(conversation=conversation)
            self.n_dispatches = 0

        elif self.max_dispatches <= self.n_dispatches:
            formatted_response = self.fail(conversation=conversation)
            self.n_dispatches = 0
            conversation.final_result = formatted_response
        else:
            conversation = await self.invoke(conversation)
            # Get the next agent from prompt_argument and route if available
            if next_agent := conversation.prompt_argument.next_agent:
                await self.dispatcher.publish(topic=next_agent, conversation=conversation)
            else:
                self.logger.error("No next agent specified for routing. This should not happen after invoke().")
                conversation.final_result = self.fail(conversation)
