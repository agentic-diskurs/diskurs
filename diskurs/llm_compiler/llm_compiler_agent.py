from typing import Optional, Self

from diskurs.agent import get_last_conductor_name, is_previous_agent_conductor
from diskurs.entities import MessageType, ToolDescription
from diskurs.llm_compiler.entities import ExecutionPlan
from diskurs.llm_compiler.parallel_executor import ParallelExecutor
from diskurs.llm_compiler.prompts import (LLMCompilerPrompt,
                                          PlanningSystemPromptArgument,
                                          PlanningUserPromptArgument)
from diskurs.multistep_agent import MultiStepAgent
from diskurs.protocols import (Conversation, ConversationDispatcher, LLMClient,
                               ToolExecutor)
from diskurs.registry import register_agent
from diskurs.utils import load_template_from_package


@register_agent("llm_compiler")
class LLMCompilerAgent(MultiStepAgent):
    """
    An agent that implements the LLM Compiler approach for parallel function calling.
    It plans, optimizes, executes in parallel, and synthesizes results.
    """

    def __init__(
        self,
        name: str,
        prompt: LLMCompilerPrompt,
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
        super().__init__(
            name=name,
            prompt=prompt,
            llm_client=llm_client,
            topics=topics,
            dispatcher=dispatcher,
            tool_executor=tool_executor,
            tools=tools,
            max_reasoning_steps=max_reasoning_steps,
            max_trials=max_trials,
            init_prompt_arguments_with_longterm_memory=init_prompt_arguments_with_longterm_memory,
            init_prompt_arguments_with_previous_agent=init_prompt_arguments_with_previous_agent,
        )
        self.executor = ParallelExecutor(call_tool=tool_executor.call_tool)

    @classmethod
    def create(
        cls,
        name: str,
        prompt: LLMCompilerPrompt,
        llm_client: LLMClient,
        **kwargs,
    ) -> Self:
        return cls(name=name, prompt=prompt, llm_client=llm_client, **kwargs)

    async def invoke(self, conversation: Conversation, message_type=MessageType.CONVERSATION) -> Conversation:
        self.logger.debug(f"Invoke called on LLM Compiler agent {self.name}")

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

        if not is_previous_agent_conductor(conversation) and self.init_prompt_arguments_with_previous_agent:
            conversation = conversation.update_prompt_argument_with_previous_agent(previous_user_prompt_augment)

        system_prompt_argument: PlanningSystemPromptArgument = conversation.system_prompt_argument
        system_prompt_argument.tools = self.tools
        system_prompt_argument.user_query = conversation.user_prompt_argument.user_query

        conversation = conversation.update(
            system_prompt_argument=system_prompt_argument,
            system_prompt=self.prompt.render_system_template(name=self.name, prompt_args=system_prompt_argument),
            user_prompt=self.prompt.render_user_template(
                name=self.name, prompt_args=conversation.user_prompt_argument
            ),
        )

        evaluate_replanning = False

        for step in range(self.max_reasoning_steps):
            self.logger.debug(f"Step {step} in LLM Compiler agent {self.name}")

            # Generate a validated response from the LLM
            conversation = await self.generate_validated_response(conversation)
            system_prompt_argument: PlanningSystemPromptArgument = conversation.system_prompt_argument
            user_prompt_argument: PlanningUserPromptArgument = conversation.user_prompt_argument

            system_prompt_argument = conversation.update_prompt_argument(user_prompt_argument, system_prompt_argument)

            if evaluate_replanning:
                system_prompt_argument.evaluate_replanning = False
                user_prompt_argument.evaluate_replanning = False
                evaluate_replanning = False
            else:
                # Execute the plan and update flags based on step outcomes.
                executed_plan = await self.executor.execute_plan(
                    plan=user_prompt_argument.execution_plan, metadata=conversation.metadata
                )
                system_prompt_argument.execution_plan = executed_plan
                user_prompt_argument.execution_plan = executed_plan

                if not all(item.status == "completed" for item in executed_plan):
                    self.logger.info("Not all steps completed successfully in the plan.")
                    system_prompt_argument.replan = True
                    user_prompt_argument.replan = True
                    explanation = "Not all steps completed successfully in the plan."
                    system_prompt_argument.replan_explanation = explanation
                    user_prompt_argument.replan_explanation = explanation
                else:
                    self.logger.info("Successfully executed all tools in plan.")
                    system_prompt_argument.evaluate_replanning = True
                    user_prompt_argument.evaluate_replanning = True
                    evaluate_replanning = True

            # Update conversation with new prompt values.
            conversation = conversation.update(
                system_prompt_argument=system_prompt_argument,
                user_prompt_argument=user_prompt_argument,
                system_prompt=self.prompt.render_system_template(self.name, prompt_args=system_prompt_argument),
                user_prompt=self.prompt.render_user_template(name=self.name, prompt_args=user_prompt_argument),
            )

            if not evaluate_replanning and not user_prompt_argument.replan:
                conversation = await self.generate_validated_response(conversation)
                break

        return conversation

    def render_summary_template(self, executed_plan: ExecutionPlan) -> str:
        """Render the summary system template."""
        try:
            template = load_template_from_package("diskurs.assets", "llm_compiler_summary_system_template.jinja2")

            return template.render(
                user_query=executed_plan.user_query, executed_steps=[step.__dict__ for step in executed_plan.steps]
            )
        except Exception as e:
            self.logger.error(f"Error rendering summary template: {e}")
            # Fallback template string
            fallback = "You are an AI assistant. Synthesize the results from these steps to answer: "
            fallback += f"\n{executed_plan.user_query}\n\n"
            for step in executed_plan.steps:
                fallback += f"Step {step.step_id}: {step.description}\nResult: {step.result}\n\n"
            return fallback
