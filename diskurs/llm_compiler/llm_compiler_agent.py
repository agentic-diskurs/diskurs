from typing import Optional, Self

from diskurs.entities import MessageType, ToolDescription
from diskurs.llm_compiler.entities import ExecutionPlan
from diskurs.llm_compiler.parallel_executor import ParallelExecutor
from diskurs.llm_compiler.prompts import LLMCompilerPrompt, PlanningPromptArgument, PlanningPromptArgument
from diskurs.multistep_agent import MultiStepAgent
from diskurs.protocols import Conversation, ConversationDispatcher, LLMClient, ToolExecutor
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

    async def invoke(
        self, conversation: Conversation, message_type=MessageType.CONVERSATION, reset_prompt=True
    ) -> Conversation:
        self.logger.debug(f"Invoke called on LLM Compiler agent {self.name}")

        # TODO: ensure to pass prompt_argument.tools = self.tools
        conversation = self.prompt.initialize_prompt(
            agent_name=self.name,
            conversation=conversation,
            locked_fields=self.locked_fields,
            init_from_longterm_memory=self.init_prompt_arguments_with_longterm_memory,
        )

        evaluate_replanning = False

        for step in range(self.max_reasoning_steps):
            self.logger.debug(f"Step {step} in LLM Compiler agent {self.name}")

            # Generate a validated response from the LLM
            conversation = await self.generate_validated_response(conversation)
            prompt_argument: PlanningPromptArgument = conversation.prompt_argument

            if evaluate_replanning:
                prompt_argument.evaluate_replanning = False
                evaluate_replanning = False
            else:
                # Execute the plan and update flags based on step outcomes.
                executed_plan = await self.executor.execute_plan(
                    plan=prompt_argument.execution_plan, metadata=conversation.metadata
                )
                prompt_argument.execution_plan = executed_plan
                prompt_argument.execution_plan = executed_plan

                if not all(item.status == "completed" for item in executed_plan):
                    self.logger.info("Not all steps completed successfully in the plan.")
                    prompt_argument.replan = True
                    prompt_argument.replan = True
                    explanation = "Not all steps completed successfully in the plan."
                    prompt_argument.replan_explanation = explanation
                    prompt_argument.replan_explanation = explanation
                else:
                    self.logger.info("Successfully executed all tools in plan.")
                    prompt_argument.evaluate_replanning = True
                    evaluate_replanning = True

            # Update conversation with new prompt values.
            conversation = conversation.update(
                prompt_argument=prompt_argument,
                system_prompt=self.prompt.render_system_template(
                    self.name,
                ),
                user_prompt=self.prompt.render_user_template(name=self.name, prompt_args=prompt_argument),
            )

            if not evaluate_replanning and not prompt_argument.replan:
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
