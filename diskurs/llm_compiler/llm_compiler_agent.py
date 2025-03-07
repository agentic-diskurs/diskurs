from typing import Optional, Self

from diskurs.agent import get_last_conductor_name, is_previous_agent_conductor
from diskurs.multistep_agent import MultiStepAgent
from diskurs.registry import register_agent
from diskurs.entities import MessageType, ChatMessage, Role, ToolDescription
from diskurs.protocols import LLMClient, ConversationDispatcher, Conversation, ToolExecutor

from diskurs.llm_compiler.parallel_executor import ParallelExecutor
from diskurs.llm_compiler.entities import ExecutionPlan, PlanStep
from diskurs.llm_compiler.prompts import LLMCompilerPrompt
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
            system_prompt_argument=self.prompt.create_system_prompt_argument(tools=self.tools),
            user_prompt_argument=self.prompt.create_user_prompt_argument(),
        )
        if self.init_prompt_arguments_with_longterm_memory:
            conversation = conversation.update_prompt_argument_with_longterm_memory(
                conductor_name=get_last_conductor_name(conversation.chat)
            )

        # Apply previous agent info if needed
        if not is_previous_agent_conductor(conversation) and self.init_prompt_arguments_with_previous_agent:
            conversation = conversation.update_prompt_argument_with_previous_agent(previous_user_prompt_augment)

        # 1. Planning phase: Generate an execution plan
        conversation = await self.generate_validated_response(conversation)
        if not self.prompt.is_valid(conversation.user_prompt_argument):
            return conversation

        # 2. Execution phase: Execute the plan in parallel
        execution_plan_data = conversation.user_prompt_argument.execution_plan

        # Enhanced defensive parsing of execution plan
        step_list = []
        if isinstance(execution_plan_data, list):
            step_list = execution_plan_data
        elif isinstance(execution_plan_data, dict) and any(
            key in execution_plan_data for key in ["steps", "execution_plan", "plan"]
        ):
            for key in ["steps", "execution_plan", "plan"]:
                if key in execution_plan_data and isinstance(execution_plan_data[key], list):
                    step_list = execution_plan_data[key]
                    break

        # Convert to our entity classes
        steps = []
        for i, step_dict in enumerate(step_list):
            if isinstance(step_dict, dict):
                try:
                    steps.append(
                        PlanStep(
                            step_id=step_dict.get("step_id", f"step_{i+1}"),
                            description=step_dict.get("description", ""),
                            function=step_dict.get("function", ""),
                            parameters=step_dict.get("parameters", {}),
                            depends_on=step_dict.get("depends_on", []),
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error creating PlanStep: {e}")
                    self.logger.error(f"Problematic step_dict: {step_dict}")

        # Continue only if we have steps to execute
        if not steps:
            self.logger.warning("No valid execution steps found in the plan")
            return conversation

        execution_plan = ExecutionPlan(steps=steps, user_query=conversation.user_prompt_argument.user_query)

        # Execute the plan
        executor = ParallelExecutor(self.tool_executor.call_tool)
        executed_plan = await executor.execute_plan(execution_plan)

        # 3. Summarization phase: Generate a summary of the results
        # Create a new conversation for summarization
        summary_system_prompt = self.prompt.create_system_prompt_argument(
            user_query=executed_plan.user_query, executed_steps=[step.__dict__ for step in executed_plan.steps]
        )

        summary_user_prompt = self.prompt.create_user_prompt_argument(
            user_query=executed_plan.user_query, execution_plan=None  # Reset so we get the summary template
        )

        # Update conversation with new prompts for summary generation
        conversation = conversation.update(
            system_prompt_argument=summary_system_prompt,
            user_prompt_argument=summary_user_prompt,
        )

        # Render the summary templates
        summary_system_message = ChatMessage(
            role=Role.SYSTEM, content=self.render_summary_template(executed_plan), name=self.name
        )

        summary_user_message = ChatMessage(
            role=Role.USER,
            content=f"Provide a comprehensive response to the original query: {executed_plan.user_query}",
            name=self.name,
        )

        # Update conversation with the summary messages
        conversation = conversation.append(summary_system_message)
        conversation = conversation.append(summary_user_message)

        # Generate the final summary response
        final_response = await self.llm_client.generate(
            conversation=conversation, tools=None, message_type=message_type
        )

        # Update the user prompt argument with the summary
        final_user_prompt = self.prompt.create_user_prompt_argument(
            user_query=executed_plan.user_query,
            execution_plan={
                "executed_steps": [step.__dict__ for step in executed_plan.steps],
                "summary": final_response.last_message.content,
            },
        )

        return final_response.update(user_prompt_argument=final_user_prompt)

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
