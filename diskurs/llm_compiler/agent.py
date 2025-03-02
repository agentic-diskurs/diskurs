from typing import Optional, List, Self

from diskurs.multistep_agent import MultiStepAgent
from diskurs.registry import register_agent
from diskurs.entities import MessageType, ChatMessage, Role, ToolDescription
from diskurs.protocols import LLMClient, ConversationDispatcher, Conversation, ToolExecutor

from .parallel_executor import ParallelExecutor
from .entities import ExecutionPlan, PlanStep
from .prompts import LLMCompilerPrompt


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

        # First, prepare the conversation with initial prompts
        conversation = self.prepare_conversation(
            conversation,
            system_prompt_argument=self.prompt.create_system_prompt_argument(
                tools=[ToolDescription.from_function(tool).__dict__ for tool in self.tools]
            ),
            user_prompt_argument=self.prompt.create_user_prompt_argument(
                user_query=conversation.user_prompt_argument.user_query if conversation.user_prompt_argument else ""
            ),
        )

        # Apply longterm memory if needed
        if self.init_prompt_arguments_with_longterm_memory:
            conversation = conversation.update_prompt_argument_with_longterm_memory(
                conductor_name=get_last_conductor_name(conversation.chat)
            )

        # Apply previous agent info if needed
        if not is_previous_agent_conductor(conversation) and self.init_prompt_arguments_with_previous_agent:
            conversation = conversation.update_prompt_argument_with_previous_agent(conversation.user_prompt_argument)

        # 1. Planning phase: Generate an execution plan
        conversation = await self.generate_validated_response(conversation)
        if not self.prompt.is_valid(conversation.user_prompt_argument):
            return conversation

        # 2. Execution phase: Execute the plan in parallel
        plan_dict = conversation.user_prompt_argument.execution_plan

        # Convert to our entity classes
        steps = []
        for step_dict in plan_dict:
            steps.append(
                PlanStep(
                    step_id=step_dict["step_id"],
                    description=step_dict["description"],
                    function=step_dict["function"],
                    parameters=step_dict["parameters"],
                    depends_on=step_dict.get("depends_on", []),
                )
            )

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
        template = load_template_from_package("diskurs.assets", "llm_compiler_summary_system_template.jinja2")

        return template.render(
            user_query=executed_plan.user_query, executed_steps=[step.__dict__ for step in executed_plan.steps]
        )
