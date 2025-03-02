from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from diskurs.entities import ChatMessage, MessageType, PromptArgument, Role, prompt_field
from diskurs.prompt import validate_json
from diskurs.protocols import MultistepPrompt
from diskurs.utils import load_template_from_package
from diskurs.registry import register_prompt


@dataclass
class PlanningSystemPromptArgument(PromptArgument):
    """System prompt arguments for planning."""

    tools: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PlanningUserPromptArgument(PromptArgument):
    """User prompt arguments for planning."""

    user_query: str = ""
    execution_plan: Optional[Dict[str, Any]] = None


@dataclass
class SummarySystemPromptArgument(PromptArgument):
    """System prompt arguments for summarizing results."""

    user_query: str = ""
    executed_steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SummaryUserPromptArgument(PromptArgument):
    """User prompt arguments for summarizing results."""

    summary: str = ""


@register_prompt("llm_compiler")
class LLMCompilerPrompt(MultistepPrompt):
    """Prompt implementation for the LLM Compiler agent."""

    user_prompt_argument = PlanningUserPromptArgument
    system_prompt_argument = PlanningSystemPromptArgument

    def create_system_prompt_argument(self, **prompt_args) -> PlanningSystemPromptArgument:
        return PlanningSystemPromptArgument(**prompt_args)

    def create_user_prompt_argument(self, **prompt_args) -> PlanningUserPromptArgument:
        return PlanningUserPromptArgument(**prompt_args)

    def render_system_template(self, name: str, prompt_args: PromptArgument, return_json: bool = True) -> ChatMessage:
        """Render the system prompt template."""
        template = load_template_from_package("diskurs.assets", "llm_compiler_planning_system_template.jinja2")

        rendered_content = template.render(**prompt_args.__dict__)

        return ChatMessage(role=Role.SYSTEM, content=rendered_content, name=name)

    def render_user_template(
        self, name: str, prompt_args: PlanningUserPromptArgument, message_type: MessageType = MessageType.CONVERSATION
    ) -> ChatMessage:
        """Render the user prompt template."""
        if prompt_args.execution_plan:
            # This is the summary phase
            template = load_template_from_package("diskurs.assets", "llm_compiler_summary_user_template.jinja2")
        else:
            # This is the planning phase
            template = load_template_from_package("diskurs.assets", "llm_compiler_planning_user_template.jinja2")

        rendered_content = template.render(**prompt_args.__dict__)

        return ChatMessage(role=Role.USER, content=rendered_content, name=name, type=message_type)

    def is_valid(self, user_prompt_argument: PlanningUserPromptArgument) -> bool:
        """Check if the prompt argument is valid."""
        if user_prompt_argument.execution_plan:
            return True
        return False

    def is_final(self, user_prompt_argument: PlanningUserPromptArgument) -> bool:
        """Check if the prompt argument represents a final state."""
        return user_prompt_argument.execution_plan is not None

    def parse_user_prompt(
        self,
        name: str,
        llm_response: str,
        old_user_prompt_argument: PlanningUserPromptArgument,
        message_type: MessageType = MessageType.CONDUCTOR,
    ) -> PlanningUserPromptArgument | ChatMessage:
        """Parse the user prompt response."""
        try:
            # Try to parse as JSON
            execution_plan = validate_json(llm_response)

            # Update the prompt argument
            return PlanningUserPromptArgument(
                user_query=old_user_prompt_argument.user_query, execution_plan=execution_plan
            )
        except Exception as e:
            # If parsing fails, return a corrective message
            return ChatMessage(
                role=Role.USER,
                content=f"I couldn't parse your response as a valid execution plan. Please provide a valid JSON array of steps. Error: {str(e)}",
                name=name,
                type=message_type,
            )
