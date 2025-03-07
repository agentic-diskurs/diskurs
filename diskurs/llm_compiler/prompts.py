from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional, Self

from jinja2 import Template

from diskurs.entities import ChatMessage, MessageType, PromptArgument, Role
from diskurs.prompt import BasePrompt, SystemPromptArg, UserPromptArg, validate_json
from diskurs.registry import register_prompt
from diskurs.utils import load_template_from_package


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


@register_prompt("llm_compiler_prompt")
class LLMCompilerPrompt(BasePrompt):
    """Prompt implementation for the LLM Compiler agent."""

    user_prompt_argument = PlanningUserPromptArgument
    system_prompt_argument = PlanningSystemPromptArgument

    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        summary_user_template: Template,
        system_prompt_argument_class: type[SystemPromptArg],
        user_prompt_argument_class: type[UserPromptArg],
        is_valid: Callable[[Any], bool] | None = None,
        is_final: Callable[[Any], bool] | None = None,
    ):
        super().__init__(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            system_prompt_argument_class=system_prompt_argument_class,
            user_prompt_argument_class=user_prompt_argument_class,
            return_json=False,
            is_valid=is_valid,
            is_final=is_final,
        )

        self.summary_user_template = summary_user_template

    @classmethod
    def create(cls, **kwargs) -> Self:
        """
        Create a new LLMCompilerPrompt instance.

        :param kwargs: Configuration parameters
        :return: A new LLMCompilerPrompt instance
        """
        # Use default classes from this module if not specified
        location = kwargs.get("location", None)

        user_arg_cls = PlanningUserPromptArgument
        system_arg_cls = PlanningSystemPromptArgument

        agent_description = ""
        if location:
            agent_description_filename = kwargs.get("agent_description_filename", "agent_description.txt")
            agent_description_path = location / agent_description_filename
            if agent_description_path.exists():
                with open(agent_description_path, "r") as f:
                    agent_description = f.read()

        planning_system_template = load_template_from_package(
            "diskurs.assets", "llm_compiler_planning_system_template.jinja2"
        )
        planning_user_template = load_template_from_package(
            "diskurs.assets", "llm_compiler_planning_user_template.jinja2"
        )
        summary_user_template = load_template_from_package(
            "diskurs.assets", "llm_compiler_summary_user_template.jinja2"
        )
        json_formatting_template = load_template_from_package("diskurs.assets", "json_formatting.jinja2")

        instance = cls(
            agent_description=agent_description,
            system_template=planning_system_template,
            user_template=planning_user_template,
            summary_user_template=summary_user_template,
            system_prompt_argument_class=system_arg_cls,
            user_prompt_argument_class=user_arg_cls,
        )

        return instance

    def create_system_prompt_argument(self, **prompt_args) -> PlanningSystemPromptArgument:
        return PlanningSystemPromptArgument(**prompt_args)

    def create_user_prompt_argument(self, **prompt_args) -> PlanningUserPromptArgument:
        return PlanningUserPromptArgument(**prompt_args)

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
            parsed_json = validate_json(llm_response)

            # If the parsed JSON is already an array, use it directly
            if isinstance(parsed_json, list):
                execution_plan = parsed_json
            # If it's a dict with steps or similar key, extract that
            elif isinstance(parsed_json, dict) and any(
                key in parsed_json for key in ["steps", "execution_plan", "plan"]
            ):
                for key in ["steps", "execution_plan", "plan"]:
                    if key in parsed_json and isinstance(parsed_json[key], list):
                        execution_plan = parsed_json[key]
                        break
                else:
                    execution_plan = [parsed_json]  # Wrap the dict in a list as fallback
            else:
                # Treat as a single step if it's a dict
                execution_plan = [parsed_json] if isinstance(parsed_json, dict) else []

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
