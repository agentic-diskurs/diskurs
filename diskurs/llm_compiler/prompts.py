from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, Self

from jinja2 import Template

from diskurs.entities import PromptArgument, PromptField
from diskurs.llm_compiler.entities import PlanStep
from diskurs.prompt import BasePrompt, load_template
from diskurs.registry import register_prompt
from diskurs.utils import load_template_from_package


@dataclass
class PlanningSystemPromptArgument(PromptArgument):
    """System prompt arguments for planning."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    execution_plan: Optional[list[PlanStep]] = field(default_factory=list)
    user_query: str = ""
    replan: bool = False
    replan_explanation: str = ""
    evaluate_replanning: Annotated[bool, PromptField(include=False)] = False


@dataclass
class PlanningUserPromptArgument(PromptArgument):
    """User prompt arguments for planning."""

    user_query: str = ""
    execution_plan: Optional[list[PlanStep]] = field(default_factory=list)
    answer: str = ""
    evaluate_replanning: Annotated[bool, PromptField(include=False)] = False
    replan: bool = False
    replan_explanation: str = ""


@register_prompt("llm_compiler_prompt")
class LLMCompilerPrompt(BasePrompt):
    """Prompt implementation for the LLM Compiler agent."""

    user_prompt_argument = PlanningUserPromptArgument
    system_prompt_argument = PlanningSystemPromptArgument

    @classmethod
    def create(cls, **kwargs) -> Self:
        """
        Create a new LLMCompilerPrompt instance.

        :param kwargs: Configuration parameters
        :return: A new LLMCompilerPrompt instance
        """
        # Use default classes from this module if not specified
        location = kwargs.get("location", None)
        json_formatting_filename = kwargs.get("json_formatting_filename", "")

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

        json_render_template: Template = (
            load_template(location / json_formatting_filename)
            if json_formatting_filename
            else load_template_from_package("diskurs.assets", "json_formatting.jinja2")
        )

        instance = cls(
            agent_description=agent_description,
            system_template=planning_system_template,
            user_template=planning_user_template,
            system_prompt_argument_class=system_arg_cls,
            json_formatting_template=json_render_template,
            user_prompt_argument_class=user_arg_cls,
        )

        return instance

    def create_system_prompt_argument(self, **prompt_args) -> PlanningSystemPromptArgument:
        return PlanningSystemPromptArgument(**prompt_args)

    def create_user_prompt_argument(self, **prompt_args) -> PlanningUserPromptArgument:
        return PlanningUserPromptArgument(**prompt_args)

    def is_valid(self, user_prompt_argument: PlanningUserPromptArgument) -> bool:
        """Check if the prompt argument is valid."""
        return True

    def is_final(self, user_prompt_argument: PlanningUserPromptArgument) -> bool:
        """Check if the prompt argument represents a final state."""
        return user_prompt_argument.execution_plan is not None
