from dataclasses import dataclass, field
from typing import Any, Optional, Self

from jinja2 import Template

from diskurs.entities import PromptArgument, InputField
from diskurs.llm_compiler.entities import PlanStep
from diskurs.prompt import BasePrompt, load_template
from diskurs.registry import register_prompt
from diskurs.utils import load_template_from_package


@dataclass
class PlanningPromptArgument(PromptArgument):
    """User prompt arguments for planning."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    user_query: str = ""
    execution_plan: Optional[list[PlanStep]] = field(default_factory=list)
    answer: str = ""
    evaluate_replanning: InputField[bool] = False
    replan: bool = False
    replan_explanation: str = ""


@register_prompt("llm_compiler_prompt")
class LLMCompilerPrompt(BasePrompt):
    """Prompt implementation for the LLM Compiler agent."""

    prompt_argument = PlanningPromptArgument

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

        prompt_argument_cls = PlanningPromptArgument

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
            json_formatting_template=json_render_template,
            prompt_argument_class=prompt_argument_cls,
        )

        return instance

    def create_prompt_argument(self, **prompt_args) -> PlanningPromptArgument:
        return PlanningPromptArgument(**prompt_args)

    def is_valid(self, prompt_argument: PlanningPromptArgument) -> bool:
        """Check if the prompt argument is valid."""
        return True

    def is_final(self, prompt_argument: PlanningPromptArgument) -> bool:
        """Check if the prompt argument represents a final state."""
        return prompt_argument.execution_plan is not None
