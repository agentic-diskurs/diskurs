from dataclasses import dataclass
from typing import Optional

from diskurs.entities import PromptArgument


@dataclass
class ConductorSystemPromptArgument(PromptArgument):
    agent_descriptions: dict[str, str]


@dataclass
class ConductorUserPromptArgument(PromptArgument):
    content: Optional[str] = None


def is_valid(prompt_arguments: ConductorUserPromptArgument) -> bool:
    return True


def is_final(prompt_arguments: ConductorUserPromptArgument) -> bool:
    return True
