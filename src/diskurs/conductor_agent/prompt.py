from dataclasses import dataclass
from typing import Optional

from diskurs.entities import PromptArgument, LongtermMemory


@dataclass
class ConductorSystemPromptArgument(PromptArgument):
    agent_descriptions: dict[str, str]


@dataclass
class ConductorUserPromptArgument(PromptArgument):
    content: Optional[str] = None
    next_agent: Optional[str] = None


@dataclass
class MyConductorLongtermMemory(LongtermMemory):
    my_memory: str


def can_finalize(longterm_memory: MyConductorLongtermMemory) -> bool:
    return longterm_memory.my_memory == "I remember"
