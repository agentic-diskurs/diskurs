from dataclasses import dataclass, asdict
from typing import Optional, Any

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
    user_query: Optional[str] = ""
    test_flag: int = 0


def can_finalize(longterm_memory: MyConductorLongtermMemory) -> bool:
    return longterm_memory.test_flag != 0


def finalize(longterm_memory: MyConductorLongtermMemory) -> dict[str, Any]:
    return asdict(longterm_memory)
