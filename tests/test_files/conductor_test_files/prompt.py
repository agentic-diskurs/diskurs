from dataclasses import dataclass, asdict
from typing import Optional, Any, Annotated

from diskurs.entities import PromptArgument, LongtermMemory, prompt_field, AccessMode


@dataclass
class ConductorPromptArgument(PromptArgument):
    agent_descriptions: Annotated[Optional[dict[str, str]], prompt_field(mode=AccessMode.INPUT)] = None
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


def fail(longterm_memory: MyConductorLongtermMemory) -> dict[str, Any]:
    return {"error": "Failed to finalize", **asdict(longterm_memory)}
