from dataclasses import dataclass
from typing import Any, Optional
from diskurs.entities import LongtermMemory


@dataclass
class LongTermMemory(LongtermMemory):
    user_query: str = ""
    context: Optional[dict[str, Any]] = None


def can_finalize(longterm_memory: LongTermMemory) -> bool:
    # We can finalize when we have processed the request through the compiler
    return hasattr(longterm_memory, "context") and longterm_memory.context is not None
