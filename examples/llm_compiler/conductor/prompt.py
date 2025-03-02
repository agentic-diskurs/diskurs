from dataclasses import dataclass
from diskurs.entities import LongtermMemory


@dataclass
class LongTermMemory(LongtermMemory):
    user_query: str = ""
    answer: str = ""


def can_finalize(longterm_memory: LongTermMemory) -> bool:
    # We can finalize when we have processed the request through the compiler
    return hasattr(longterm_memory, "answer") and longterm_memory.answer != ""


def finalize(longterm_memory: LongTermMemory) -> str:
    return longterm_memory.answer
