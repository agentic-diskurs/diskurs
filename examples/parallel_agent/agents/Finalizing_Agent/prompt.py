from dataclasses import dataclass, field
from typing import Optional

from diskurs import PromptArgument, LLMClient, Conversation, CallTool
from diskurs.entities import JsonSerializable


@dataclass(kw_only=True)
class URLInstruction(JsonSerializable):
    url: str
    list_name: str
    action: str


@dataclass(kw_only=True)
class FinalizingUserPrompt(PromptArgument):
    instructions: Optional[list[URLInstruction]] = field(default_factory=list)


async def heuristic_sequence(
    conversation: Conversation, call_tool: Optional[CallTool], llm_client: Optional[LLMClient]
) -> Conversation:

    return conversation
