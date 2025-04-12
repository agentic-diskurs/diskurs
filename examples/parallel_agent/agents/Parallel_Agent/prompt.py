from dataclasses import dataclass, field
from typing import Optional

from diskurs import PromptArgument
from diskurs.entities import JsonSerializable, InputField


@dataclass(kw_only=True)
class ParallelMultiStepSystemPrompt(PromptArgument):
    branching: bool = True
    joining: bool = False
    urls: Optional[list[str] | str] = None
    list_name: Optional[str] = None
    action: Optional[str] = None


@dataclass(kw_only=True)
class URLInstruction(JsonSerializable):
    url: str
    list_name: str
    action: str


@dataclass(kw_only=True)
class ParallelMultiStepUserPrompt(PromptArgument):
    branching: bool = True
    joining: bool = False
    urls: Optional[list[str] | str] = None
    list_name: Optional[str] = None
    action: Optional[str] = None
    instructions: InputField[list[URLInstruction]] = field(default_factory=list)
