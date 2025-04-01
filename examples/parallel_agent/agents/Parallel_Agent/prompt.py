from dataclasses import dataclass, field
from typing import Annotated, Optional
from diskurs import PromptArgument
from diskurs.entities import JsonSerializable, prompt_field


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
    instructions: Annotated[list[URLInstruction], prompt_field(include=False)] = field(default_factory=list)
