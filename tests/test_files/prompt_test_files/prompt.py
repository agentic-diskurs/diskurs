from dataclasses import dataclass, field

from diskurs import PromptValidationError
from diskurs.entities import JsonSerializable, PromptArgument


@dataclass
class MyPromptArgument(PromptArgument):
    agent_name: str = "Jane"
    topic: str = "food"
    mode: str = "friendly"
    name: str = ""
    user_question: str = ""
    answer: str = ""


def is_valid(arg: MyPromptArgument) -> bool:
    if not arg.name:
        raise PromptValidationError("Please extract the user's name")
    if not arg.topic:
        raise PromptValidationError("Please extract the topic of the user's query")
    if not arg.user_question:
        raise PromptValidationError("Please extract a concise description of the user's question")
    return True


def is_final(arg: MyPromptArgument) -> bool:
    if len(arg.answer) > 10:
        return True


@dataclass
class Step(JsonSerializable):
    """Represents a single step in an execution plan."""

    topic: str = ""


@dataclass
class MyUserPromptWithArrayArgument(PromptArgument):
    name: str = ""
    steps: list[Step] = field(default_factory=list)
