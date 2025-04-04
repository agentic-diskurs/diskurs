from dataclasses import dataclass

from diskurs.entities import PromptArgument
from diskurs import PromptValidationError


@dataclass
class SecondSystemPromptArgument(PromptArgument):
    agent_name: str = "Jane"
    topic: str = "food"
    mode: str = "friendly"


@dataclass
class SecondUserPromptArgument(PromptArgument):
    name: str = ""
    topic: str = ""
    user_question: str = ""
    answer: str = ""


def is_valid(arg: SecondUserPromptArgument) -> bool:
    if not arg.name:
        raise PromptValidationError("Please extract the user's name")
    if not arg.topic:
        raise PromptValidationError("Please extract the topic of the user's query")
    if not arg.user_question:
        raise PromptValidationError("Please extract a concise description of the user's question")
    return True


def is_final(arg: SecondUserPromptArgument) -> bool:
    if len(arg.answer) > 10:
        return True
