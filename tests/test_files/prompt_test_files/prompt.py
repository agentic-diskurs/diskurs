from dataclasses import dataclass

from entities import PromptArgument
from prompt import PromptValidationError


@dataclass
class MySystemPromptArgument(PromptArgument):
    agent_name: str = "Jane"
    topic: str = "food"
    mode: str = "friendly"


@dataclass
class MyUserPromptArgument(PromptArgument):
    name: str
    topic: str = ""
    user_question: str = ""
    answer: str = ""


def is_valid(arg: MyUserPromptArgument) -> bool:
    if not arg.name:
        raise PromptValidationError("Please extract the user's name")
    if not arg.topic:
        raise PromptValidationError("Please extract the topic of the user's query")
    if not arg.user_question:
        raise PromptValidationError("Please extract a concise description of the user's question")
    return True


def is_final(arg: MyUserPromptArgument) -> bool:
    if len(arg.answer) > 10:
        return True