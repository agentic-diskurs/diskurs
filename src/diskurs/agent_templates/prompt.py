from dataclasses import dataclass

from diskurs.entities import PromptArgument


@dataclass
class MySystemPromptArgument(PromptArgument):
    # Your implementation here
    pass


@dataclass
class MyUserPromptArgument(PromptArgument):
    # Your implementation here
    pass


def is_valid(prompt_arguments: MyUserPromptArgument) -> bool:
    # Your implementation here
    return True


def is_final(prompt_arguments: MyUserPromptArgument) -> bool:
    # Your implementation here
    return True
