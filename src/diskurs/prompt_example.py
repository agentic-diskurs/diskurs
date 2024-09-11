from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

from jinja2 import Template

from prompt import Prompt, PromptValidationError
from entities import PromptArgument

import logging

logging.basicConfig(level=logging.INFO)
# Define the template

system_template = Template(
    """
System Prompt:
--------------
- Your are a helpful assistant, your name is {{ agent_name }}.
- You are a seasoned expert in the area of {{ topic }}.
- You always keep you tone {{ mode }}.
- You are able to solve two kinds of tasks: firstly you can extract entities from the text and return valid JSON, and secondly you can answer a user query.
"""
)


@dataclass
class SystemPromptArgument(PromptArgument):
    agent_name: str = "Jane"
    topic: str = "food"
    mode: str = "friendly"


# Create a Template object
user_template = Template(
    """
User Prompt:
------------
{% if not name or not topic or not user_question %}
Extract all of the following entities from the user's query':
- name: the user's name
- topic: the topic of the user's query
- user_question: a concise description of the user's question
And return them as valid JSON:
User's query: {{ content }}
{% else %}
- The users name is {{ name }} 
- They are asking questions about {{ topic }}.
- The user's query is: {{ user_question }}.
Answer the user's question, and output the answer in a valid JSON with the key "answer".
{% endif %}
"""
)


@dataclass
class UserPromptArgument(PromptArgument):
    name: str = ""
    topic: str = ""
    user_question: str = ""
    answer: str = ""


def is_valid(arg: UserPromptArgument) -> bool:
    if not arg.name:
        raise PromptValidationError("Please extract the user's name")
    if not arg.topic:
        raise PromptValidationError("Please extract the topic of the user's query")
    if not arg.user_question:
        raise PromptValidationError(
            "Please extract a concise description of the user's question"
        )
    return True


def is_final(arg: UserPromptArgument) -> bool:
    if len(arg.answer) > 10:
        return True


prompt = Prompt(
    system_template=system_template,
    user_template=user_template,
    system_prompt_argument=SystemPromptArgument,
    user_prompt_argument=UserPromptArgument,
    is_valid=is_valid,
    is_final=is_final,
)


def run_prompt_render():
    pprint(prompt.render_system_template(SystemPromptArgument()))
    pprint(prompt.render_user_template("Hardy", UserPromptArgument()))
    pprint(
        prompt.render_user_template(
            "Hardy",
            UserPromptArgument(
                name="Hardy", topic="Surfing", user_question="Is it wavey?"
            ),
        )
    )


def validate_prompt():
    pprint(prompt.validate_prompt(name="Joanna", prompt_args=UserPromptArgument()))


def using_static_factory_method():
    prompt = Prompt.create(
        location=Path(__file__).parent.parent.parent / "tests/prompt_test_files",
        system_prompt_argument="SystemPromptArgument",
        user_prompt_argument="UserPromptArgument",
    )
    pprint(prompt.render_system_template(SystemPromptArgument()))


if __name__ == "__main__":
    # run_prompt_render()
    # validate_prompt()
    using_static_factory_method()
