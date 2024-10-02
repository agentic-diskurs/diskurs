from dataclasses import dataclass
from pathlib import Path

import pytest

from diskurs.entities import MessageType
from diskurs.prompt import MultistepPrompt, PromptValidationError

from diskurs import PromptArgument
from diskurs.prompt import PromptParserMixin


@pytest.fixture
def prompt_instance():
    return MultistepPrompt.create(
        location=Path(__file__).parent / "test_files" / "prompt_test_files",
        system_prompt_argument_class="MySystemPromptArgument",
        user_prompt_argument_class="MyUserPromptArgument",
    )


mock_llm_response = """{
  "name": "John Doe",
  "topic": "Python Programming",
  "user_question": "What is a decorator in Python?",
  "answer": "A decorator in Python is a function that modifies the behavior of another function."
}"""

mock_illegal_llm_response = """{
  "name": "John Doe"
  "topic": "Python Programming",
  "user_question": "What is a decorator in Python?",
  "answer": "A decorator in Python is a function that modifies the behavior of another function."
}"""

mock_missing_keys_llm_response = """{
  "topic": "Python Programming",
  "user_question": "What is a decorator in Python?",
  "answer": "A decorator in Python is a function that modifies the behavior of another function."
}"""


def test_parse_prompt(prompt_instance):
    res = prompt_instance.parse_user_prompt(mock_llm_response)

    assert res.name == "John Doe"
    assert res.topic == "Python Programming"


def test_fail_parse_prompt(prompt_instance):
    res = prompt_instance.parse_user_prompt(mock_illegal_llm_response)

    assert (
        res.content
        == "LLM response is not valid JSON. Error: Expecting ',' delimiter at line 3, column 3. Please ensure "
        + "the response is valid JSON and follows the correct format."
    )


def test_missing_key_parse_prompt(prompt_instance):
    res = prompt_instance.parse_user_prompt(mock_missing_keys_llm_response)

    assert res.content == "Missing required fields: name. Valid fields are: name, topic, user_question, answer."


@dataclass
class ExamplePromptArg(PromptArgument):
    url: str = ""
    comment: str = ""
    username: str = ""


def test_validate_dataclass():
    response = {"url": "https://diskurs.dev", "comment": "Do what thou wilt", "username": "Jane"}
    res_prompt_arg = PromptParserMixin.validate_dataclass(
        parsed_response=response, user_prompt_argument=ExamplePromptArg
    )

    assert (
        res_prompt_arg.url == response["url"]
        and res_prompt_arg.comment == response["comment"]
        and res_prompt_arg.username == response["username"]
    )


def test_validate_dataclass_additional_fields():
    response = {"url": "https://diskurs.dev", "foo": "just foo"}

    with pytest.raises(PromptValidationError) as exc_info:
        res_prompt_arg = PromptParserMixin.validate_dataclass(
            parsed_response=response, user_prompt_argument=ExamplePromptArg
        )
    assert (
        str(exc_info.value)
        == "Extra fields provided: foo. Please remove them. Valid fields are: url, comment, username."
    )
