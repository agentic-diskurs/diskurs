from pathlib import Path

import pytest

from prompt import MultistepPrompt


@pytest.fixture
def prompt_instance():
    return MultistepPrompt.create(
        location=Path(__file__).parent / "prompt_test_files",
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
