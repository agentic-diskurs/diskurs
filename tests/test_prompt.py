from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import are_classes_structurally_similar
from diskurs import PromptArgument, ImmutableConversation
from diskurs.entities import ChatMessage, Role
from diskurs.prompt import MultistepPrompt, PromptValidationError, ConductorPrompt, HeuristicPrompt
from diskurs.prompt import PromptParserMixin
from test_files.heuristic_agent_test_files.prompt import MyHeuristicPromptArgument
from tests.test_files.prompt_test_files.prompt import MyUserPromptArgument


@pytest.fixture
def prompt_instance():
    return MultistepPrompt.create(
        location=Path(__file__).parent / "test_files" / "prompt_test_files",
        system_prompt_argument_class="MySystemPromptArgument",
        user_prompt_argument_class="MyUserPromptArgument",
    )


@pytest.fixture
def prompt_testing_conversation(longterm_memories):
    ltm1, ltm2 = longterm_memories
    conversation = ImmutableConversation(
        conversation_id="my_conversation_id",
        user_prompt_argument=MyUserPromptArgument(
            name="",
            topic="",
            user_question="",
            answer="",
        ),
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
        longterm_memory={
            "my_conductor": ltm1(
                field1="longterm_val1",
                field2="longterm_val2",
                field3="longterm_val3",
                user_query="How's the weather?",
            ),
            "my_conductor_2": ltm2(
                user_query="How's the aquarium?",
            ),
        },
        active_agent="my_conductor",
    )
    return conversation


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


def test_parse_prompt(prompt_instance, prompt_testing_conversation):
    res = prompt_instance.parse_user_prompt(mock_llm_response, prompt_testing_conversation.user_prompt_argument)

    assert res.name == "John Doe"
    assert res.topic == "Python Programming"


def test_fail_parse_prompt(prompt_instance, prompt_testing_conversation):
    res = prompt_instance.parse_user_prompt(
        mock_illegal_llm_response, prompt_testing_conversation.user_prompt_argument
    )

    assert (
        res.content
        == "LLM response is not valid JSON. Error: Expecting ',' delimiter at line 3, column 3. Please ensure "
        + "the response is valid JSON and follows the correct format."
    )


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


prompt_config = {
    "location": Path(__file__).parent / "test_files" / "conductor_test_files",
    "user_prompt_argument_class": "ConductorUserPromptArgument",
    "system_prompt_argument_class": "ConductorSystemPromptArgument",
    "longterm_memory_class": "MyConductorLongtermMemory",
    "can_finalize_name": "can_finalize",
    "fail_name": "fail",
}


def test_conductor_custom_system_prompt():
    prompt = ConductorPrompt.create(**prompt_config)
    rendered_system_prompt = prompt.render_system_template(
        name="test_conductor",
        prompt_args=prompt.system_prompt_argument(
            agent_descriptions={"first_agent": "I am the first agent", "second_agen": "I am the second agent"}
        ),
    )
    print(rendered_system_prompt)
    assert rendered_system_prompt.content.startswith("Custom system template")


def test_parse_user_prompt_partial_update(prompt_instance, prompt_testing_conversation):
    old_user_prompt_argument = MyUserPromptArgument(name="Alice", topic="Wonderland")
    returned_property = """{
        "user_question": "Am I updated correctly?"
        }"""
    conversation_with_prompt_args = prompt_testing_conversation.update(user_prompt_argument=old_user_prompt_argument)
    print(conversation_with_prompt_args.user_prompt_argument)

    res = prompt_instance.parse_user_prompt(
        llm_response=returned_property, old_user_prompt_argument=conversation_with_prompt_args.user_prompt_argument
    )

    assert res.name == "Alice"
    assert res.topic == "Wonderland"
    assert res.user_question == "Am I updated correctly?"


def test_parse_user_prompt(prompt_instance, prompt_testing_conversation):
    res = prompt_instance.parse_user_prompt(
        '"{\\"topic\\": \\"Secure Web Gateway\\"}"', prompt_testing_conversation.user_prompt_argument
    )

    assert isinstance(res, PromptArgument)
    assert res.topic == "Secure Web Gateway"


def test_fail():
    prompt = ConductorPrompt.create(**prompt_config)
    msg = prompt.fail(prompt.longterm_memory())
    assert msg["error"] == "Failed to finalize"


heuristic_prompt_config = {
    "location": Path(__file__).parent / "test_files" / "heuristic_agent_test_files",
    "user_prompt_argument": "MyHeuristicPromptArgument",
    "heuristic_sequence_name": "heuristic_sequence",
}


def test_heuristic_prompt_create():
    prompt = HeuristicPrompt.create(**heuristic_prompt_config)

    assert callable(prompt.heuristic_sequence)
    assert are_classes_structurally_similar(prompt.user_prompt_argument, MyHeuristicPromptArgument)
