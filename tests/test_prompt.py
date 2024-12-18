from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, Mock

import pytest

from conftest import are_classes_structurally_similar
from diskurs import ImmutableConversation, ToolExecutor
from diskurs.entities import ChatMessage, Role, PromptArgument
from diskurs.prompt import (
    MultistepPrompt,
    validate_dataclass,
    PromptValidationError,
    ConductorPrompt,
    HeuristicPrompt,
)
from test_files.heuristic_agent_test_files.prompt import MyHeuristicPromptArgument
from test_files.prompt_test_files.prompt import MyUserPromptArgument


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
    res = prompt_instance.parse_user_prompt(
        name="test_agent",
        llm_response=mock_llm_response,
        old_user_prompt_argument=prompt_testing_conversation.user_prompt_argument,
    )

    assert res.name == "John Doe"
    assert res.topic == "Python Programming"


def test_fail_parse_prompt(prompt_instance, prompt_testing_conversation):
    res = prompt_instance.parse_user_prompt(
        name="test_agent",
        llm_response=mock_illegal_llm_response,
        old_user_prompt_argument=prompt_testing_conversation.user_prompt_argument,
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


@dataclass
class ExampleTypedPromptArg(PromptArgument):
    url: Optional[str] = ""
    is_valid: Optional[bool] = None
    comments: Optional[list[str]] = None


def test_validate_dataclass():
    response = {"url": "https://diskurs.dev", "comment": "Do what thou wilt", "username": "Jane"}
    res_prompt_arg = validate_dataclass(parsed_response=response, user_prompt_argument=ExamplePromptArg)

    assert (
        res_prompt_arg.url == response["url"]
        and res_prompt_arg.comment == response["comment"]
        and res_prompt_arg.username == response["username"]
    )


def test_validate_dataclass_typed():
    response = {
        "url": "https://diskurs.dev",
        "is_valid": "true",
        "comments": ["Do what thou wilt", "Do what thou wilt"],
    }
    res_prompt_arg = validate_dataclass(parsed_response=response, user_prompt_argument=ExampleTypedPromptArg)

    assert (
        res_prompt_arg.url == response["url"]
        and res_prompt_arg.is_valid == True
        and type(res_prompt_arg.comments) == list
        and type(res_prompt_arg.comments[0]) == str
    )


def test_validate_dataclass_typed_empty():
    response = {
        "url": "https://diskurs.dev",
        "comments": ["Do what thou wilt", "Do what thou wilt"],
    }
    res_prompt_arg = validate_dataclass(parsed_response=response, user_prompt_argument=ExampleTypedPromptArg)

    assert (
        res_prompt_arg.url == response["url"]
        and type(res_prompt_arg.comments) == list
        and res_prompt_arg.is_valid is None
        and type(res_prompt_arg.comments[0]) == str
    )


def test_validate_dataclass_additional_fields():
    response = {"url": "https://www.diskurs.dev", "foo": "just foo"}

    with pytest.raises(PromptValidationError) as exc_info:
        res_prompt_arg = validate_dataclass(parsed_response=response, user_prompt_argument=ExamplePromptArg)
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


prompt_config_no_finalize = {
    "location": Path(__file__).parent / "test_files" / "conductor_no_finalize_test_files",
    "user_prompt_argument_class": "ConductorUserPromptArgument",
    "system_prompt_argument_class": "ConductorSystemPromptArgument",
    "longterm_memory_class": "MyConductorLongtermMemory",
    "can_finalize_name": "can_finalize",
    "fail_name": "fail",
}


def test_conductor_no_finalize_function():
    prompt = ConductorPrompt.create(**prompt_config_no_finalize)

    assert prompt._finalize is None


def test_parse_user_prompt_partial_update(prompt_instance, prompt_testing_conversation):
    old_user_prompt_argument = MyUserPromptArgument(name="Alice", topic="Wonderland")
    returned_property = """{
        "user_question": "Am I updated correctly?"
        }"""
    conversation_with_prompt_args = prompt_testing_conversation.update(user_prompt_argument=old_user_prompt_argument)
    print(conversation_with_prompt_args.user_prompt_argument)

    res = prompt_instance.parse_user_prompt(
        name="test_agent",
        llm_response=returned_property,
        old_user_prompt_argument=conversation_with_prompt_args.user_prompt_argument,
    )

    assert res.name == "Alice"
    assert res.topic == "Wonderland"
    assert res.user_question == "Am I updated correctly?"


def test_parse_user_prompt(prompt_instance, prompt_testing_conversation):
    res = prompt_instance.parse_user_prompt(
        name="test_agent",
        llm_response='"{\\"topic\\": \\"Secure Web Gateway\\"}"',
        old_user_prompt_argument=prompt_testing_conversation.user_prompt_argument,
    )

    assert isinstance(res, PromptArgument)
    assert res.topic == "Secure Web Gateway"


def test_fail():

    prompt = ConductorPrompt.create(**prompt_config)
    msg = prompt.fail(prompt.longterm_memory())
    assert msg["error"] == "Failed to finalize"


heuristic_prompt_config = {
    "location": Path(__file__).parent / "test_files" / "heuristic_agent_test_files",
    "user_prompt_argument_class": "MyHeuristicPromptArgument",
    "heuristic_sequence_name": "heuristic_sequence",
}


def test_heuristic_prompt_create():
    prompt = HeuristicPrompt.create(**heuristic_prompt_config)

    assert callable(prompt.heuristic_sequence)
    assert are_classes_structurally_similar(prompt.user_prompt_argument, MyHeuristicPromptArgument)


@pytest.fixture
def tool_executor():
    executor = Mock(spec=ToolExecutor)
    return executor


@pytest.fixture
def heuristic_prompt(conversation):
    prompt = AsyncMock(spec=HeuristicPrompt)  # Change to AsyncMock

    # Create an instance of MyHeuristicPromptArgument to be returned
    prompt_arg_instance = MyHeuristicPromptArgument()

    # Configure create_user_prompt_argument to return the specific instance
    prompt.create_user_prompt_argument.return_value = prompt_arg_instance

    # Side effect function for heuristic_sequence
    async def heuristic_sequence_side_effect(user_prompt_argument, metadata, call_tool):
        # Assert that heuristic_sequence is called with the correct Conversation instance
        assert user_prompt_argument == conversation.user_prompt_argument, "Expected correct user_prompt_argument"
        assert metadata == conversation.metadata, "Expected correct metadata"
        # Return the specific instance
        return prompt_arg_instance

    # Set the heuristic_sequence to the side effect function
    prompt.heuristic_sequence.side_effect = heuristic_sequence_side_effect

    return prompt


@pytest.mark.asyncio
async def test_heuristic_prompt(heuristic_prompt, conversation):
    result = heuristic_prompt.create_user_prompt_argument()
    assert isinstance(result, MyHeuristicPromptArgument)

    result = await heuristic_prompt.heuristic_sequence(
        user_prompt_argument=conversation.user_prompt_argument,
        metadata=conversation.metadata,
        call_tool=lambda x: x,  # Mock or real function as needed
    )
    assert isinstance(result, MyHeuristicPromptArgument)


def test_create_loads_agent_description():
    location = Path(__file__).parent / "test_files" / "heuristic_agent_test_files"

    with open(location / "agent_description.txt") as f:
        agent_description = f.read()

    prompt = HeuristicPrompt.create(
        location=location,
        user_prompt_argument_class="MyHeuristicPromptArgument",
    )

    assert prompt.agent_description == agent_description
