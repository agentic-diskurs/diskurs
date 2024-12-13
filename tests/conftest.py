import inspect
from dataclasses import dataclass
from typing import Type
from unittest.mock import Mock, AsyncMock

import pytest

from diskurs import (
    LongtermMemory,
    PromptArgument,
    ConductorAgent,
    ImmutableConversation,
    MultistepPrompt,
)
from diskurs.entities import ChatMessage, Role, MessageType
from diskurs.protocols import ConductorPrompt


@dataclass
class MyLongtermMemory(LongtermMemory):
    field1: str = ""
    field2: str = ""
    field3: str = ""


class MyLongtermMemory2(LongtermMemory):
    field4: str = ""
    field5: str = ""
    field6: str = ""


@dataclass
class MyUserPromptArgument(PromptArgument):
    field1: str = ""
    field2: str = ""
    field3: str = ""


class MySystemPromptArgument(PromptArgument):
    field1: str = ""
    field2: str = ""


@pytest.fixture
def prompt_arguments():
    return MyUserPromptArgument, MySystemPromptArgument


@pytest.fixture
def longterm_memories():
    return MyLongtermMemory, MyLongtermMemory2


def create_conductor_mock(name, system_prompt_argument, user_prompt_argument, longterm_memory):
    agent_mock = Mock(spec=ConductorAgent)
    prompt = Mock(spec=ConductorPrompt)
    prompt.system_prompt_argument = system_prompt_argument
    prompt.user_prompt_argument = user_prompt_argument
    prompt.longterm_memory = longterm_memory
    agent_mock.prompt = prompt
    agent_mock.name = name

    return agent_mock


@pytest.fixture
def conductor_mock():
    return create_conductor_mock(
        name="my_conductor",
        system_prompt_argument=MySystemPromptArgument(),
        user_prompt_argument=MyUserPromptArgument(),
        longterm_memory=MyLongtermMemory,
    )


@pytest.fixture()
def conductor_mock2():
    return create_conductor_mock(
        name="my_conductor_2",
        system_prompt_argument=MySystemPromptArgument(),
        user_prompt_argument=MyUserPromptArgument(),
        longterm_memory=MyLongtermMemory2,
    )


@pytest.fixture
def conversation():
    conversation = ImmutableConversation(
        conversation_id="my_conversation_id",
        user_prompt_argument=MyUserPromptArgument(
            field1="user prompt field 1",
            field2="user prompt field 2",
            field3="user prompt field 3",
        ),
        chat=[
            ChatMessage(role=Role.USER, content="Hello, world!", name="Alice"),
            ChatMessage(
                role=Role.USER, content="{'next_agent': 'my_agent'}", name="Alice", type=MessageType.CONDUCTOR
            ),
        ],
        longterm_memory={
            "my_conductor": MyLongtermMemory(
                field1="longterm_val1",
                field2="longterm_val2",
                field3="longterm_val3",
                user_query="How's the weather?",
            ),
            "my_conductor_2": MyLongtermMemory2(
                user_query="How's the aquarium?",
            ),
        },
        active_agent="my_conductor",
    )
    return conversation


def are_classes_structurally_similar(class_a: Type, class_b: Type) -> bool:
    """
    Check if two classes are structurally similar, i.e. they have the same class name, parent classes, and fields.
    This can be useful, if we want to check if two classes are the same, but one has been dynamically imported
    :param class_a: The first class to compare
    :param class_b: The second class to compare
    :return: True if the classes are structurally similar, False otherwise
    """
    # Check class names
    if class_a.__name__ != class_b.__name__:
        return False

    # Check parent classes
    parents_a = set(inspect.getmro(class_a))
    parents_b = set(inspect.getmro(class_b))
    if not parents_a.intersection(parents_b):
        return False

    # Get type hints and attributes of each class
    type_hints_a = inspect.get_annotations(class_a)
    type_hints_b = inspect.get_annotations(class_b)

    # Check field names and types
    if type_hints_a.keys() != type_hints_b.keys():
        return False

    for field, type_hint_a in type_hints_a.items():
        type_hint_b = type_hints_b.get(field)
        if type_hint_a != type_hint_b:
            return False

    return True


def create_prompt(user_prompt_argument):
    prompt = AsyncMock(spec=MultistepPrompt)
    prompt.create_system_prompt_argument.return_value = AsyncMock()
    prompt.create_user_prompt_argument.return_value = user_prompt_argument
    prompt.render_user_template.return_value = ChatMessage(
        role=Role.USER,
        name="my_multistep",
        content="rendered template",
        type=MessageType.CONVERSATION,
    )
    prompt.is_final.return_value = True
    prompt.user_prompt_argument = user_prompt_argument

    return prompt


@pytest.fixture
def mock_prompt():
    return create_prompt(MyUserPromptArgument())


@dataclass
class MyExtendedLongtermMemory(LongtermMemory):
    field2: str = ""
    field3: str = ""


@dataclass
class MySourceUserPromptArgument(PromptArgument):
    field3: str = ""
    field4: str = ""


@dataclass
class MyExtendedUserPromptArgument(PromptArgument):
    field1: str = "extended user prompt field 1"
    field2: str = "extended user prompt field 2"
    field3: str = "extended user prompt field 3"
    field4: str = "extended user prompt field 4"


@pytest.fixture
def mock_extended_prompt():
    return create_prompt(MyExtendedUserPromptArgument())


@pytest.fixture
def extended_conversation():
    conversation = ImmutableConversation(
        conversation_id="my_conversation_id",
        user_prompt_argument=MySourceUserPromptArgument(
            field3="user prompt field 3",
            field4="user prompt field 4",
        ),
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
        longterm_memory={
            "my_conductor": MyExtendedLongtermMemory(
                field2="longterm val 2",
                field3="longterm val 3",
                user_query="longterm user query",
            ),
        },
        active_agent="my_conductor",
    )
    return conversation


@pytest.fixture
def finalizer_conversation():
    conversation = ImmutableConversation(
        conversation_id="my_conversation_id",
        user_prompt_argument=MyUserPromptArgument(field1="user prompt field 1", field2="user prompt field 2"),
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
        longterm_memory={
            "my_conductor": MyLongtermMemory(user_query="longterm user query"),
        },
        active_agent="my_conductor",
    )
    return conversation
