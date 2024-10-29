from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from diskurs import LongtermMemory, PromptArgument, ConductorAgent, ImmutableConversation
from diskurs.entities import ChatMessage, Role
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
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
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
