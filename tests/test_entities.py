from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from diskurs import Conversation, PromptArgument, LongtermMemory, ConductorAgent
from diskurs.entities import ChatMessage, Role
from diskurs.protocols import ConductorPromptProtocol


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


def test_basic_update():
    conversation = Conversation()
    longterm_memory = MyLongtermMemory(
        field1="longterm_val1", field2="longterm_val2", field3="longterm_val3", user_query="How's the weather?"
    )
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(user_prompt_argument=MyUserPromptArgument())

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.user_prompt_argument.field1 == "longterm_val1"
    assert updated_conversation.user_prompt_argument.field2 == "longterm_val2"
    assert updated_conversation.user_prompt_argument.field3 == "longterm_val3"


def test_partial_update():
    conversation = Conversation()
    longterm_memory = MyLongtermMemory(field1="longterm_val1", field3="longterm_val3", user_query="How's the weather?")
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(
        user_prompt_argument=MyUserPromptArgument(field1="initial_prompt_argument1", field2="initial_prompt_argument2")
    )

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.user_prompt_argument.field1 == "longterm_val1"
    assert updated_conversation.user_prompt_argument.field2 == "initial_prompt_argument2"
    assert updated_conversation.user_prompt_argument.field3 == "longterm_val3"


def test_empty_longterm_memory():
    conversation = Conversation()
    longterm_memory = MyLongtermMemory(user_query="How's the weather?")  # user query must always be present
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(
        user_prompt_argument=MyUserPromptArgument(field1="initial_prompt_argument1", field2="initial_prompt_argument2")
    )

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.user_prompt_argument == conversation.user_prompt_argument  # No changes


def test_chat_message_from_dict():
    msg = ChatMessage.from_dict(
        {
            "role": "user",
            "content": "Hello, world!",
            "name": "Alice",
            "tool_call_id": "1234",
            "tool_calls": [
                {
                    "tool_call_id": "call_FthC9qRpsL5kBpwwyw6c7j4k",
                    "function_name": "get_current_temperature",
                    "arguments": {
                        "location": "San Francisco, CA",
                        "unit": "Fahrenheit",
                    },
                }
            ],
        }
    )

    assert isinstance(msg, ChatMessage)
    assert msg.role == Role.USER
    assert msg.content == "Hello, world!"
    assert msg.name == "Alice"
    assert msg.tool_call_id == "1234"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].tool_call_id == "call_FthC9qRpsL5kBpwwyw6c7j4k"


def test_chat_message_to_dict():
    msg = ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")
    msg_dict = msg.to_dict()

    assert isinstance(msg_dict, dict)
    assert msg_dict["role"] == "user"
    assert msg_dict["content"] == "Hello, world!"
    assert msg_dict["name"] == "Alice"


@dataclass
class MySerializableDataclass(PromptArgument):
    name: str
    id: str
    spirit_animal: str


def test_prompt_argument_from_dict():
    prompt_argument = MySerializableDataclass.from_dict({"name": "Jane", "id": "1234", "spirit_animal": "unicorn"})
    assert isinstance(prompt_argument, MySerializableDataclass)
    assert prompt_argument.name == "Jane"
    assert prompt_argument.id == "1234"
    assert prompt_argument.spirit_animal == "unicorn"


def test_prompt_argument_to_dict():
    prompt_argument = MySerializableDataclass(name="Jane", id="1234", spirit_animal="unicorn")
    arg_dict = prompt_argument.to_dict()

    assert isinstance(arg_dict, dict)
    assert arg_dict["name"] == "Jane"
    assert arg_dict["id"] == "1234"
    assert arg_dict["spirit_animal"] == "unicorn"


@pytest.fixture
def conversation():
    conversation = Conversation(
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
        user_prompt_argument=MyUserPromptArgument(
            field1="user prompt field 1",
            field2="user prompt field 2",
            field3="user prompt field 3",
        ),
        longterm_memory={
            "my_conductor": MyLongtermMemory(
                field1="longterm_val1",
                field2="longterm_val2",
                field3="longterm_val3",
                user_query="How's the weather?",
            ),
            "my_conductor2": MyLongtermMemory2(
                user_query="How's the aquarium?",
            ),
        },
        active_agent="my_conductor",
    )
    return conversation


def test_conversation_to_dict(conversation):

    conversation_dict = conversation.to_dict()

    assert isinstance(conversation_dict, dict)
    assert isinstance(conversation_dict["chat"], list)
    assert conversation_dict["chat"][0]["role"] == "user"
    assert conversation_dict["chat"][0]["content"] == "Hello, world!"
    assert conversation_dict["longterm_memory"]["my_conductor"]["field1"] == "longterm_val1"
    assert conversation_dict["active_agent"] == "my_conductor"


def create_conductor_mock(name, system_prompt_argument, user_prompt_argument, longterm_memory):
    agent_mock = Mock(spec=ConductorAgent)
    prompt = Mock(spec=ConductorPromptProtocol)
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
        name="my_conductor2",
        system_prompt_argument=MySystemPromptArgument(),
        user_prompt_argument=MyUserPromptArgument(),
        longterm_memory=MyLongtermMemory2,
    )


def test_conversation_from_dict(conversation, conductor_mock, conductor_mock2):

    conversation_dict = conversation.to_dict()
    new_conversation = Conversation.from_dict(data=conversation_dict, agents=[conductor_mock, conductor_mock2])

    assert new_conversation.chat[0].role == conversation.chat[0].role
    assert new_conversation.chat[0].content == conversation.chat[0].content
    assert (
        new_conversation._longterm_memory["my_conductor"].field1
        == conversation._longterm_memory["my_conductor"].field1
    )
    assert (
        new_conversation._longterm_memory["my_conductor"].field2
        == conversation._longterm_memory["my_conductor"].field2
    )
    assert (
        new_conversation._longterm_memory["my_conductor"].field3
        == conversation._longterm_memory["my_conductor"].field3
    )
    assert (
        new_conversation._longterm_memory["my_conductor"].user_query
        == conversation._longterm_memory["my_conductor"].user_query
    )
    assert new_conversation.user_prompt_argument.field1 == conversation.user_prompt_argument.field1
    assert new_conversation.user_prompt_argument.field2 == conversation.user_prompt_argument.field2
    assert new_conversation.user_prompt_argument.field3 == conversation.user_prompt_argument.field3
    assert new_conversation.user_prompt == conversation.user_prompt
    assert new_conversation.system_prompt == conversation.system_prompt
    assert new_conversation.active_agent == conversation.active_agent
