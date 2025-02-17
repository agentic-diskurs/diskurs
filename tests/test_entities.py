from dataclasses import dataclass

from conftest import MyLongtermMemory, MyUserPromptArgument, conductor_mock, conductor_mock2, conversation
from diskurs import ImmutableConversation, PromptArgument
from diskurs.entities import ChatMessage, Role, prompt_field, PromptField
from typing import Annotated, get_type_hints


def test_basic_update():
    conversation = ImmutableConversation()
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
    conversation = ImmutableConversation()
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
    conversation = ImmutableConversation()
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


def test_conversation_to_dict(conversation):

    conversation_dict = conversation.to_dict()

    assert isinstance(conversation_dict, dict)
    assert isinstance(conversation_dict["chat"], list)
    assert conversation_dict["chat"][0]["role"] == "user"
    assert conversation_dict["chat"][0]["content"] == "Hello, world!"
    assert conversation_dict["longterm_memory"]["my_conductor"]["field1"] == "longterm_val1"
    assert conversation_dict["active_agent"] == "my_conductor"


def test_conversation_from_dict(conversation, conductor_mock, conductor_mock2):

    conversation_dict = conversation.to_dict()
    new_conversation = ImmutableConversation.from_dict(
        data=conversation_dict, agents=[conductor_mock, conductor_mock2]
    )

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


@dataclass
class PromptFieldTestDataclass(PromptArgument):
    # Test different field configurations
    visible_field: str = "visible"
    hidden_field: Annotated[str, prompt_field(include=False)] = "hidden"
    default_visible: Annotated[str, prompt_field()] = "default visible"


def test_prompt_field_creation():
    # Test creating a PromptField directly
    field = prompt_field(include=True)
    assert isinstance(field, PromptField)
    assert field.include is True

    field = prompt_field(include=False)
    assert isinstance(field, PromptField)
    assert field.include is False


def test_prompt_field_default():
    # Test default value (include=True)
    field = prompt_field()
    assert field.include is True
    assert field.should_include() is True


def test_prompt_field_in_dataclass():
    # Test prompt fields within a dataclass
    instance = PromptFieldTestDataclass()

    # Get field metadata using typing.get_type_hints() with include_extras=True
    hints = get_type_hints(PromptFieldTestDataclass, include_extras=True)

    # Hidden field should have PromptField with include=False
    hidden_meta = hints["hidden_field"].__metadata__[0]
    assert isinstance(hidden_meta, PromptField)
    assert hidden_meta.should_include() is False

    # Default visible field should have PromptField with include=True
    default_meta = hints["default_visible"].__metadata__[0]
    assert isinstance(default_meta, PromptField)
    assert default_meta.should_include() is True

    # Regular field should not have PromptField metadata
    assert "__metadata__" not in dir(hints["visible_field"])


def test_prompt_field_str_representation():
    field = prompt_field(include=False)
    assert str(field) == "PromptField(include=False)"

    field = prompt_field(include=True)
    assert str(field) == "PromptField(include=True)"


def test_prompt_field_should_include():
    # Test the should_include() method directly
    field = prompt_field(include=True)
    assert field.should_include() is True

    field = prompt_field(include=False)
    assert field.should_include() is False


def test_prompt_field_in_inheritance():
    # Test that prompt fields work correctly with inheritance
    @dataclass
    class ChildPromptClass(PromptFieldTestDataclass):
        child_visible: Annotated[str, prompt_field(include=True)] = "child visible"
        child_hidden: Annotated[str, prompt_field(include=False)] = "child hidden"

    hints = get_type_hints(ChildPromptClass, include_extras=True)

    # Check parent class fields are preserved
    parent_hidden_meta = hints["hidden_field"].__metadata__[0]
    assert parent_hidden_meta.should_include() is False

    # Check child class fields
    child_hidden_meta = hints["child_hidden"].__metadata__[0]
    assert child_hidden_meta.should_include() is False

    child_visible_meta = hints["child_visible"].__metadata__[0]
    assert child_visible_meta.should_include() is True
