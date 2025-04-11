from dataclasses import dataclass
import pytest

from diskurs.entities import (
    MessageType,
    PromptArgument,
    JsonSerializable,
    ChatMessage,
    Role,
    prompt_field,
    PromptField,
    AccessMode,
    PromptField2,
    prompt_field2,
)
from diskurs.immutable_conversation import ImmutableConversation
from .conftest import MyLongtermMemory, MyPromptArgument, EnumPromptArgument, ChatType, Priority
from typing import Annotated, get_type_hints


def test_basic_update():
    conversation = ImmutableConversation()
    longterm_memory = MyLongtermMemory(
        field1="longterm_val1", field2="longterm_val2", field3="longterm_val3", user_query="How's the weather?"
    )
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(prompt_argument=MyPromptArgument())

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.prompt_argument.field1 == "longterm_val1"
    assert updated_conversation.prompt_argument.field2 == "longterm_val2"
    assert updated_conversation.prompt_argument.field3 == "longterm_val3"


def test_partial_update():
    conversation = ImmutableConversation()
    longterm_memory = MyLongtermMemory(field1="longterm_val1", field3="longterm_val3", user_query="How's the weather?")
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(
        prompt_argument=MyPromptArgument(field1="initial_prompt_argument1", field2="initial_prompt_argument2")
    )

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.prompt_argument.field1 == "longterm_val1"
    assert updated_conversation.prompt_argument.field2 == "initial_prompt_argument2"
    assert updated_conversation.prompt_argument.field3 == "longterm_val3"


def test_empty_longterm_memory():
    conversation = ImmutableConversation()
    longterm_memory = MyLongtermMemory(user_query="How's the weather?")  # user query must always be present
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(
        prompt_argument=MyPromptArgument(field1="initial_prompt_argument1", field2="initial_prompt_argument2")
    )

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.prompt_argument == conversation.prompt_argument  # No changes


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


class TestEnumSerialization:
    """Tests focused on enum serialization"""

    def test_simple_enum_serialization(self):
        """Test direct serialization of enum values"""
        # Direct serialization
        arg = EnumPromptArgument()
        data = arg.to_dict()

        assert data["chat_type"] == "direct"
        assert data["priority"] == 1
        assert data["message_type"] is None

    def test_simple_enum_deserialization(self):
        """Test direct deserialization of enum values"""
        # Direct deserialization
        data = {"chat_type": "group", "priority": 2, "message_type": "conductor"}

        arg = EnumPromptArgument.from_dict(data)

        assert arg.chat_type == ChatType.GROUP
        assert arg.priority == Priority.HIGH
        assert arg.message_type == MessageType.CONDUCTOR

    def test_nested_enum_serialization(self):
        """Test enums within nested structures"""

        @dataclass
        class NestedStruct(JsonSerializable):
            inner_enum: ChatType = ChatType.DIRECT

        @dataclass
        class OuterStruct(JsonSerializable):
            nested: NestedStruct = None
            enum_list: list[ChatType] = None
            enum_dict: dict[str, ChatType] = None

        # Create a complex nested structure with enums
        test_obj = OuterStruct(
            nested=NestedStruct(inner_enum=ChatType.GROUP),
            enum_list=[ChatType.DIRECT, ChatType.CHANNEL],
            enum_dict={"first": ChatType.GROUP, "second": ChatType.CHANNEL},
        )

        # Serialize
        data = test_obj.to_dict()

        # Verify serialized values
        assert data["nested"]["inner_enum"] == "group"
        assert data["enum_list"] == ["direct", "channel"]
        assert data["enum_dict"] == {"first": "group", "second": "channel"}

        # Deserialize
        restored = OuterStruct.from_dict(data)

        # Verify deserialized values
        assert restored.nested.inner_enum == ChatType.GROUP
        assert restored.enum_list == [ChatType.DIRECT, ChatType.CHANNEL]
        assert restored.enum_dict == {"first": ChatType.GROUP, "second": ChatType.CHANNEL}

    def test_optional_enum_handling(self):
        """Test optional enum fields"""
        # Test with value
        arg1 = EnumPromptArgument(message_type=MessageType.CONDUCTOR)
        data1 = arg1.to_dict()
        assert data1["message_type"] == "conductor"

        # Test with None
        arg2 = EnumPromptArgument(message_type=None)
        data2 = arg2.to_dict()
        assert data2["message_type"] is None

        # Deserialize None value
        arg3 = EnumPromptArgument.from_dict({"message_type": None})
        assert arg3.message_type is None

    def test_enum_error_handling(self):
        """Test error handling with invalid enum values"""
        # Invalid string value
        with pytest.raises(ValueError):
            EnumPromptArgument.from_dict({"chat_type": "invalid_value"})

        # Invalid numeric value
        with pytest.raises(ValueError):
            EnumPromptArgument.from_dict({"priority": 99})


def test_access_mode_enum():
    """Test the AccessMode enum values and string representation"""
    assert AccessMode.INPUT.value == "input"
    assert AccessMode.OUTPUT.value == "output"
    assert AccessMode.LOCKED.value == "locked"

    assert str(AccessMode.INPUT) == "input"
    assert str(AccessMode.OUTPUT) == "output"
    assert str(AccessMode.LOCKED) == "locked"

    # Test enum from string value
    assert AccessMode("input") == AccessMode.INPUT
    assert AccessMode("output") == AccessMode.OUTPUT
    assert AccessMode("locked") == AccessMode.LOCKED


def test_prompt_field2_creation():
    """Test creating a PromptField2 directly"""
    # Test default (OUTPUT mode)
    field = prompt_field2()
    assert isinstance(field, PromptField2)
    assert field.access_mode == AccessMode.OUTPUT
    assert field.should_include() is True  # OUTPUT mode should be included (backward compatibility)
    assert field.is_output() is True
    assert field.is_input() is False
    assert field.is_locked() is False

    # Test INPUT mode
    field = prompt_field2(mode=AccessMode.INPUT)
    assert field.access_mode == AccessMode.INPUT
    assert field.should_include() is False  # INPUT mode should not be included (backward compatibility)
    assert field.is_output() is False
    assert field.is_input() is True
    assert field.is_locked() is False

    # Test LOCKED mode
    field = prompt_field2(mode=AccessMode.LOCKED)
    assert field.access_mode == AccessMode.LOCKED
    assert field.should_include() is False  # LOCKED mode should not be included in output
    assert field.is_output() is False
    assert field.is_input() is False
    assert field.is_locked() is True


def test_prompt_field2_str_representation():
    """Test string representation of PromptField2"""
    field = prompt_field2(mode=AccessMode.INPUT)
    assert str(field) == "PromptField2(access_mode=AccessMode.INPUT)"

    field = prompt_field2(mode=AccessMode.OUTPUT)
    assert str(field) == "PromptField2(access_mode=AccessMode.OUTPUT)"

    field = prompt_field2(mode=AccessMode.LOCKED)
    assert str(field) == "PromptField2(access_mode=AccessMode.LOCKED)"


def test_prompt_field2_in_dataclass():
    """Test PromptField2 within a dataclass"""

    @dataclass
    class TestPromptArg2(PromptArgument):
        # Default field (no annotation)
        default_field: str = "default"
        # Field with OUTPUT mode (equivalent to include=True)
        output_field: Annotated[str, prompt_field2(mode=AccessMode.OUTPUT)] = "output"
        # Field with INPUT mode (equivalent to include=False)
        input_field: Annotated[str, prompt_field2(mode=AccessMode.INPUT)] = "input"
        # Field with LOCKED mode (new functionality)
        locked_field: Annotated[str, prompt_field2(mode=AccessMode.LOCKED)] = "locked"

    # Create an instance and get type hints
    instance = TestPromptArg2()
    hints = get_type_hints(TestPromptArg2, include_extras=True)

    # Check OUTPUT field
    output_meta = hints["output_field"].__metadata__[0]
    assert isinstance(output_meta, PromptField2)
    assert output_meta.access_mode == AccessMode.OUTPUT
    assert output_meta.should_include() is True

    # Check INPUT field
    input_meta = hints["input_field"].__metadata__[0]
    assert isinstance(input_meta, PromptField2)
    assert input_meta.access_mode == AccessMode.INPUT
    assert input_meta.should_include() is False

    # Check LOCKED field
    locked_meta = hints["locked_field"].__metadata__[0]
    assert isinstance(locked_meta, PromptField2)
    assert locked_meta.access_mode == AccessMode.LOCKED
    assert locked_meta.is_locked() is True
    assert locked_meta.should_include() is False
