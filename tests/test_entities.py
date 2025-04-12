from dataclasses import dataclass, field
from typing import get_type_hints, Optional

import pytest

from diskurs.entities import (
    MessageType,
    PromptArgument,
    JsonSerializable,
    ChatMessage,
    Role,
    AccessMode,
    OutputField,
    InputField,
    LockedField,
)
from diskurs.immutable_conversation import ImmutableConversation
from .conftest import MyLongtermMemory, MyPromptArgument, EnumPromptArgument, ChatType, Priority


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
class AccessModeTestDataclass(PromptArgument):
    # Test different access mode configurations
    output_field: OutputField[str] = "output"  # type: ignore
    input_field: InputField[str] = "input"  # type: ignore
    locked_field: LockedField[str] = "locked"  # type: ignore
    default_field: str = "default"  # No annotation, should behave like OUTPUT


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


def test_prompt_field_in_dataclass():
    """Test the field type annotations within a dataclass"""
    instance = AccessModeTestDataclass()

    # Get field metadata using typing.get_type_hints() with include_extras=True
    hints = get_type_hints(AccessModeTestDataclass, include_extras=True)

    # Check the OUTPUT field
    output_meta = hints["output_field"].__metadata__[0]
    assert output_meta.is_output() is True
    assert output_meta.is_input() is False
    assert output_meta.is_locked() is False

    # Check the INPUT field
    input_meta = hints["input_field"].__metadata__[0]
    assert input_meta.is_output() is False
    assert input_meta.is_input() is True
    assert input_meta.is_locked() is False

    # Check the LOCKED field
    locked_meta = hints["locked_field"].__metadata__[0]
    assert locked_meta.is_output() is False
    assert locked_meta.is_input() is False
    assert locked_meta.is_locked() is True

    # Regular field should not have metadata
    assert "__metadata__" not in dir(hints["default_field"])


def test_prompt_field_in_inheritance():
    """Test that field type annotations work correctly with inheritance"""

    @dataclass
    class ChildPromptClass(AccessModeTestDataclass):
        child_output: OutputField[str] = "child output"
        child_input: InputField[str] = "child input"
        child_locked: LockedField[str] = "child locked"

    hints = get_type_hints(ChildPromptClass, include_extras=True)

    # Check parent class fields are preserved
    parent_output_meta = hints["output_field"].__metadata__[0]
    assert parent_output_meta.is_output() is True
    assert parent_output_meta.is_input() is False
    assert parent_output_meta.is_locked() is False

    parent_input_meta = hints["input_field"].__metadata__[0]
    assert parent_input_meta.is_input() is True
    assert parent_input_meta.is_output() is False
    assert parent_input_meta.is_locked() is False

    parent_locked_meta = hints["locked_field"].__metadata__[0]
    assert parent_locked_meta.is_locked() is True
    assert parent_locked_meta.is_output() is False
    assert parent_locked_meta.is_input() is False

    # Check child class fields
    child_output_meta = hints["child_output"].__metadata__[0]
    assert child_output_meta.is_output() is True
    assert child_output_meta.is_input() is False
    assert child_output_meta.is_locked() is False

    child_input_meta = hints["child_input"].__metadata__[0]
    assert child_input_meta.is_input() is True
    assert child_input_meta.is_output() is False
    assert child_input_meta.is_locked() is False

    child_locked_meta = hints["child_locked"].__metadata__[0]
    assert child_locked_meta.is_locked() is True
    assert child_locked_meta.is_output() is False
    assert child_locked_meta.is_input() is False


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


def test_access_mode_with_multiple_annotations():
    """Test fields with multiple annotations including our field types"""
    from typing import Annotated

    @dataclass
    class MultiAnnotatedClass(PromptArgument):
        # Field with multiple annotations including OutputField
        output_with_doc: Annotated[OutputField[str], "This is documentation"] = OutputField("test output")
        # Field with multiple annotations including InputField
        input_with_doc: Annotated[InputField[bool], "This is documentation"] = InputField(False)
        # Field with multiple annotations including LockedField
        locked_with_doc: Annotated[LockedField[int], "This is documentation"] = LockedField(42)

    hints = get_type_hints(MultiAnnotatedClass, include_extras=True)

    # Check that we can extract the proper metadata from complex annotations
    assert hints["output_with_doc"].__metadata__[1] == "This is documentation"
    assert hints["input_with_doc"].__metadata__[1] == "This is documentation"
    assert hints["locked_with_doc"].__metadata__[1] == "This is documentation"

    # Create an instance and verify the values are assigned correctly
    instance = MultiAnnotatedClass()
    assert instance.output_with_doc == "test output"
    assert instance.input_with_doc is False
    assert instance.locked_with_doc == 42


def test_field_type_value_assignment():
    """Test direct value assignment to fields with our type annotations"""

    @dataclass
    class TestAssignmentClass(PromptArgument):
        # Test assignment of different types
        string_field: OutputField[str] = "default string"
        int_field: InputField[int] = 42
        bool_field: LockedField[bool] = True
        float_field: OutputField[float] = 3.14

    # Create an instance with default values
    instance1 = TestAssignmentClass()
    assert instance1.string_field == "default string"
    assert instance1.int_field == 42
    assert instance1.bool_field is True
    assert instance1.float_field == 3.14

    # Create an instance with custom values
    instance2 = TestAssignmentClass(string_field="custom string", int_field=99, bool_field=False, float_field=2.71)
    assert instance2.string_field == "custom string"
    assert instance2.int_field == 99
    assert instance2.bool_field is False
    assert instance2.float_field == 2.71


def test_field_type_with_complex_types():
    """Test field types with more complex Python types"""

    @dataclass
    class NestedClass:
        name: str = "nested"
        value: int = 0

    @dataclass
    class ComplexTypesClass(PromptArgument):
        # Test with list, dict and custom class types
        list_field: InputField[list[str]] = field(default_factory=lambda: ["a", "b", "c"])
        dict_field: OutputField[dict[str, int]] = field(default_factory=lambda: {"one": 1, "two": 2})
        nested_field: LockedField[NestedClass] = field(default_factory=NestedClass)
        optional_field: OutputField[Optional[str]] = OutputField(None)

    instance = ComplexTypesClass()
    assert instance.list_field == ["a", "b", "c"]
    assert instance.dict_field == {"one": 1, "two": 2}
    assert instance.nested_field.name == "nested"
    assert instance.nested_field.value == 0
    assert instance.optional_field is None

    # Test assignment of new values
    instance.list_field = ["x", "y", "z"]
    instance.dict_field = {"three": 3, "four": 4}
    instance.nested_field = NestedClass(name="updated", value=42)
    instance.optional_field = "now it has a value"

    assert instance.list_field == ["x", "y", "z"]
    assert instance.dict_field == {"three": 3, "four": 4}
    assert instance.nested_field.name == "updated"
    assert instance.nested_field.value == 42
    assert instance.optional_field == "now it has a value"


def test_field_serialization():
    """Test that fields with our type annotations serialize correctly"""

    @dataclass
    class SerializableClass(PromptArgument):
        input_field: InputField[str] = "input value"
        output_field: OutputField[int] = 42
        locked_field: LockedField[bool] = True

    instance = SerializableClass()
    serialized = instance.to_dict()

    # Verify serialization works
    assert serialized["input_field"] == "input value"
    assert serialized["output_field"] == 42
    assert serialized["locked_field"] is True

    # Deserialize and verify
    deserialized = SerializableClass.from_dict(serialized)
    assert deserialized.input_field == "input value"
    assert deserialized.output_field == 42
    assert deserialized.locked_field is True

    # Test with updated values
    updated = {"input_field": "new input", "output_field": 99, "locked_field": False}
    deserialized2 = SerializableClass.from_dict(updated)
    assert deserialized2.input_field == "new input"
    assert deserialized2.output_field == 99
    assert deserialized2.locked_field is False
