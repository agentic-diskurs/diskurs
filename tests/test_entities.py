from dataclasses import dataclass, field
from typing import Any, get_type_hints, Optional

import pytest

from diskurs.entities import (
    LongtermMemory,
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
from .conftest import EnumPromptArgument, ChatType, Priority


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
        output_with_doc: Annotated[OutputField[str], "This is documentation"] = "test output"
        # Field with multiple annotations including InputField
        input_with_doc: Annotated[InputField[bool], "This is documentation"] = False
        # Field with multiple annotations including LockedField
        locked_with_doc: Annotated[LockedField[int], "This is documentation"] = 42

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
        optional_field: OutputField[Optional[str]] = None

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


@dataclass
class TestPromptArgument(PromptArgument):
    input_field: InputField[str] = "default_input"
    output_field: OutputField[str] = "default_output"
    locked_field: LockedField[str] = "default_locked"
    regular_field: str = "default_regular"


@dataclass
class TestLongtermMemory(LongtermMemory):
    input_field: str = "ltm_input"
    output_field: str = "ltm_output"
    locked_field: str = "ltm_locked"
    regular_field: str = "ltm_regular"
    user_query: str = "What is the meaning of life?"


class TestPromptFieldMethods:
    """Tests for the new methods added to PromptArgument and LongtermMemory"""

    def test_prompt_argument_init_from_longterm_memory(self):
        """Test initializing a PromptArgument from a LongtermMemory"""
        # Create a LongtermMemory instance with values
        ltm = TestLongtermMemory(
            input_field="ltm_input_value",
            output_field="ltm_output_value",
            locked_field="ltm_locked_value",
            regular_field="ltm_regular_value",
        )

        # Create a fresh PromptArgument
        prompt_arg = TestPromptArgument()

        # Initialize from LongtermMemory
        result = prompt_arg.init(ltm)

        # Verify that only InputField values were copied
        assert result.input_field == "ltm_input_value"  # Should be copied (InputField)
        assert result.output_field == "default_output"  # Should not be copied (OutputField)
        assert result.locked_field == "default_locked"  # Should not be copied (LockedField)
        assert result.regular_field == "default_regular"  # Should not be copied (regular field)

    def test_prompt_argument_init_from_prompt_argument(self):
        """Test initializing a PromptArgument from another PromptArgument"""
        # Create a source PromptArgument with values
        source = TestPromptArgument(
            input_field="source_input",
            output_field="source_output",
            locked_field="source_locked",
            regular_field="source_regular",
        )

        # Create a target PromptArgument with default values
        target = TestPromptArgument()

        # Initialize from source PromptArgument
        result = target.init(source)

        # Verify that only InputField values were copied
        assert result.input_field == "source_input"  # Should be copied (InputField)
        assert result.output_field == "default_output"  # Should not be copied (OutputField)
        assert result.locked_field == "default_locked"  # Should not be copied (LockedField)
        assert result.regular_field == "default_regular"  # Should not be copied (regular field)

    def test_prompt_argument_init_with_null_source(self):
        """Test initializing a PromptArgument with a null source"""
        prompt_arg = TestPromptArgument()
        result = prompt_arg.init(None)

        # Should return self unchanged
        assert result == prompt_arg

    def test_prompt_argument_update(self):
        """Test updating a PromptArgument with values from a dictionary"""
        # Create a PromptArgument with default values
        prompt_arg = TestPromptArgument()

        other = TestPromptArgument(
            input_field="new_input",
            output_field="new_output",
            locked_field="new_locked",
            regular_field="new_regular",
        )

        # Update with the TestPromptArgument instance
        result = prompt_arg.update(other)

        # Verify that only non-InputField and non-LockedField values were updated
        assert result.input_field == "default_input"  # Should not be updated (InputField)
        assert result.output_field == "new_output"  # Should be updated (OutputField)
        assert result.locked_field == "default_locked"  # Should not be updated (LockedField)
        assert result.regular_field == "new_regular"  # Should be updated (regular field)

    def test_longterm_memory_update(self):
        """Test updating a LongtermMemory with values from a PromptArgument"""
        # Create a PromptArgument with values
        prompt_arg = TestPromptArgument(
            input_field="prompt_input",
            output_field="prompt_output",
            locked_field="prompt_locked",
            regular_field="prompt_regular",
        )

        # Create a LongtermMemory with default values
        ltm = TestLongtermMemory()

        # Update LongtermMemory with PromptArgument
        result = ltm.update(prompt_arg)

        # Verify that only OutputField values were copied
        assert result.input_field == "ltm_input"  # Should not be updated (not OutputField in source)
        assert result.output_field == "prompt_output"  # Should be updated (OutputField in source)
        assert result.locked_field == "ltm_locked"  # Should not be updated (not OutputField in source)
        assert result.regular_field == "ltm_regular"  # Should not be updated (not OutputField in source)

    def test_longterm_memory_update_with_null_prompt_arg(self):
        """Test updating a LongtermMemory with a null PromptArgument"""
        ltm = TestLongtermMemory()
        result = ltm.update(None)

        # Should return self unchanged
        assert result == ltm

    def test_longterm_memory_update_with_different_fields(self):
        """Test updating a LongtermMemory with a PromptArgument that has different fields"""

        @dataclass
        class DifferentPromptArgument(PromptArgument):
            different_output_field: OutputField[str] = "different_output"
            common_output_field: OutputField[str] = "common_output"

        @dataclass
        class TestLongtermMemory2(LongtermMemory):
            common_output_field: str = "ltm_common"
            other_field: str = "ltm_other"
            user_query: str = "What is the meaning of life?"

        prompt_arg = DifferentPromptArgument()
        ltm = TestLongtermMemory2()

        result = ltm.update(prompt_arg)

        # Only common fields with OutputField type should be updated
        assert result.common_output_field == "common_output"  # Should be updated (common field, OutputField in source)
        assert result.other_field == "ltm_other"  # Should not be updated (not present in source)

    def test_integration_with_immutable_conversation(self):
        """Test integration of new methods with ImmutableConversation's existing functionality"""
        # Create test instances
        prompt_arg = TestPromptArgument(
            input_field="prompt_input",
            output_field="prompt_output",
            locked_field="prompt_locked",
            regular_field="prompt_regular",
        )

        ltm = TestLongtermMemory(
            input_field="ltm_input", output_field="ltm_output", locked_field="ltm_locked", regular_field="ltm_regular"
        )

        # Create a conversation and add the longterm memory
        conversation = ImmutableConversation()
        conversation = conversation.update_agent_longterm_memory(agent_name="test_agent", longterm_memory=ltm)
        conversation = conversation.update(prompt_argument=prompt_arg)

        # Test update pattern that would be used in conductor_agent.py
        # This simulates the create_or_update_longterm_memory method
        agent_ltm = conversation.get_agent_longterm_memory("test_agent")
        updated_ltm = agent_ltm.update(conversation.prompt_argument)
        conversation = conversation.update_agent_longterm_memory(agent_name="test_agent", longterm_memory=updated_ltm)

        # Verify that OutputField values from prompt_arg were copied to longterm memory
        updated_agent_ltm = conversation.get_agent_longterm_memory("test_agent")
        assert updated_agent_ltm.output_field == "prompt_output"  # Should be updated (OutputField in source)
        assert updated_agent_ltm.input_field == "ltm_input"  # Should not be updated

        # Test init pattern that would be used in multistep_agent.py
        new_prompt_arg = TestPromptArgument()
        initialized_prompt_arg = new_prompt_arg.init(conversation.get_agent_longterm_memory("test_agent"))
        conversation = conversation.update(prompt_argument=initialized_prompt_arg)

        # Verify that InputField values from longterm_memory were copied to prompt argument
        assert conversation.prompt_argument.input_field == "ltm_input"  # Should be copied (InputField)
        assert conversation.prompt_argument.output_field == "default_output"  # Should not be copied
        assert conversation.prompt_argument.locked_field == "default_locked"  # Should not be copied


class TestGetOutputFields:
    """Tests specifically for the get_output_fields method of PromptArgument class"""

    def test_get_output_fields_basic(self):
        """Test that get_output_fields correctly identifies OutputField and regular fields"""

        @dataclass
        class TestOutputFieldsArgument(PromptArgument):
            input_field: InputField[str] = "input value"
            output_field: OutputField[str] = "output value"
            locked_field: LockedField[str] = "locked value"
            regular_field: str = "regular value"

        arg = TestOutputFieldsArgument()
        output_fields = arg.get_output_fields()

        # Should include OutputField and regular field, but not InputField or LockedField
        assert "output_field" in output_fields
        assert "regular_field" in output_fields
        assert "input_field" not in output_fields
        assert "locked_field" not in output_fields
        assert output_fields["output_field"] == "output value"
        assert output_fields["regular_field"] == "regular value"

    def test_get_output_fields_inheritance(self):
        """Test that get_output_fields works correctly with inheritance"""

        @dataclass
        class BaseArgument(PromptArgument):
            base_input: InputField[str] = "base input"
            base_output: OutputField[str] = "base output"
            base_locked: LockedField[str] = "base locked"
            base_regular: str = "base regular"

        @dataclass
        class ChildArgument(BaseArgument):
            child_input: InputField[str] = "child input"
            child_output: OutputField[str] = "child output"
            child_locked: LockedField[str] = "child locked"
            child_regular: str = "child regular"

        arg = ChildArgument()
        output_fields = arg.get_output_fields()

        # Should include OutputField and regular field from both parent and child classes
        assert "base_output" in output_fields
        assert "base_regular" in output_fields
        assert "child_output" in output_fields
        assert "child_regular" in output_fields

        # Should exclude InputField and LockedField from both parent and child classes
        assert "base_input" not in output_fields
        assert "base_locked" not in output_fields
        assert "child_input" not in output_fields
        assert "child_locked" not in output_fields

    def test_get_output_fields_complex_types(self):
        """Test that get_output_fields handles complex field types correctly"""

        @dataclass
        class NestedClass:
            value: str = "nested value"

        @dataclass
        class ComplexArgument(PromptArgument):
            output_list: OutputField[list[str]] = field(default_factory=lambda: ["one", "two", "three"])
            output_dict: OutputField[dict[str, int]] = field(default_factory=lambda: {"a": 1, "b": 2})
            output_nested: OutputField[NestedClass] = field(default_factory=NestedClass)
            input_complex: InputField[list[dict[str, str]]] = field(default_factory=lambda: [{"key": "value"}])
            locked_complex: LockedField[dict[str, list[int]]] = field(default_factory=lambda: {"numbers": [1, 2, 3]})
            regular_complex: dict[str, Any] = field(default_factory=lambda: {"regular": "value"})

        arg = ComplexArgument()
        output_fields = arg.get_output_fields()

        # Should include OutputField and regular field with complex types
        assert "output_list" in output_fields
        assert "output_dict" in output_fields
        assert "output_nested" in output_fields
        assert "regular_complex" in output_fields

        # Should exclude InputField and LockedField with complex types
        assert "input_complex" not in output_fields
        assert "locked_complex" not in output_fields

        # Verify values are correctly preserved
        assert output_fields["output_list"] == ["one", "two", "three"]
        assert output_fields["output_dict"] == {"a": 1, "b": 2}
        assert isinstance(output_fields["output_nested"], NestedClass)
        assert output_fields["regular_complex"] == {"regular": "value"}

    def test_get_output_fields_empty_and_none(self):
        """Test that get_output_fields correctly handles empty and None values"""

        @dataclass
        class EmptyArgument(PromptArgument):
            output_none: OutputField[Optional[str]] = None
            output_empty_list: OutputField[list] = field(default_factory=list)
            output_empty_dict: OutputField[dict] = field(default_factory=dict)
            regular_none: Optional[str] = None

        arg = EmptyArgument()
        output_fields = arg.get_output_fields()

        # Should include all fields
        assert "output_none" in output_fields
        assert "output_empty_list" in output_fields
        assert "output_empty_dict" in output_fields
        assert "regular_none" in output_fields

        # Verify values are correctly preserved
        assert output_fields["output_none"] is None
        assert output_fields["output_empty_list"] == []
        assert output_fields["output_empty_dict"] == {}
        assert output_fields["regular_none"] is None

    def test_get_output_fields_integration_with_prompt(self):
        """Test integration of get_output_fields with the prompt rendering logic"""
        from diskurs.prompt import BasePrompt
        from jinja2 import Template

        @dataclass
        class TestIntegrationArgument(PromptArgument):
            input_field: InputField[str] = "input value"
            output_field: OutputField[str] = "output value"
            locked_field: LockedField[str] = "locked value"
            regular_field: str = "regular value"

        # Create a simple BasePrompt for testing
        system_template = Template("System: {{ regular_field }}")
        user_template = Template("User: {{ regular_field }}")
        json_template = Template("JSON Schema: {{ schema | tojson }}")

        prompt = BasePrompt(
            agent_description="Test Agent",
            system_template=system_template,
            user_template=user_template,
            prompt_argument_class=TestIntegrationArgument,
            json_formatting_template=json_template,
            return_json=True,
        )

        arg = TestIntegrationArgument()
        message = prompt.render_system_template("test", arg)

        # The system template should contain the JSON schema with only output and regular fields
        assert "output_field" in message.content
        assert "regular_field" in message.content
        assert "input_field" not in message.content
        assert "locked_field" not in message.content
