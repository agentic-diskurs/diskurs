from dataclasses import dataclass, field
from typing import List, Dict, get_type_hints

from diskurs.entities import (
    LongtermMemory,
    JsonSerializable,
    PerTurnField,
)


@dataclass
class TestLongtermMemoryWithPerTurnFields(LongtermMemory):
    """Test class with various field types including PerTurnField."""

    # Normal fields
    persistent_str: str = "persistent string"
    persistent_int: int = 42

    # Per-turn fields that should reset
    per_turn_str: PerTurnField[str] = "per-turn string"
    per_turn_int: PerTurnField[int] = 99
    per_turn_bool: PerTurnField[bool] = True
    per_turn_list: PerTurnField[List[str]] = field(default_factory=lambda: ["one", "two", "three"])
    per_turn_dict: PerTurnField[Dict[str, int]] = field(default_factory=lambda: {"a": 1, "b": 2})


class TestPerTurnField:
    """Tests for the PerTurnField functionality."""

    def test_per_turn_field_metadata(self):
        """Test that PerTurnField adds the correct metadata."""
        # Get field metadata using typing.get_type_hints() with include_extras=True
        hints = get_type_hints(TestLongtermMemoryWithPerTurnFields, include_extras=True)

        # Check per-turn field metadata
        per_turn_str_meta = hints["per_turn_str"].__metadata__[0]
        assert per_turn_str_meta.is_per_turn() is True
        assert per_turn_str_meta.is_input() is False
        assert per_turn_str_meta.is_output() is False
        assert per_turn_str_meta.is_locked() is False

    def test_reset_per_turn_fields_method(self):
        """Test that reset_per_turn_fields() properly resets only per-turn fields."""
        # Create a memory instance with modified values
        memory = TestLongtermMemoryWithPerTurnFields(
            persistent_str="modified persistent",
            persistent_int=100,
            per_turn_str="modified per-turn",
            per_turn_int=200,
            per_turn_bool=False,
            per_turn_list=["modified", "list"],
            per_turn_dict={"modified": 999},
        )

        # Reset per-turn fields
        reset_memory = memory.reset_per_turn_fields()

        # Verify that only per-turn fields were reset to defaults
        # Per-turn fields should be reset
        assert reset_memory.per_turn_str == "per-turn string"
        assert reset_memory.per_turn_int == 99
        assert reset_memory.per_turn_bool is True
        assert reset_memory.per_turn_list == ["one", "two", "three"]
        assert reset_memory.per_turn_dict == {"a": 1, "b": 2}

        # Other fields should remain modified
        assert reset_memory.persistent_str == "modified persistent"
        assert reset_memory.persistent_int == 100

        # Original instance should remain unchanged (immutability)
        assert memory.per_turn_str == "modified per-turn"

    def test_from_dict_without_reset(self):
        """Test that from_dict without reset_per_turn preserves all field values."""
        # Create a dictionary with modified values
        data = {
            "persistent_str": "serialized persistent",
            "persistent_int": 300,
            "per_turn_str": "serialized per-turn",
            "per_turn_int": 400,
            "per_turn_bool": False,
            "per_turn_list": ["serialized", "list"],
            "per_turn_dict": {"serialized": 888},
            "user_query": "serialized query",
        }

        # Deserialize without resetting per-turn fields
        memory = TestLongtermMemoryWithPerTurnFields.from_dict(data, reset_per_turn=False)

        # Verify that all fields match the serialized values
        assert memory.persistent_str == "serialized persistent"
        assert memory.persistent_int == 300
        assert memory.per_turn_str == "serialized per-turn"
        assert memory.per_turn_int == 400
        assert memory.per_turn_bool is False
        assert memory.per_turn_list == ["serialized", "list"]
        assert memory.per_turn_dict == {"serialized": 888}
        assert memory.user_query == "serialized query"

    def test_from_dict_with_reset(self):
        """Test that from_dict with reset_per_turn resets per-turn fields to defaults."""
        # Create a dictionary with modified values
        data = {
            "persistent_str": "serialized persistent",
            "persistent_int": 300,
            "per_turn_str": "serialized per-turn",
            "per_turn_int": 400,
            "per_turn_bool": False,
            "per_turn_list": ["serialized", "list"],
            "per_turn_dict": {"serialized": 888},
            "user_query": "serialized query",
        }

        # Deserialize with resetting per-turn fields
        memory = TestLongtermMemoryWithPerTurnFields.from_dict(data, reset_per_turn=True)

        # Verify that per-turn fields are reset to defaults
        assert memory.per_turn_str == "per-turn string"
        assert memory.per_turn_int == 99
        assert memory.per_turn_bool is True
        assert memory.per_turn_list == ["one", "two", "three"]
        assert memory.per_turn_dict == {"a": 1, "b": 2}

        # Verify that other fields match the serialized values
        assert memory.persistent_str == "serialized persistent"
        assert memory.persistent_int == 300
        assert memory.user_query == "serialized query"

    def test_from_dict_with_missing_fields(self):
        """Test that from_dict with reset_per_turn works correctly with missing fields."""
        # Create a dictionary with only some fields
        data = {
            "persistent_str": "serialized persistent",
            "per_turn_str": "serialized per-turn",
            "user_query": "serialized query",
        }

        # Deserialize with resetting per-turn fields
        memory = TestLongtermMemoryWithPerTurnFields.from_dict(data, reset_per_turn=True)

        # Verify that present non-per-turn fields are set from input
        assert memory.persistent_str == "serialized persistent"
        assert memory.user_query == "serialized query"

        # Verify that per-turn fields are reset to defaults, even if present in input
        assert memory.per_turn_str == "per-turn string"

        # Verify that missing fields have default values
        assert memory.persistent_int == 42

    def test_inheritance_with_per_turn_fields(self):
        """Test that per-turn fields work correctly with inheritance."""

        @dataclass
        class ChildMemory(TestLongtermMemoryWithPerTurnFields):
            # Add new fields
            child_persistent: str = "child persistent"
            child_per_turn: PerTurnField[str] = "child per-turn"

        # Create a dictionary with modified values
        data = {
            "persistent_str": "parent persistent",
            "per_turn_str": "parent per-turn",
            "child_persistent": "modified child",
            "child_per_turn": "modified child per-turn",
        }

        # Deserialize with resetting per-turn fields
        memory = ChildMemory.from_dict(data, reset_per_turn=True)

        # Verify that all per-turn fields (parent and child) are reset
        assert memory.per_turn_str == "per-turn string"  # Parent per-turn field
        assert memory.child_per_turn == "child per-turn"  # Child per-turn field

        # Verify that non-per-turn fields are preserved
        assert memory.persistent_str == "parent persistent"
        assert memory.child_persistent == "modified child"

    def test_to_dict_serialization_of_per_turn_fields(self):
        """Test that to_dict correctly serializes per-turn fields."""
        # Create a memory instance
        memory = TestLongtermMemoryWithPerTurnFields(
            per_turn_str="modified per-turn",
            per_turn_int=200,
        )

        # Serialize to dictionary
        data = memory.to_dict()

        # Verify that per-turn fields are serialized with current values
        assert data["per_turn_str"] == "modified per-turn"
        assert data["per_turn_int"] == 200

        # Reset the per-turn fields
        reset_memory = memory.reset_per_turn_fields()

        # Serialize the reset memory
        reset_data = reset_memory.to_dict()

        # Verify that per-turn fields are serialized with reset values
        assert reset_data["per_turn_str"] == "per-turn string"
        assert reset_data["per_turn_int"] == 99

    def test_complex_per_turn_field_types(self):
        """Test that complex data types work correctly with per-turn fields."""

        @dataclass
        class NestedClass(JsonSerializable):
            name: str = "default"
            value: int = 0

        @dataclass
        class ComplexMemory(LongtermMemory):
            # Complex per-turn fields
            nested_object: PerTurnField[NestedClass] = field(default_factory=NestedClass)
            nested_list: PerTurnField[List[NestedClass]] = field(
                default_factory=lambda: [NestedClass(name="item1"), NestedClass(name="item2")]
            )

        # Create a memory with modified complex values
        memory = ComplexMemory(
            nested_object=NestedClass(name="modified", value=42),
            nested_list=[
                NestedClass(name="modified1", value=1),
                NestedClass(name="modified2", value=2),
                NestedClass(name="modified3", value=3),
            ],
        )

        # Serialize to dictionary
        data = memory.to_dict()

        # Verify serialization
        assert data["nested_object"]["name"] == "modified"
        assert len(data["nested_list"]) == 3

        # Deserialize with reset
        reset_memory = ComplexMemory.from_dict(data, reset_per_turn=True)

        # Verify reset of complex objects
        assert reset_memory.nested_object.name == "default"
        assert reset_memory.nested_object.value == 0
        assert len(reset_memory.nested_list) == 2
        assert reset_memory.nested_list[0].name == "item1"

    def test_integration_with_conversation_store(self):
        """Test integration with a conversation store that uses from_dict with reset_per_turn."""
        from diskurs.entities import JsonSerializable

        @dataclass
        class SimpleConversationStore(JsonSerializable):
            """Minimal implementation of conversation store for testing."""

            @classmethod
            def deserialize_longterm_memory(cls, data, reset_per_turn=True):
                """Simulate conversation store loading longterm memory."""
                return {
                    agent_name: TestLongtermMemoryWithPerTurnFields.from_dict(
                        memory_data, reset_per_turn=reset_per_turn
                    )
                    for agent_name, memory_data in data.items()
                }

        # Create data that would be stored in a conversation store
        stored_data = {
            "agent1": {
                "persistent_str": "agent1 persistent",
                "per_turn_str": "agent1 per-turn",
                "user_query": "What is the capital of France?",
            },
            "agent2": {
                "persistent_str": "agent2 persistent",
                "per_turn_str": "agent2 per-turn",
                "user_query": "What is the capital of France?",
            },
        }

        # Simulate loading conversation without resetting
        longterm_memory_without_reset = SimpleConversationStore.deserialize_longterm_memory(
            stored_data, reset_per_turn=False
        )

        # Verify per-turn fields are preserved when reset is off
        assert longterm_memory_without_reset["agent1"].per_turn_str == "agent1 per-turn"
        assert longterm_memory_without_reset["agent2"].per_turn_str == "agent2 per-turn"

        # Simulate loading conversation with resetting
        longterm_memory_with_reset = SimpleConversationStore.deserialize_longterm_memory(
            stored_data, reset_per_turn=True
        )

        # Verify per-turn fields are reset to defaults
        assert longterm_memory_with_reset["agent1"].per_turn_str == "per-turn string"
        assert longterm_memory_with_reset["agent2"].per_turn_str == "per-turn string"

        # Verify non-per-turn fields are preserved
        assert longterm_memory_with_reset["agent1"].persistent_str == "agent1 persistent"
        assert longterm_memory_with_reset["agent2"].persistent_str == "agent2 persistent"
        assert longterm_memory_with_reset["agent1"].user_query == "What is the capital of France?"

    def test_practical_use_case(self):
        """Test a practical use case with conversation state fields."""

        @dataclass
        class ConversationMemory(LongtermMemory):
            # Persistent context
            conversation_history: str = ""
            user_preferences: Dict[str, str] = field(default_factory=dict)

            # Per-turn state that resets each turn
            user_query: PerTurnField[str] = ""
            intermediate_answer: PerTurnField[str] = ""
            answer: PerTurnField[str] = ""
            current_reasoning: PerTurnField[str] = ""

        # Create a memory with values from a previous turn
        memory = ConversationMemory(
            conversation_history="User asked about Python. AI explained basics.",
            user_preferences={"language": "English", "detail_level": "high"},
            user_query="How do I use decorators in Python?",
            intermediate_answer="Decorators are a form of metaprogramming...",
            answer="Decorators allow you to modify functions and methods...",
            current_reasoning="I should explain decorators with simple examples.",
        )

        # Serialize the memory (as would happen when storing conversation)
        data = memory.to_dict()

        # Deserialize with reset (as would happen when loading for a new turn)
        new_turn_memory = ConversationMemory.from_dict(data, reset_per_turn=True)

        # Verify persistent context is maintained
        assert "Python" in new_turn_memory.conversation_history
        assert new_turn_memory.user_preferences["language"] == "English"

        # Verify per-turn state is reset
        assert new_turn_memory.user_query == ""
        assert new_turn_memory.intermediate_answer == ""
        assert new_turn_memory.answer == ""
        assert new_turn_memory.current_reasoning == ""
