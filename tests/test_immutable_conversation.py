import enum
import json
from dataclasses import dataclass

import pytest

from conftest import EnumPromptArgument, EnumLongtermMemory, ChatType, Priority
from diskurs import ImmutableConversation
from diskurs.entities import Role, ChatMessage, MessageType, PromptArgument


# Add this enum at module level so it can be imported during deserialization
class Status(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


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
    assert new_conversation.prompt_argument.field1 == conversation.prompt_argument.field1
    assert new_conversation.prompt_argument.field2 == conversation.prompt_argument.field2
    assert new_conversation.prompt_argument.field3 == conversation.prompt_argument.field3
    assert new_conversation.user_prompt == conversation.user_prompt
    assert new_conversation.system_prompt == conversation.system_prompt
    assert new_conversation.active_agent == conversation.active_agent


def test_conversation_from_dict_with_heuristic_agent(heuristic_agent_mock, conversation_dict):
    """Test that from_dict works correctly with an agent that has no system_prompt_argument."""
    # Modify conversation_dict to not include system_prompt_argument
    conversation_dict["active_agent"] = "heuristic_agent"
    conversation_dict["system_prompt_argument"] = None

    new_conversation = ImmutableConversation.from_dict(data=conversation_dict, agents=[heuristic_agent_mock])

    assert new_conversation.prompt_argument is None
    assert new_conversation.active_agent == "heuristic_agent"


def test_conversation_from_dict_missing_system_prompt_argument(conductor_mock):
    """Test that from_dict handles missing system_prompt_argument in data dict."""
    conversation_dict = {
        "system_prompt": None,
        "user_prompt": None,
        "prompt_argument": None,
        "chat": [],
        "longterm_memory": {},
        "metadata": {},
        "active_agent": "my_conductor",
        "conversation_id": "",
    }

    new_conversation = ImmutableConversation.from_dict(data=conversation_dict, agents=[conductor_mock])

    assert new_conversation.prompt_argument is None


def test_conversation_from_dict_none_values(conductor_mock):
    """Test that from_dict properly handles None values for all optional fields."""
    conversation_dict = {
        "system_prompt": None,
        "user_prompt": None,
        "system_prompt_argument": None,
        "prompt_argument": None,
        "chat": [],
        "longterm_memory": {},
        "metadata": {},
        "active_agent": "my_conductor",
    }

    new_conversation = ImmutableConversation.from_dict(data=conversation_dict, agents=[conductor_mock])

    assert new_conversation.system_prompt is None
    assert new_conversation.user_prompt is None
    assert new_conversation.prompt_argument is None
    assert new_conversation.chat == []
    assert new_conversation.metadata == {}


class TestImmutableConversationWithEnums:
    """Tests for enum handling in ImmutableConversation"""

    @pytest.fixture
    def enum_conversation(self):
        """Create a test conversation with enum values"""
        # Create conversation with enum-based prompt arguments and longterm memory
        ltm = EnumLongtermMemory(user_query="Test query", preferred_chat_type=ChatType.GROUP)
        prompt_arg = EnumPromptArgument(
            chat_type=ChatType.CHANNEL, priority=Priority.HIGH, message_type=MessageType.CONDUCTOR
        )

        conversation = ImmutableConversation(
            system_prompt=ChatMessage(role=Role.SYSTEM, content="System prompt"),
            user_prompt=ChatMessage(role=Role.USER, content="User message"),
            prompt_argument=prompt_arg,
            active_agent="test_agent",
        )

        # Add longterm memory
        conversation = conversation.update_agent_longterm_memory(agent_name="test_conductor", longterm_memory=ltm)

        return conversation

    def test_conversation_serialization_with_enums(self, enum_conversation):
        """Test serializing a conversation with enum values"""
        data = enum_conversation.to_dict()

        # Check enum values are serialized correctly
        assert data["prompt_argument"]["chat_type"] == "channel"
        assert data["prompt_argument"]["priority"] == 2
        assert data["prompt_argument"]["message_type"] == "conductor"
        assert data["longterm_memory"]["test_conductor"]["preferred_chat_type"] == "group"

    def test_conversation_round_trip(self, enum_conversation, monkeypatch):
        """Test round-trip serialization and deserialization"""
        data = enum_conversation.to_dict()

        # Create a mock agent class for from_dict
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.prompt = type(
                    "Prompt",
                    (),
                    {
                        "prompt_argument": EnumPromptArgument,
                        "system_prompt_argument": None,
                        "longterm_memory": EnumLongtermMemory,
                    },
                )

        # Create mock agent instances
        mock_agents = [MockAgent("test_agent"), MockAgent("test_conductor")]

        # Deserialize conversation
        restored = ImmutableConversation.from_dict(data=data, agents=mock_agents)

        # Verify enum values are correctly deserialized
        assert restored.prompt_argument.chat_type == ChatType.CHANNEL
        assert restored.prompt_argument.priority == Priority.HIGH
        assert restored.prompt_argument.message_type == MessageType.CONDUCTOR
        assert restored._longterm_memory["test_conductor"].preferred_chat_type == ChatType.GROUP

    def test_special_enum_values(self):
        """Test enums with special characters or values"""

        class SpecialEnum(enum.Enum):
            WITH_SPACE = "value with space"
            WITH_SPECIAL = "value-with.special:chars"
            EMPTY = ""

        @dataclass
        class SpecialPrompt(PromptArgument):
            special: SpecialEnum = SpecialEnum.WITH_SPACE

        # Test serialization
        prompt = SpecialPrompt(special=SpecialEnum.WITH_SPECIAL)
        data = prompt.to_dict()
        assert data["special"] == "value-with.special:chars"

        # Test deserialization
        restored = SpecialPrompt.from_dict(data)
        assert restored.special == SpecialEnum.WITH_SPECIAL

        # Test empty string
        prompt = SpecialPrompt(special=SpecialEnum.EMPTY)
        data = prompt.to_dict()
        assert data["special"] == ""

        restored = SpecialPrompt.from_dict(data)
        assert restored.special == SpecialEnum.EMPTY

    def test_metadata_with_enums(self):
        """Test storing enum values in metadata and serializing/deserializing them"""
        from tests.test_files.test_entities import ChatType

        # Create a conversation with enum in metadata
        conversation = ImmutableConversation(
            system_prompt=ChatMessage(role=Role.SYSTEM, content="System prompt"),
            user_prompt=ChatMessage(role=Role.USER, content="User message"),
            metadata={"chat_type": ChatType.CHAT, "regular_value": "normal string"},
            active_agent="test_agent",
        )

        # Serialize the conversation
        data = conversation.to_dict()

        assert json.dumps(data), "Metadata with enums should be JSON serializable"

        # Check that the enum is correctly serialized
        assert isinstance(data["metadata"]["chat_type"], dict)
        assert data["metadata"]["chat_type"]["__enum__"] is True
        assert data["metadata"]["chat_type"]["module"] == "tests.test_files.test_entities"
        assert data["metadata"]["chat_type"]["class"] == "ChatType"
        assert data["metadata"]["chat_type"]["value"] == "CHAT"

        # Check that regular values remain untouched
        assert data["metadata"]["regular_value"] == "normal string"

        # Create a mock agent for deserialization
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.prompt = type(
                    "Prompt",
                    (),
                    {
                        "prompt_argument": None,
                        "system_prompt_argument": None,
                        "longterm_memory": None,
                    },
                )

        mock_agent = MockAgent("test_agent")

        # Deserialize the conversation
        restored = ImmutableConversation.from_dict(data=data, agents=[mock_agent])

        # Check that the enum was correctly deserialized
        assert restored.metadata["chat_type"] == ChatType.CHAT
        assert restored.metadata["regular_value"] == "normal string"

    def test_metadata_with_multiple_enums(self):
        """Test storing multiple different enum values in metadata"""
        from tests.test_files.test_entities import ChatType

        # Create a conversation with multiple enums in metadata
        conversation = ImmutableConversation(
            system_prompt=ChatMessage(role=Role.SYSTEM, content="System prompt"),
            user_prompt=ChatMessage(role=Role.USER, content="User message"),
            metadata={
                "chat_type": ChatType.KEYWORD,
                "status": Status.PENDING,
                "message_type": MessageType.CONVERSATION,
            },
            active_agent="test_agent",
        )

        # Serialize the conversation
        data = conversation.to_dict()

        assert json.dumps(data), "Metadata with enums should be JSON serializable"

        # Check that all enums are correctly serialized
        assert data["metadata"]["chat_type"]["value"] == "KEYWORD"
        assert data["metadata"]["status"]["class"] == "Status"
        assert data["metadata"]["status"]["value"] == "PENDING"
        assert data["metadata"]["message_type"]["value"] == "CONVERSATION"

        # Create a mock agent for deserialization
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.prompt = type(
                    "Prompt",
                    (),
                    {
                        "prompt_argument": None,
                        "system_prompt_argument": None,
                        "longterm_memory": None,
                    },
                )

        mock_agent = MockAgent("test_agent")

        # Deserialize the conversation
        restored = ImmutableConversation.from_dict(data=data, agents=[mock_agent])

        # Check that the enums were correctly deserialized
        assert restored.metadata["chat_type"] == ChatType.KEYWORD
        assert restored.metadata["status"] == Status.PENDING
        assert restored.metadata["message_type"] == MessageType.CONVERSATION

    def test_metadata_with_enum_error_handling(self):
        """Test error handling when deserializing enums in metadata"""
        # Create a conversation dict with a non-existent enum class
        conversation_dict = {
            "system_prompt": None,
            "user_prompt": None,
            "system_prompt_argument": None,
            "prompt_argument": None,
            "chat": [],
            "longterm_memory": {},
            "metadata": {
                "non_existent_enum": {
                    "__enum__": True,
                    "module": "non.existent.module",
                    "class": "NonExistentEnum",
                    "value": "SOME_VALUE",
                }
            },
            "active_agent": "test_agent",
        }

        # Create a mock agent for deserialization
        class MockAgent:
            def __init__(self, name):
                self.name = name
                self.prompt = type(
                    "Prompt",
                    (),
                    {
                        "prompt_argument": None,
                        "system_prompt_argument": None,
                        "longterm_memory": None,
                    },
                )

        mock_agent = MockAgent("test_agent")

        # Deserialize the conversation - this should not raise an exception
        restored = ImmutableConversation.from_dict(data=conversation_dict, agents=[mock_agent])

        # The original serialized form should be preserved
        assert restored.metadata["non_existent_enum"] == conversation_dict["metadata"]["non_existent_enum"]
