from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest

from diskurs import ImmutableConversation
from diskurs.entities import DiskursInput, ChatMessage, Role, MessageType, RoutingRule
from diskurs.forum import Forum, ForumFactory
from diskurs.protocols import ConversationStore, Conversation

# Apply asyncio mark to specific async tests rather than using global pytestmark
# This avoids conflicts with non-async tests
# pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


@pytest.fixture
def mock_store():
    """Create a mock store following the pattern from other test modules"""
    store = AsyncMock(spec=ConversationStore)
    store.exists = AsyncMock(return_value=True)
    store.fetch = AsyncMock()
    store.persist = AsyncMock()
    return store


@pytest.fixture
def forum(mock_store):
    """Create a forum instance with ImmutableConversation"""
    return Forum(
        agents=[],
        dispatcher=Mock(),
        tool_executor=Mock(),
        first_contact=Mock(),
        conversation_store=mock_store,
        conversation_class=ImmutableConversation,
    )


@pytest.mark.asyncio
async def test_fetch_existing_conversation(forum, mock_store):
    """Test fetching an existing conversation with valid conversation_id"""
    input_data = DiskursInput(conversation_id="test-id", user_query="Hello")
    existing_conversation = ImmutableConversation(conversation_id="test-id", conversation_store=mock_store)
    mock_store.fetch.return_value = existing_conversation

    conversation = await forum.fetch_or_create_conversation(input_data)

    assert conversation.conversation_id == "test-id"
    mock_store.exists.assert_awaited_once_with("test-id")
    mock_store.fetch.assert_awaited_once_with("test-id")


@pytest.mark.asyncio
async def test_create_new_persistent_conversation(forum, mock_store):
    """Test creating a new conversation with conversation_id"""
    input_data = DiskursInput(conversation_id="new-id", user_query="Hello")
    mock_store.exists.return_value = False

    conversation = await forum.fetch_or_create_conversation(input_data)

    assert isinstance(conversation, ImmutableConversation)
    assert conversation.conversation_store == mock_store
    assert conversation.conversation_id == "new-id"
    mock_store.exists.assert_awaited_once_with("new-id")


@pytest.mark.asyncio
async def test_create_ephemeral_conversation(forum, mock_store):
    """Test creating an ephemeral conversation without conversation_id"""
    input_data = DiskursInput(user_query="Hello")

    conversation = await forum.fetch_or_create_conversation(input_data)

    assert isinstance(conversation, ImmutableConversation)
    assert conversation.conversation_store is None
    assert conversation.conversation_id == ""
    mock_store.persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_persist_with_store_and_id(mock_store):
    """Test maybe_persist with both store and conversation_id"""
    conversation = ImmutableConversation(conversation_id="test-id", conversation_store=mock_store)

    await conversation.maybe_persist()

    mock_store.persist.assert_awaited_once_with(conversation)


@pytest.mark.asyncio
async def test_maybe_persist_without_store():
    """Test maybe_persist without store doesn't raise"""
    conversation = ImmutableConversation(conversation_id="test-id")

    await conversation.maybe_persist()


@pytest.mark.asyncio
async def test_maybe_persist_without_id(mock_store):
    """Test maybe_persist without conversation_id doesn't persist"""
    conversation = ImmutableConversation(conversation_store=mock_store)

    await conversation.maybe_persist()

    mock_store.persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_append_message_persists_conversation(forum, mock_store):
    """Test that appending a message triggers persistence"""
    # Setup
    input_data = DiskursInput(conversation_id="test-id", user_query="Hello")
    existing_conversation = ImmutableConversation(conversation_id="test-id", conversation_store=mock_store)
    mock_store.fetch.return_value = existing_conversation
    mock_store.exists.return_value = True

    conversation = await forum.fetch_or_create_conversation(input_data)
    message = ChatMessage(role=Role.USER, content="Hello", name="forum", type=MessageType.CONVERSATION)
    new_conversation = conversation.append(message)
    await new_conversation.maybe_persist()

    assert len(new_conversation.chat) > len(conversation.chat)
    assert new_conversation.chat[-1] == message
    mock_store.persist.assert_awaited_once_with(new_conversation)


@dataclass
class MockPrompt:
    """Mock dataclass to avoid asdict() issues"""

    type: str = "test_prompt"
    location: Path = Path("test/path")
    agent_description: str = "Test agent"


@pytest.fixture
def test_files_path():
    """Return the path to the test_files directory"""
    current_dir = Path(__file__).parent
    return current_dir / "test_files"


@pytest.fixture
def mock_prompt_registry():
    with patch("diskurs.forum.PROMPT_REGISTRY") as mock_registry:
        mock_prompt_cls = MagicMock()
        mock_prompt = MockPrompt()
        mock_prompt_cls.create.return_value = mock_prompt
        mock_registry.get.return_value = mock_prompt_cls
        yield mock_registry, mock_prompt


def test_create_agents_loads_rules(test_files_path, mock_prompt_registry):
    """Test that ForumFactory.create_agents correctly loads and processes rules"""
    config_path = Path("test_files") / "config.yaml"
    forum_factory = ForumFactory(config_path=config_path, base_path=Path(__file__).parent)

    # Mock entities on forum factory
    forum_factory.llm_clients = {"test-llm": MagicMock()}
    forum_factory.dispatcher = MagicMock()
    forum_factory.modules_to_import = [
        Path(__file__).parent.parent / mdls
        for mdls in [
            "llm_client.py",
            "azure_llm_client.py",
            "dispatcher.py",
            "agent.py",
            "conductor_agent.py",
            "prompt.py",
            "immutable_conversation.py",
            "filesystem_conversation_store.py",
            "heuristic_agent.py",
        ]
    ]

    forum_factory.create_agents()

    conductor_agent = [agent for agent in forum_factory.agents if agent.name == "Test_Conductor_Agent"][0]

    assert isinstance(conductor_agent.rules, list)
    assert all(isinstance(rule, RoutingRule) for rule in conductor_agent.rules)
    assert len(conductor_agent.rules) == 3

    # Find the rules by name
    true_rule = next(rule for rule in conductor_agent.rules if rule.name == "test_rule_true")
    false_rule = next(rule for rule in conductor_agent.rules if rule.name == "test_rule_false")
    metadata_rule = next(rule for rule in conductor_agent.rules if rule.name == "metadata_check_rule")

    # Check that rules point to correct agent to dispatch to
    assert true_rule.target_agent == "Target_Agent_One"
    assert false_rule.target_agent == "Target_Agent_Two"
    assert metadata_rule.target_agent == "Metadata_Agent"

    test_conversation = MagicMock(spec=Conversation)

    assert true_rule.condition(test_conversation) is True
    assert false_rule.condition(test_conversation) is False

    assert conductor_agent.fallback_to_llm is True
