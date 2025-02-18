from unittest.mock import AsyncMock, Mock

import pytest

from diskurs.entities import DiskursInput, ChatMessage, Role, MessageType
from diskurs.forum import Forum
from diskurs.protocols import ConversationStore
from diskurs import ImmutableConversation

pytestmark = pytest.mark.asyncio


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


async def test_fetch_existing_conversation(forum, mock_store):
    """Test fetching an existing conversation with valid conversation_id"""
    input_data = DiskursInput(conversation_id="test-id", user_query="Hello")
    existing_conversation = ImmutableConversation(conversation_id="test-id", conversation_store=mock_store)
    mock_store.fetch.return_value = existing_conversation

    conversation = await forum.fetch_or_create_conversation(input_data)

    assert conversation.conversation_id == "test-id"
    mock_store.exists.assert_awaited_once_with("test-id")
    mock_store.fetch.assert_awaited_once_with("test-id")


async def test_create_new_persistent_conversation(forum, mock_store):
    """Test creating a new conversation with conversation_id"""
    input_data = DiskursInput(conversation_id="new-id", user_query="Hello")
    mock_store.exists.return_value = False

    conversation = await forum.fetch_or_create_conversation(input_data)

    assert isinstance(conversation, ImmutableConversation)
    assert conversation.conversation_store == mock_store
    assert conversation.conversation_id == "new-id"
    mock_store.exists.assert_awaited_once_with("new-id")


async def test_create_ephemeral_conversation(forum, mock_store):
    """Test creating an ephemeral conversation without conversation_id"""
    input_data = DiskursInput(user_query="Hello")

    conversation = await forum.fetch_or_create_conversation(input_data)

    assert isinstance(conversation, ImmutableConversation)
    assert conversation.conversation_store is None
    assert conversation.conversation_id == ""
    mock_store.persist.assert_not_awaited()


async def test_maybe_persist_with_store_and_id(mock_store):
    """Test maybe_persist with both store and conversation_id"""
    conversation = ImmutableConversation(conversation_id="test-id", conversation_store=mock_store)

    await conversation.maybe_persist()

    mock_store.persist.assert_awaited_once_with(conversation)


async def test_maybe_persist_without_store():
    """Test maybe_persist without store doesn't raise"""
    conversation = ImmutableConversation(conversation_id="test-id")

    await conversation.maybe_persist()


async def test_maybe_persist_without_id(mock_store):
    """Test maybe_persist without conversation_id doesn't persist"""
    conversation = ImmutableConversation(conversation_store=mock_store)

    await conversation.maybe_persist()

    mock_store.persist.assert_not_awaited()


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
