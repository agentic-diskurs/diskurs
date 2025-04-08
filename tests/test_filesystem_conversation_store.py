from unittest.mock import Mock

import pytest
import pytest_asyncio

from diskurs import ImmutableConversation, ConductorAgent
from diskurs.filesystem_conversation_store import AsyncFilesystemConversationStore


def setup_agent(agent, longterm_memory, pargs, agent_name):
    agent.name = agent_name
    prompt = Mock()
    prompt.prompt_argument = pargs
    prompt.longterm_memory = longterm_memory
    agent.prompt = prompt


@pytest_asyncio.fixture
async def conversation_store(tmp_path, prompt_arguments, longterm_memories, conversation):
    ltm1, ltm2 = longterm_memories

    directory = tmp_path / "store_test_files"
    directory.mkdir(parents=True, exist_ok=True)

    agents = [Mock(spec=ConductorAgent) for _ in range(2)]
    setup_agent(agents[0], ltm1, prompt_arguments, agent_name="my_conductor")
    setup_agent(agents[1], ltm2, prompt_arguments, agent_name="my_conductor_2")

    store = AsyncFilesystemConversationStore.create(
        base_path=directory, agents=agents, conversation_class=ImmutableConversation, is_persistent=True
    )

    yield store

    # Updated cleanup to handle nested directory structure
    conversations_dir = directory / "conversations"
    if conversations_dir.exists():
        for file in conversations_dir.glob("*.json"):
            file.unlink()
        conversations_dir.rmdir()
    directory.rmdir()


@pytest.mark.asyncio
async def test_persist_no_id_raises_error(conversation_store):
    with pytest.raises(AssertionError):
        conversation = Mock()
        conversation.conversation_id = ""
        await conversation_store.persist(conversation)


@pytest.mark.asyncio
async def test_persist_file_created(conversation_store, conversation):
    await conversation_store.persist(conversation)
    assert await conversation_store.exists(conversation.conversation_id)


@pytest.mark.asyncio
async def test_fetch(conversation_store, conversation):
    await conversation_store.persist(conversation)
    fetched_conversation = await conversation_store.fetch(conversation.conversation_id)

    assert fetched_conversation.active_agent == conversation.active_agent
    assert fetched_conversation.chat == conversation.chat


@pytest.mark.asyncio
async def test_delete_conversation(conversation_store, conversation):
    await conversation_store.persist(conversation)
    assert await conversation_store.exists(conversation.conversation_id)

    await conversation_store.delete(conversation.conversation_id)
    assert not await conversation_store.exists(conversation.conversation_id)


@pytest.mark.asyncio
async def test_delete_nonexistent_conversation(conversation_store):
    # Try deleting a conversation ID that doesn't exist
    non_existent_id = "nonexistent_convo"
    await conversation_store.delete(non_existent_id)  # Should not raise an error
    assert not await conversation_store.exists(non_existent_id)


@pytest.mark.asyncio
async def test_fetch_nonexistent_conversation(conversation_store):
    with pytest.raises(FileNotFoundError):
        await conversation_store.fetch("nonexistent_convo")


@pytest.mark.asyncio
async def test_persist_and_fetch_content_integrity(conversation_store, conversation):
    await conversation_store.persist(conversation)
    fetched_conversation = await conversation_store.fetch(conversation.conversation_id)

    assert (
        fetched_conversation.to_dict() == conversation.to_dict()
    ), "Persisted and fetched conversations do not match."


@pytest.mark.asyncio
async def test_exists_for_nonexistent_conversation(conversation_store):
    assert not await conversation_store.exists("nonexistent_convo")
