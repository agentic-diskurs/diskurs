from pathlib import Path
from unittest.mock import Mock

import pytest

from diskurs import ImmutableConversation, ConductorAgent
from diskurs.filesystem_conversation_store import FilesystemConversationStore


def setup_agent(agent, longterm_memory, system_pargs, user_pargs, agent_name):
    agent.name = agent_name
    prompt = Mock()
    prompt.system_prompt_argument = system_pargs
    prompt.user_prompt_argument = user_pargs
    prompt.longterm_memory = longterm_memory
    agent.prompt = prompt


@pytest.fixture
def conversation_store(prompt_arguments, longterm_memories, conversation):
    user_pargs, system_pargs = prompt_arguments
    ltm1, ltm2 = longterm_memories

    directory = Path(__file__).parent / "store_test_files"
    directory.mkdir(parents=True, exist_ok=True)

    agents = [Mock(spec=ConductorAgent) for _ in range(2)]
    setup_agent(agents[0], ltm1, system_pargs, user_pargs, agent_name="my_conductor")
    setup_agent(agents[1], ltm2, system_pargs, user_pargs, agent_name="my_conductor_2")

    store = FilesystemConversationStore(directory=directory, agents=agents, conversation_class=ImmutableConversation)

    yield store

    # cleanup
    for file in directory.glob("*.json"):
        file.unlink()
    directory.rmdir()


def test_persist_no_id_raises_error(conversation_store):
    with pytest.raises(AssertionError):
        conversation = Mock()
        conversation.conversation_id = ""
        conversation_store.persist(conversation)


def test_persist_file_created(conversation_store, conversation):
    conversation_store.persist(conversation)
    assert conversation_store.exists(conversation.conversation_id)


def test_fetch(conversation_store, conversation):
    conversation_store.persist(conversation)
    fetched_conversation = conversation_store.fetch(conversation.conversation_id)

    assert fetched_conversation.active_agent == conversation.active_agent
    assert fetched_conversation.chat == conversation.chat


def test_delete_conversation(conversation_store, conversation):
    conversation_store.persist(conversation)
    assert conversation_store.exists(conversation.conversation_id)

    conversation_store.delete(conversation.conversation_id)
    assert not conversation_store.exists(conversation.conversation_id)


def test_delete_nonexistent_conversation(conversation_store):
    # Try deleting a conversation ID that doesn't exist
    non_existent_id = "nonexistent_convo"
    conversation_store.delete(non_existent_id)  # Should not raise an error
    assert not conversation_store.exists(non_existent_id)


def test_fetch_nonexistent_conversation(conversation_store):
    with pytest.raises(FileNotFoundError):
        conversation_store.fetch("nonexistent_convo")


def test_persist_and_fetch_content_integrity(conversation_store, conversation):
    conversation_store.persist(conversation)
    fetched_conversation = conversation_store.fetch(conversation.conversation_id)

    assert (
        fetched_conversation.to_dict() == conversation.to_dict()
    ), "Persisted and fetched conversations do not match."


def test_exists_for_nonexistent_conversation(conversation_store):
    assert not conversation_store.exists("nonexistent_convo")
