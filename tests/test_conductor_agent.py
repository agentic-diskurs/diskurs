import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Optional

from diskurs.entities import Conversation, ChatMessage, Role
from diskurs import ConductorAgent, PromptArgument
from diskurs.protocols import LLMClient, ConversationDispatcher, ConductorPromptProtocol


@dataclass
class LongTermMemory:
    field1: Optional[str] = None
    field2: Optional[str] = None
    field3: Optional[str] = None


@dataclass
class UserPromptArgument(PromptArgument):
    field1: Optional[str] = ""
    field2: Optional[str] = ""
    field3: Optional[str] = ""


@pytest.fixture
def mock_prompt():
    prompt = Mock(spec=ConductorPromptProtocol)
    prompt.init_longterm_memory.return_value = LongTermMemory()
    prompt.create_system_prompt_argument.return_value = Mock()
    prompt.create_user_prompt_argument.return_value = Mock()
    prompt.can_finalize.return_value = False
    prompt.finalize.return_value = {}
    return prompt


@pytest.fixture
def mock_llm_client():
    llm_client = Mock(spec=LLMClient)
    llm_client.generate.return_value = ChatMessage(role=Role.ASSISTANT, content='{"next_agent": "agent1"}')
    return llm_client


@pytest.fixture
def mock_dispatcher():
    dispatcher = Mock(spec=ConversationDispatcher)
    dispatcher.publish = Mock()
    dispatcher.finalize = Mock()
    return dispatcher


@pytest.fixture
def conductor_agent(mock_prompt, mock_llm_client, mock_dispatcher):
    agent = ConductorAgent(
        name="conductor",
        prompt=mock_prompt,
        llm_client=mock_llm_client,
        topics=[],
        agent_descriptions={},
        finalizer_name="finalizer",
        dispatcher=mock_dispatcher,
        max_trials=5,
    )
    return agent


def test_update_longterm_memory(conductor_agent):
    conversation = Conversation(
        user_prompt_argument=UserPromptArgument(field1="value1", field2="value2", field3="value3")
    )
    longterm_memory = LongTermMemory()

    conversation.update_agent_longterm_memory = Mock(return_value=conversation)
    conductor_agent.prompt.init_longterm_memory.return_value = longterm_memory

    updated_conversation = conductor_agent.update_longterm_memory(conversation)

    assert longterm_memory.field1 == "value1"
    assert longterm_memory.field2 == "value2"
    assert longterm_memory.field3 == "value3"
    conversation.update_agent_longterm_memory.assert_called_once_with(
        agent_name=conductor_agent.name, longterm_memory=longterm_memory
    )


def test_update_longterm_memory_existing_fields(conductor_agent):
    conversation = Conversation(
        user_prompt_argument=UserPromptArgument(field1="value1", field2="value2", field3="value3")
    )
    longterm_memory = LongTermMemory(field1="existing_value1")

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
    conversation.update_agent_longterm_memory = Mock(return_value=conversation)
    conductor_agent.prompt.init_longterm_memory = Mock()

    updated_conversation = conductor_agent.update_longterm_memory(conversation, overwrite=False)

    assert longterm_memory.field1 == "existing_value1"  # Should not be overwritten
    assert longterm_memory.field2 == "value2"
    conversation.update_agent_longterm_memory.assert_called_once_with(
        agent_name=conductor_agent.name, longterm_memory=longterm_memory
    )


def test_update_longterm_memory_with_overwrite(conductor_agent):
    conversation = Conversation(user_prompt_argument=UserPromptArgument(field1="new_value1"))
    longterm_memory = LongTermMemory(field1="existing_value1")

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
    conversation.update_agent_longterm_memory = Mock(return_value=conversation)
    conductor_agent.prompt.init_longterm_memory = Mock()

    updated_conversation = conductor_agent.update_longterm_memory(conversation, overwrite=True)

    assert longterm_memory.field1 == "new_value1"  # Should be overwritten
    conversation.update_agent_longterm_memory.assert_called_once_with(
        agent_name=conductor_agent.name, longterm_memory=longterm_memory
    )


from unittest.mock import patch


def test_process_conversation_updates_longterm_memory(conductor_agent):
    conversation = Conversation().append(ChatMessage(Role.ASSISTANT, content='{"next_agent": "agent1"}'))
    longterm_memory = LongTermMemory()

    with patch.object(Conversation, "update_agent_longterm_memory", return_value=conversation) as mock_update_longterm:
        conversation.get_agent_longterm_memory = Mock(return_value=None)
        conductor_agent.prompt.init_longterm_memory.return_value = longterm_memory
        conductor_agent.prompt.can_finalize.return_value = False
        conductor_agent.prompt.create_system_prompt_argument.return_value = Mock()

        user_prompt_argument = UserPromptArgument(field1="value1", field2="value2")
        conductor_agent.prompt.create_user_prompt_argument.return_value = user_prompt_argument

        conductor_agent.generate_validated_response = Mock(return_value=conversation)

        conductor_agent.process_conversation(conversation)

        assert longterm_memory.field1 == "value1"
        assert longterm_memory.field2 == "value2"
        mock_update_longterm.assert_called_once()

        conductor_agent.dispatcher.publish.assert_called_once_with(topic="agent1", conversation=conversation)


def test_process_conversation_finalize(conductor_agent):
    conversation = Conversation()
    longterm_memory = LongTermMemory()

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
    conductor_agent.prompt.can_finalize.return_value = True
    conductor_agent.prompt.finalize.return_value = {"result": "final result"}
    conductor_agent.dispatcher.finalize = Mock()

    conductor_agent.generate_validated_response = Mock()

    conductor_agent.process_conversation(conversation)

    conductor_agent.prompt.finalize.assert_called_once_with(longterm_memory)
    conductor_agent.dispatcher.finalize.assert_called_once_with(response={"result": "final result"})
    conductor_agent.generate_validated_response.assert_not_called()
