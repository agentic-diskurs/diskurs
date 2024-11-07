from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock, ANY

import pytest

from agents.Conductor_Agent.prompt import ConductorUserPromptArgument
from diskurs import ImmutableConversation
from diskurs.prompt import PromptValidationError
from diskurs.conductor_agent import ConductorAgent
from diskurs.entities import ChatMessage, Role, MessageType, LongtermMemory, PromptArgument
from diskurs.protocols import (
    LLMClient,
    ConversationDispatcher,
    ConductorPrompt,
)


@dataclass
class MyLongTermMemory(LongtermMemory):
    user_query: Optional[str] = ""
    field1: Optional[str] = ""
    field2: Optional[str] = ""
    field3: Optional[str] = ""


@dataclass
class MyUserPromptArgument(PromptArgument):
    field1: Optional[str] = ""
    field2: Optional[str] = ""
    field3: Optional[str] = ""


@pytest.fixture
def mock_prompt():
    prompt = Mock(spec=ConductorPrompt)
    prompt.init_longterm_memory.return_value = MyLongTermMemory()
    prompt.create_system_prompt_argument.return_value = Mock()
    prompt.create_user_prompt_argument.return_value = Mock()
    prompt.can_finalize.return_value = False
    prompt.finalize.return_value = {}
    prompt.fail.return_value = {}
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
        max_dispatches=5,
    )
    return agent


def test_update_longterm_memory(conductor_agent):
    conversation = ImmutableConversation(
        user_prompt_argument=MyUserPromptArgument(field1="value1", field2="value2", field3="value3")
    )
    longterm_memory = MyLongTermMemory()

    conversation.update_agent_longterm_memory = Mock(return_value=conversation)
    conductor_agent.prompt.init_longterm_memory.return_value = longterm_memory

    updated_conversation = conductor_agent.create_or_update_longterm_memory(conversation=conversation)

    assert longterm_memory.field1 == "value1"
    assert longterm_memory.field2 == "value2"
    assert longterm_memory.field3 == "value3"
    conversation.update_agent_longterm_memory.assert_called_once_with(
        agent_name=conductor_agent.name, longterm_memory=longterm_memory
    )


def test_update_longterm_memory_existing_fields(conductor_agent):
    conversation = ImmutableConversation(
        user_prompt_argument=MyUserPromptArgument(field1="value1", field2="value2", field3="value3")
    )
    longterm_memory = MyLongTermMemory(field1="existing_value1")

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
    conversation.update_agent_longterm_memory = Mock(return_value=conversation)
    conductor_agent.prompt.init_longterm_memory = Mock()

    updated_conversation = conductor_agent.create_or_update_longterm_memory(conversation, overwrite=False)

    assert longterm_memory.field1 == "existing_value1"  # Should not be overwritten
    assert longterm_memory.field2 == "value2"
    conversation.update_agent_longterm_memory.assert_called_once_with(
        agent_name=conductor_agent.name, longterm_memory=longterm_memory
    )


def test_update_longterm_memory_with_overwrite(conductor_agent):
    conversation = ImmutableConversation(user_prompt_argument=MyUserPromptArgument(field1="new_value1"))
    longterm_memory = MyLongTermMemory(field1="existing_value1")

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
    conversation.update_agent_longterm_memory = Mock(return_value=conversation)
    conductor_agent.prompt.init_longterm_memory = Mock()

    updated_conversation = conductor_agent.create_or_update_longterm_memory(conversation, overwrite=True)

    assert longterm_memory.field1 == "new_value1"  # Should be overwritten
    conversation.update_agent_longterm_memory.assert_called_once_with(
        agent_name=conductor_agent.name, longterm_memory=longterm_memory
    )


from unittest.mock import patch


def test_process_conversation_updates_longterm_memory(conductor_agent):
    conversation = ImmutableConversation(
        user_prompt_argument=MyUserPromptArgument(field1="value1", field2="value2", field3="value3")
    ).append(ChatMessage(Role.ASSISTANT, content='{"next_agent": "agent1"}'))
    longterm_memory = MyLongTermMemory()

    with patch.object(
        ImmutableConversation, "update_agent_longterm_memory", return_value=conversation
    ) as mock_update_longterm:
        conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
        conductor_agent.prompt.init_longterm_memory.return_value = longterm_memory
        conductor_agent.prompt.can_finalize.return_value = False
        conductor_agent.prompt.create_system_prompt_argument.return_value = Mock()

        user_prompt_argument = MyUserPromptArgument(field1="value1", field2="value2")
        conductor_agent.prompt.create_user_prompt_argument.return_value = user_prompt_argument

        conductor_agent.generate_validated_response = Mock(return_value=conversation)

        conductor_agent.process_conversation(conversation)

        assert longterm_memory.field1 == "value1"
        assert longterm_memory.field2 == "value2"
        mock_update_longterm.assert_called_once()


def test_process_conversation_finalize(conductor_agent):
    conversation = ImmutableConversation(user_prompt_argument=MyUserPromptArgument())
    longterm_memory = MyLongTermMemory()

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)

    conductor_agent.prompt.can_finalize.return_value = True
    conductor_agent.prompt.finalize.return_value = {"result": "final result"}
    conductor_agent.dispatcher.finalize = Mock()

    conductor_agent.generate_validated_response = Mock()

    conductor_agent.process_conversation(conversation)

    conductor_agent.prompt.finalize.assert_called_once_with(longterm_memory)
    conductor_agent.dispatcher.finalize.assert_called_once_with(response={"result": "final result"})
    conductor_agent.generate_validated_response.assert_not_called()


def test_max_dispatches(conductor_agent):
    conductor_agent.n_dispatches = 49
    conversation = ImmutableConversation(user_prompt_argument=MyUserPromptArgument()).append(
        ChatMessage(Role.ASSISTANT, content='{"next_agent": "agent1"}')
    )
    longterm_memory = MyLongTermMemory()

    conversation = conversation.update_agent_longterm_memory(
        agent_name=conductor_agent.name, longterm_memory=longterm_memory
    )

    conductor_agent.process_conversation(conversation)

    conductor_agent.prompt.fail.assert_called_once_with(longterm_memory)


def test_conductor_agent_valid_next_agent(conductor_agent, mock_llm_client):
    conversation = ImmutableConversation()
    llm_response = '{"next_agent": "valid_agent"}'
    assistant_message = ChatMessage(role=Role.ASSISTANT, content=llm_response, type=MessageType.ROUTING)
    conversation = conversation.append(assistant_message)
    mock_llm_client.generate.return_value = conversation

    parsed_prompt_argument = ConductorUserPromptArgument(next_agent="valid_agent")
    conductor_agent.prompt.parse_user_prompt.return_value = parsed_prompt_argument

    conductor_agent.process_conversation(conversation)

    conductor_agent.dispatcher.publish.assert_called_once_with(topic="valid_agent", conversation=ANY)


def test_conductor_agent_fail_on_max_dispatches(conductor_agent):
    conductor_agent.n_dispatches = conductor_agent.max_dispatches - 1

    conversation = ImmutableConversation()
    conversation = conversation.append(
        ChatMessage(role=Role.ASSISTANT, content='{"next_agent": "invalid_agent"}', type=MessageType.ROUTING)
    )

    def is_valid(prompt_args):
        raise PromptValidationError(
            f"{prompt_args.next_agent} cannot be routed to from this agent. Valid agents are: {conductor_agent.topics}"
        )

    conductor_agent.prompt.is_valid = is_valid

    conductor_agent.process_conversation(conversation)

    conductor_agent.dispatcher.finalize.assert_called_once()
    conductor_agent.prompt.fail.assert_called_once()
