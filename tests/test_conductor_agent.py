from dataclasses import dataclass
from typing import Optional
from unittest.mock import ANY, AsyncMock

import pytest

from diskurs import ImmutableConversation
from diskurs.conductor_agent import ConductorAgent, has_unique_execution_path
from diskurs.entities import ChatMessage, Role, MessageType, LongtermMemory, PromptArgument
from diskurs.prompt import PromptValidationError, DefaultConductorUserPromptArgument
from diskurs.protocols import (
    LLMClient,
    ConversationDispatcher,
    ConductorPrompt,
)

FINALIZER_NAME = "finalizer"


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


def create_conductor_prompt(
    has_finalizer: bool = True, can_finalize_return_value: bool = True, has_can_finalize: bool = True
):
    def finalize(longterm_memory):
        return {"result": "final result"}

    def can_finalize(longterm_memory):
        return can_finalize_return_value

    prompt = Mock(spec=ConductorPrompt)
    prompt.init_longterm_memory.return_value = MyLongTermMemory()
    prompt.create_system_prompt_argument.return_value = Mock()
    prompt.create_user_prompt_argument.return_value = Mock()

    finalize_mock = AsyncMock(side_effect=finalize)
    can_finalize_mock = Mock(side_effect=can_finalize)

    prompt.finalize = finalize_mock
    prompt.can_finalize = can_finalize_mock
    prompt._finalize = finalize if has_finalizer else None
    prompt._can_finalize = can_finalize if has_can_finalize else None
    prompt.fail.return_value = {}
    return prompt


@pytest.fixture
def mock_prompt():
    return create_conductor_prompt()


@pytest.fixture
def mock_prompt_no_finalize():
    return create_conductor_prompt(has_finalizer=False)


@pytest.fixture
def mock_llm_client():
    async def stub_generate_validated_response(conversation, message_type):
        return conversation.append(ChatMessage(role=Role.ASSISTANT, content='{"next_agent": "agent1"}'))

    llm_client = Mock(spec=LLMClient)
    llm_client.generate = AsyncMock(side_effect=stub_generate_validated_response)
    return llm_client


@pytest.fixture
def mock_dispatcher():
    dispatcher = AsyncMock(spec=ConversationDispatcher)
    dispatcher.publish = AsyncMock()
    dispatcher.finalize = AsyncMock()
    return dispatcher


def create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt, finalizer=None, supervisor=None):
    agent = ConductorAgent(
        name="conductor",
        prompt=mock_prompt,
        llm_client=mock_llm_client,
        topics=[],
        agent_descriptions={},
        finalizer_name=finalizer,
        dispatcher=mock_dispatcher,
        max_trials=5,
        max_dispatches=5,
        supervisor=supervisor,
    )
    return agent


@pytest.fixture
def conductor_agent(mock_prompt, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt)


@pytest.fixture
def conductor_agent_with_supervisor(mock_prompt_no_finalize, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt_no_finalize, supervisor="supervisor")


@pytest.fixture
def conductor_agent_with_finalizer_function(mock_prompt, mock_llm_client, mock_dispatcher):

    return create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt)


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


@pytest.mark.asyncio
async def test_process_conversation_updates_longterm_memory(conductor_agent):
    conversation = ImmutableConversation(
        user_prompt_argument=MyUserPromptArgument(field1="value1", field2="value2", field3="value3")
    ).append(ChatMessage(Role.ASSISTANT, content='{"next_agent": "agent1"}'))
    longterm_memory = MyLongTermMemory()

    with patch.object(
        ImmutableConversation, "update_agent_longterm_memory", return_value=conversation
    ) as mock_update_longterm:
        # Instead of making get_agent_longterm_memory an AsyncMock, make it a regular Mock
        conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
        conductor_agent.prompt.init_longterm_memory.return_value = longterm_memory
        conductor_agent.prompt.can_finalize.return_value = False
        conductor_agent.prompt.create_system_prompt_argument.return_value = MyUserPromptArgument(
            field1="sys1", field2="sys2"
        )

        user_prompt_argument = MyUserPromptArgument(field1="value1", field2="value2")
        conductor_agent.prompt.create_user_prompt_argument.return_value = user_prompt_argument

        conductor_agent.generate_validated_response = AsyncMock(return_value=conversation)

        await conductor_agent.process_conversation(conversation)

        assert longterm_memory.field1 == "value1"
        assert longterm_memory.field2 == "value2"
        mock_update_longterm.assert_called_once()


@pytest.mark.asyncio
async def test_process_conversation_finalize(conductor_agent):
    conversation = ImmutableConversation(user_prompt_argument=MyUserPromptArgument())
    longterm_memory = MyLongTermMemory()

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)

    conductor_agent.prompt.can_finalize.return_value = True
    conductor_agent.prompt.finalize.return_value = {"result": "final result"}
    conductor_agent.dispatcher.finalize = Mock()

    conductor_agent.generate_validated_response = Mock()

    await conductor_agent.process_conversation(conversation)

    assert conversation.final_result == {"result": "final result"}
    conductor_agent.generate_validated_response.assert_not_called()


@pytest.fixture
def mock_prompt_cannot_finalize():
    return create_conductor_prompt(can_finalize_return_value=False)


@pytest.fixture
def conductor_cannot_finalize(mock_prompt_cannot_finalize, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt_cannot_finalize)


@pytest.mark.asyncio
async def test_max_dispatches(conductor_cannot_finalize):
    conductor_cannot_finalize.n_dispatches = 49
    conversation = ImmutableConversation(user_prompt_argument=MyUserPromptArgument()).append(
        ChatMessage(Role.ASSISTANT, content='{"next_agent": "agent1"}')
    )
    longterm_memory = MyLongTermMemory()

    conversation = conversation.update_agent_longterm_memory(
        agent_name=conductor_cannot_finalize.name, longterm_memory=longterm_memory
    )

    await conductor_cannot_finalize.process_conversation(conversation)

    conductor_cannot_finalize.prompt.fail.assert_called_once_with(longterm_memory)


@pytest.mark.asyncio
async def test_conductor_agent_valid_next_agent(conductor_cannot_finalize, mock_llm_client):
    conversation = ImmutableConversation()
    llm_response = '{"next_agent": "valid_agent"}'
    assistant_message = ChatMessage(role=Role.ASSISTANT, content=llm_response, type=MessageType.CONDUCTOR)
    conversation = conversation.append(assistant_message)
    mock_llm_client.generate.return_value = conversation

    parsed_prompt_argument = DefaultConductorUserPromptArgument(next_agent="agent1")
    conductor_cannot_finalize.prompt.parse_user_prompt.return_value = parsed_prompt_argument

    await conductor_cannot_finalize.process_conversation(conversation)

    conductor_cannot_finalize.dispatcher.publish.assert_called_once_with(topic="agent1", conversation=ANY)


@pytest.mark.asyncio
async def test_conductor_agent_fail_on_max_dispatches(conductor_cannot_finalize):
    conductor_cannot_finalize.n_dispatches = conductor_cannot_finalize.max_dispatches - 1

    conversation = ImmutableConversation()
    conversation = conversation.append(
        ChatMessage(role=Role.ASSISTANT, content='{"next_agent": "invalid_agent"}', type=MessageType.CONDUCTOR)
    )

    def is_valid(prompt_args):
        raise PromptValidationError(
            f"{prompt_args.next_agent} cannot be routed to from this agent. Valid agents are: {conductor_agent.topics}"
        )

    conductor_cannot_finalize.prompt.is_valid = is_valid

    await conductor_cannot_finalize.process_conversation(conversation)

    conductor_cannot_finalize.prompt.fail.assert_called_once()


@pytest.fixture
def conductor_agent_with_finalizer(mock_prompt_no_finalize, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt_no_finalize, finalizer=FINALIZER_NAME)


@pytest.mark.asyncio
async def test_process_conversation_finalize_with_agent_calls_dispatcher(conductor_agent_with_finalizer):
    conversation = ImmutableConversation(user_prompt_argument=MyUserPromptArgument())
    longterm_memory = MyLongTermMemory()

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)

    conductor_agent_with_finalizer.generate_validated_response = Mock()

    await conductor_agent_with_finalizer.process_conversation(conversation)

    conductor_agent_with_finalizer.dispatcher.publish_final.assert_called_once_with(
        topic=FINALIZER_NAME, conversation=ANY
    )


@pytest.mark.asyncio
async def test_finalize_return_to_supervisor(conductor_agent_with_supervisor, conversation):
    await conductor_agent_with_supervisor.finalize(conversation)

    conductor_agent_with_supervisor.dispatcher.publish.assert_called_once_with(
        topic="supervisor", conversation=conversation
    )


@pytest.mark.asyncio
async def test_finalize_call_finalizer(conductor_agent_with_finalizer, conversation):
    await conductor_agent_with_finalizer.finalize(conversation)

    conductor_agent_with_finalizer.dispatcher.publish_final.assert_called_once_with(
        topic=FINALIZER_NAME, conversation=conversation
    )


@pytest.mark.asyncio
async def test_finalize_call_prompt_function(conductor_agent_with_finalizer_function, conversation):
    await conductor_agent_with_finalizer_function.finalize(conversation)

    assert conversation.final_result == {"result": "final result"}
    conductor_agent_with_finalizer_function.dispatcher.publish.assert_not_called()


import pytest
from unittest.mock import Mock


def test_has_unique_execution_path_prompt_only():
    def dummy_finalize(longterm_memory):
        return {"result": "final result"}

    prompt = Mock()
    setattr(prompt, "_finalize", dummy_finalize)
    has_unique_execution_path(
        prompt=prompt, attr_name="_test_attr", external_options=[None, None], error_message="Test error message"
    )


def test_has_unique_execution_path_agent_only():
    prompt = Mock()
    setattr(prompt, "_finalize", None)
    an_agent_name = "agent1"
    has_unique_execution_path(
        prompt=prompt,
        attr_name="_finalize",
        external_options=[an_agent_name, None],
        error_message="Test error message",
    )


def test_has_unique_execution_path_invalid_func_and_external():
    def dummy_finalize(longterm_memory):
        return {"result": "final result"}

    prompt = Mock()
    setattr(prompt, "_finalize", dummy_finalize)
    an_agent_name = "agent1"
    with pytest.raises(AssertionError, match="Test error message"):
        has_unique_execution_path(
            prompt=prompt,
            attr_name="_finalize",
            external_options=[an_agent_name, None],
            error_message="Test error message",
        )


def test_has_unique_execution_path_invalid_external():
    prompt = Mock()
    setattr(prompt, "_finalize", None)
    an_agent_name = "agent1"
    a_supervisor_name = "supervisor"
    with pytest.raises(AssertionError, match="Test error message"):
        has_unique_execution_path(
            prompt=prompt,
            attr_name="_finalize",
            external_options=[an_agent_name, a_supervisor_name],
            error_message="Test error message",
        )


@pytest.mark.asyncio
async def test_process_conversation_calls_can_finalize(conversation, conductor_agent):
    conductor_agent.finalize = AsyncMock()

    await conductor_agent.process_conversation(conversation)
    conductor_agent.finalize.assert_called_once()


@dataclass
class DefaultCanFinalizeUserPromptArgument(PromptArgument):
    can_finalize: Optional[bool] = None


@pytest.mark.asyncio
async def test_can_finalize(conversation, conductor_agent):
    conductor_agent.can_finalize_name = "Finalizer_Agent"
    conductor_agent.prompt.can_finalize.return_value = True
    conductor_agent.finalize = AsyncMock()

    parsed_prompt_argument = DefaultCanFinalizeUserPromptArgument(can_finalize=True)
    conductor_agent.prompt.parse_user_prompt.return_value = parsed_prompt_argument

    await conductor_agent.process_conversation(conversation)
    conductor_agent.finalize.assert_called_once()
