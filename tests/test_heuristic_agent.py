from unittest.mock import MagicMock, AsyncMock

import pytest

from conftest import MyExtendedUserPromptArgument
from diskurs import Conversation, PromptArgument, ToolExecutor, ImmutableConversation
from diskurs.entities import MessageType, ChatMessage, Role
from diskurs.heuristic_agent import HeuristicAgent, HeuristicAgentFinalizer
from diskurs.prompt import HeuristicPrompt

CONDUCTOR_NAME = "my_conductor"


def test_create_method():
    name = "test_agent"
    prompt = MagicMock()
    topics = ["topic1", "topic2"]
    dispatcher = MagicMock()
    tool_executor = MagicMock()
    render_prompt = True

    agent = HeuristicAgent.create(
        name=name,
        prompt=prompt,
        topics=topics,
        dispatcher=dispatcher,
        tool_executor=tool_executor,
        render_prompt=render_prompt,
    )

    assert agent.name == name
    assert agent.prompt == prompt
    assert agent.topics == topics
    assert agent.dispatcher == dispatcher
    assert agent.tool_executor == tool_executor
    assert agent.render_prompt == render_prompt


def test_get_conductor_name():
    name = "test_agent"
    prompt = MagicMock()
    topics = ["conductor1", "topic2"]

    agent = HeuristicAgent(name=name, prompt=prompt, topics=topics)

    assert agent.get_conductor_name() == "conductor1"


def test_register_dispatcher():
    name = "test_agent"
    prompt = MagicMock()
    dispatcher = MagicMock()

    agent = HeuristicAgent(name=name, prompt=prompt)
    agent.register_dispatcher(dispatcher)

    assert agent.dispatcher == dispatcher


def test_prepare_conversation():
    name = "test_agent"
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    user_prompt_argument = MagicMock(spec=PromptArgument)
    updated_conversation = MagicMock()

    conversation.update.return_value = updated_conversation

    agent = HeuristicAgent(name=name, prompt=prompt)
    result = agent.prepare_conversation(conversation, user_prompt_argument)

    conversation.update.assert_called_once_with(user_prompt_argument=user_prompt_argument, active_agent=name)
    assert result == updated_conversation


@pytest.mark.asyncio
async def test_invoke():
    name = "test_agent"
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    updated_conversation = MagicMock(spec=Conversation)
    updated_conversation_with_memory = MagicMock(spec=Conversation)
    heuristic_sequence_conversation = MagicMock(spec=Conversation)
    final_conversation = MagicMock(spec=Conversation)
    tool_executor = MagicMock()
    prompt.create_user_prompt_argument.return_value = MagicMock()

    # Make heuristic_sequence an AsyncMock
    prompt.heuristic_sequence = AsyncMock(return_value=heuristic_sequence_conversation)

    agent = HeuristicAgent(
        name=name,
        prompt=prompt,
        tool_executor=tool_executor,
        topics=["conductor_name"],
        init_prompt_arguments_with_longterm_memory=True,
        render_prompt=True,
    )

    agent.prepare_conversation = MagicMock(return_value=updated_conversation)
    updated_conversation.update_prompt_argument_with_longterm_memory.return_value = updated_conversation_with_memory
    prompt.render_user_template.return_value = "rendered_template"
    heuristic_sequence_conversation.append.return_value = final_conversation

    result = await agent.invoke(conversation)

    agent.prepare_conversation.assert_called_once_with(
        conversation=conversation, user_prompt_argument=prompt.create_user_prompt_argument.return_value
    )
    updated_conversation.update_prompt_argument_with_longterm_memory.assert_called_once_with(
        conductor_name=agent.get_conductor_name()
    )
    prompt.render_user_template.assert_called_once_with(
        agent.name,
        prompt_args=heuristic_sequence_conversation.user_prompt_argument,
        message_type=MessageType.CONVERSATION,
    )
    heuristic_sequence_conversation.append.assert_called_once_with(name=agent.name, message="rendered_template")
    assert result == final_conversation


@pytest.mark.asyncio
async def test_invoke_no_executor():
    name = "test_agent"
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    updated_conversation = MagicMock(spec=Conversation)
    updated_conversation_with_memory = MagicMock(spec=Conversation)
    conversation_with_previous = MagicMock(spec=Conversation)  # Add this
    heuristic_sequence_conversation = MagicMock(spec=Conversation)
    final_conversation = MagicMock(spec=Conversation)
    prompt.create_user_prompt_argument.return_value = MagicMock()

    prompt.heuristic_sequence = AsyncMock(return_value=heuristic_sequence_conversation)

    agent = HeuristicAgent(
        name=name,
        prompt=prompt,
        topics=["conductor_name"],
        init_prompt_arguments_with_longterm_memory=True,
        render_prompt=True,
    )

    agent.prepare_conversation = MagicMock(return_value=updated_conversation)
    updated_conversation.update_prompt_argument_with_longterm_memory.return_value = updated_conversation_with_memory
    # Add this line to mock the additional method call
    updated_conversation_with_memory.update_prompt_argument_with_previous_agent.return_value = (
        conversation_with_previous
    )
    prompt.render_user_template.return_value = "rendered_template"
    heuristic_sequence_conversation.append.return_value = final_conversation

    result = await agent.invoke(conversation)

    # Update assertion to use conversation_with_previous
    prompt.heuristic_sequence.assert_awaited_once_with(conversation_with_previous, call_tool=None)


@pytest.mark.asyncio
async def test_process_conversation():
    name = "test_agent"
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    dispatcher = MagicMock()
    updated_conversation = MagicMock(spec=Conversation)

    agent = HeuristicAgent(name=name, prompt=prompt, dispatcher=dispatcher)

    agent.invoke = AsyncMock(return_value=updated_conversation)
    agent.get_conductor_name = MagicMock(return_value="conductor_name")

    dispatcher.publish = AsyncMock()

    await agent.process_conversation(conversation)

    agent.invoke.assert_awaited_once_with(conversation)
    agent.get_conductor_name.assert_called_once()
    dispatcher.publish.assert_awaited_once_with(topic="conductor_name", conversation=updated_conversation)


def create_prompt(user_prompt_argument):
    prompt = AsyncMock(spec=HeuristicPrompt)
    prompt.create_user_prompt_argument.return_value = user_prompt_argument
    prompt.render_user_template.return_value = ChatMessage(
        role=Role.USER,
        name="my_multistep",
        content="rendered template",
        type=MessageType.CONVERSATION,
    )
    prompt.user_prompt_argument = user_prompt_argument

    return prompt


@pytest.fixture
def mock_heuristic_prompt():
    return create_prompt(MyExtendedUserPromptArgument())


def create_heuristic_agent(
    mock_heuristic_prompt,
):
    async def async_identity(conversation, call_tool):
        return conversation

    agent = HeuristicAgent.create(
        name="test_agent",
        prompt=mock_heuristic_prompt,
        topics=[CONDUCTOR_NAME],
        tool_executor=AsyncMock(spec=ToolExecutor),
        render_prompt=True,
    )
    agent.prompt.heuristic_sequence = async_identity
    return agent


@pytest.fixture
def heuristic_agent(mock_heuristic_prompt):
    return create_heuristic_agent(mock_heuristic_prompt)


@pytest.mark.asyncio
async def test_invoke_with_longterm_memory_and_previous_agent(heuristic_agent, extended_conversation):
    extended_conversation = extended_conversation.append(
        message=ChatMessage(
            content="I am a multistep agent message",
            role=Role.USER,
            type=MessageType.CONVERSATION,
        )
    )
    result = await heuristic_agent.invoke(extended_conversation)
    assert all(
        [
            result.user_prompt_argument.field1 == "extended user prompt field 1",
            result.user_prompt_argument.field2 == "longterm val 2",
            result.user_prompt_argument.field3 == "user prompt field 3",
            result.user_prompt_argument.field4 == "user prompt field 4",
        ]
    )


#################################
# HeuristicAgentFinalizer tests #
#################################
#          ,  ,
#          \\ \\
#          ) \\ \\    _p_
#          )^\))\))  /  *\
#           \_|| || / /^`-'
#  __       -\ \\--/ /
# <'  \\___/   ___. )'
#     `====\ )___/\\
#          //     `"
#          \\    /  \
#          `"
#
#################################


def create_finalizer_agent(mock_heuristic_prompt, sample_final_properties):
    return HeuristicAgentFinalizer.create(
        name="test_finalizer",
        prompt=mock_heuristic_prompt,
        topics=[CONDUCTOR_NAME],
        tool_executor=AsyncMock(spec=ToolExecutor),
        render_prompt=True,
        final_properties=sample_final_properties,
    )


TWO_FINAL_PROPERTIES = ["field1", "field2"]


@pytest.fixture
def finalizer_agent(mock_heuristic_prompt):
    return create_finalizer_agent(mock_heuristic_prompt, sample_final_properties=TWO_FINAL_PROPERTIES)


@pytest.fixture
def finalizer_agent_one_property(mock_heuristic_prompt):
    return create_finalizer_agent(mock_heuristic_prompt, sample_final_properties=["field1"])


class TestHeuristicAgentFinalizer:
    def test_finalizer_initialization(self, mock_heuristic_prompt):
        """Test if the finalizer agent initializes correctly with given properties"""
        agent = HeuristicAgentFinalizer.create(
            name="test_finalizer",
            prompt=mock_heuristic_prompt,
            topics=[CONDUCTOR_NAME],
            final_properties=TWO_FINAL_PROPERTIES,
        )

        assert agent.name == "test_finalizer"
        assert agent.final_properties == TWO_FINAL_PROPERTIES
        assert agent.topics == [CONDUCTOR_NAME]

    @pytest.mark.asyncio
    async def test_finalize_conversation_all_fields(self, finalizer_agent, finalizer_conversation):
        finalizer_agent.invoke = AsyncMock(return_value=finalizer_conversation)

        await finalizer_agent.finalize_conversation(finalizer_conversation)

        assert finalizer_conversation.final_result == {
            "field1": "user prompt field 1",
            "field2": "user prompt field 2",
        }
        finalizer_agent.invoke.assert_awaited_once_with(finalizer_conversation)

    @pytest.mark.asyncio
    async def test_finalize_conversation_one_property(self, finalizer_agent_one_property, finalizer_conversation):
        finalizer_agent_one_property.invoke = AsyncMock(return_value=finalizer_conversation)

        await finalizer_agent_one_property.finalize_conversation(finalizer_conversation)

        assert finalizer_conversation.final_result == {"field1": "user prompt field 1"}
        finalizer_agent_one_property.invoke.assert_awaited_once_with(finalizer_conversation)

    @pytest.mark.asyncio
    async def test_finalize_conversation_empty_properties(self, mock_heuristic_prompt, extended_conversation):
        """Test finalize_conversation with empty final_properties"""
        agent = HeuristicAgentFinalizer.create(
            name="test_finalizer", prompt=mock_heuristic_prompt, topics=[CONDUCTOR_NAME], final_properties=[]
        )

        agent.invoke = AsyncMock(return_value=extended_conversation)

        await agent.finalize_conversation(extended_conversation)
        assert extended_conversation.final_result == {}
