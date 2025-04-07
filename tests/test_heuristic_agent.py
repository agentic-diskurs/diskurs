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


def test_register_dispatcher():
    name = "test_agent"
    prompt = MagicMock()
    dispatcher = MagicMock()

    agent = HeuristicAgent(name=name, prompt=prompt, llm_client=None)
    agent.register_dispatcher(dispatcher)

    assert agent.dispatcher == dispatcher


def test_prepare_conversation():
    name = "test_agent"
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    prompt_argument = MagicMock(spec=PromptArgument)
    updated_conversation = MagicMock()

    conversation.update.return_value = updated_conversation

    agent = HeuristicAgent(name=name, prompt=prompt, llm_client=None)
    result = agent.prepare_conversation(conversation, prompt_argument)

    conversation.update.assert_called_once_with(prompt_argument=prompt_argument, active_agent=name)
    assert result == updated_conversation


@pytest.mark.asyncio
async def test_invoke(conversation):
    prompt = MagicMock()
    prompt.create_prompt_argument.return_value = conversation.prompt_argument
    prompt.heuristic_sequence = AsyncMock()
    prompt.render_user_template.return_value = "rendered_message"

    agent = HeuristicAgent(
        name="test_agent",
        prompt=prompt,
        llm_client=None,
        topics=["conductor_name"],
        tool_executor=MagicMock(),
        init_prompt_arguments_with_longterm_memory=True,
        render_prompt=True,
    )

    result = await agent.invoke(conversation)

    prompt.create_prompt_argument.assert_called_once()

    called_conversation = prompt.heuristic_sequence.call_args[1]["conversation"]

    assert called_conversation.conversation_id == conversation.conversation_id
    assert called_conversation.prompt_argument == conversation.prompt_argument
    assert called_conversation.chat == conversation.chat
    assert called_conversation.get_agent_longterm_memory(agent.name) == conversation.get_agent_longterm_memory(
        agent.name
    )
    assert called_conversation.active_agent == "test_agent"

    prompt.render_user_template.assert_called_once_with(
        "test_agent",
        prompt_args=prompt.heuristic_sequence.return_value.prompt_argument,
        message_type=MessageType.CONVERSATION,
    )


@pytest.mark.asyncio
async def test_invoke_no_executor():
    # Setup
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    prompt.create_prompt_argument.return_value = MagicMock()
    prompt.heuristic_sequence = AsyncMock()
    prompt.render_user_template.return_value = "rendered_template"

    agent = HeuristicAgent(
        name="test_agent",
        prompt=prompt,
        llm_client=None,
        topics=["conductor_name"],
        init_prompt_arguments_with_longterm_memory=True,
        render_prompt=True,
    )

    result = await agent.invoke(conversation)

    prompt.create_prompt_argument.assert_called_once()

    called_args = prompt.heuristic_sequence.call_args[1]
    called_conversation = called_args["conversation"]

    assert "conversation" in called_args
    assert "call_tool" in called_args
    assert called_args["call_tool"] is None

    prompt.render_user_template.assert_called_once_with(
        "test_agent",
        prompt_args=prompt.heuristic_sequence.return_value.prompt_argument,
        message_type=MessageType.CONVERSATION,
    )


@pytest.mark.asyncio
async def test_process_conversation(conversation):
    # Setup
    prompt = MagicMock()
    dispatcher = MagicMock()
    updated_conversation = conversation.update()

    agent = HeuristicAgent(
        name="test_agent", prompt=prompt, llm_client=None, topics=["conductor_name"], dispatcher=dispatcher
    )

    agent.invoke = AsyncMock(return_value=updated_conversation)
    dispatcher.publish = AsyncMock()

    await agent.process_conversation(conversation)

    agent.invoke.assert_awaited_once_with(conversation)
    dispatcher.publish.assert_awaited_once_with(topic="conductor_name", conversation=updated_conversation)


def create_prompt(prompt_argument):
    prompt = AsyncMock(spec=HeuristicPrompt)
    prompt.create_prompt_argument.return_value = prompt_argument
    prompt.render_user_template.return_value = ChatMessage(
        role=Role.USER,
        name="my_multistep",
        content="rendered template",
        type=MessageType.CONVERSATION,
    )
    prompt.prompt_argument = prompt_argument

    return prompt


@pytest.fixture
def mock_heuristic_prompt():
    return create_prompt(MyExtendedUserPromptArgument())


def create_heuristic_agent(mock_heuristic_prompt):
    # Update the async_identity function to accept **kwargs for flexibility
    async def async_identity(conversation, call_tool, **kwargs):
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
            content="{'next_agent': 'test_agent'}",
            role=Role.USER,
            type=MessageType.CONDUCTOR,
            name="my_conductor",
        )
    )
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
            result.prompt_argument.field1 == "extended user prompt field 1",
            result.prompt_argument.field2 == "longterm val 2",
            result.prompt_argument.field3 == "user prompt field 3",
            result.prompt_argument.field4 == "user prompt field 4",
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
