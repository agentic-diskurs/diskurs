from unittest.mock import AsyncMock

import pytest

from diskurs import MultiStepAgent, LLMClient, ToolExecutor
from diskurs.entities import ChatMessage, Role, MessageType
from diskurs.heuristic_agent import HeuristicAgentFinalizer
from diskurs.multistep_agent import MultistepAgentFinalizer
from test_files.tool_test_files.data_analysis_tools import (
    analyze_sales_data,
    analyze_employee_performance,
    generate_budget_projection,
)

CONDUCTOR_NAME = "my_conductor"


def create_multistep_agent(mock_prompt):
    # Update async_identity to accept **kwargs for flexibility
    async def async_identity(conversation, **kwargs):
        return conversation

    agent = MultiStepAgent(
        name="test_agent",
        prompt=mock_prompt,
        init_prompt_arguments_with_previous_agent=True,
        max_reasoning_steps=3,
        llm_client=AsyncMock(spec=LLMClient),
        topics=[CONDUCTOR_NAME],
    )
    agent.generate_validated_response = async_identity
    return agent


@pytest.fixture
def multistep_agent(mock_prompt):
    return create_multistep_agent(mock_prompt)


@pytest.fixture
def extended_multistep_agent(mock_extended_prompt):
    return create_multistep_agent(mock_extended_prompt)


@pytest.mark.asyncio
async def test_invoke_with_conductor_as_previous_agent(multistep_agent, conversation):
    conversation = conversation.append(
        message=ChatMessage(
            content="I am a conductor message", role=Role.USER, type=MessageType.CONDUCTOR, name=CONDUCTOR_NAME
        )
    )

    # Execute
    result = await multistep_agent.invoke(conversation)
    longterm_memory = conversation.get_agent_longterm_memory(CONDUCTOR_NAME)

    assert all(
        [
            longterm_memory.field1 == result.user_prompt_argument.field1,
            longterm_memory.field2 == result.user_prompt_argument.field2,
            longterm_memory.field3 == result.user_prompt_argument.field3,
        ]
    )


@pytest.mark.asyncio
async def test_invoke_with_agent_chain(multistep_agent, conversation):
    conversation = conversation.append(
        message=ChatMessage(
            content="I am a multistep agent message",
            role=Role.USER,
            type=MessageType.CONVERSATION,
        )
    )
    result = await multistep_agent.invoke(conversation)
    assert all(
        [
            conversation.user_prompt_argument.field1 == result.user_prompt_argument.field1,
            conversation.user_prompt_argument.field2 == result.user_prompt_argument.field2,
            conversation.user_prompt_argument.field3 == result.user_prompt_argument.field3,
        ]
    )


@pytest.mark.asyncio
async def test_invoke_with_longterm_memory_and_previous_agent(extended_multistep_agent, extended_conversation):
    extended_conversation = extended_conversation.append(
        message=ChatMessage(
            content="I am a conductor message", role=Role.USER, type=MessageType.CONDUCTOR, name=CONDUCTOR_NAME
        )
    )
    extended_conversation = extended_conversation.append(
        message=ChatMessage(
            content="I am a multistep agent message",
            role=Role.USER,
            type=MessageType.CONVERSATION,
        )
    )
    result = await extended_multistep_agent.invoke(extended_conversation)
    assert all(
        [
            result.user_prompt_argument.field1 == "extended user prompt field 1",
            result.user_prompt_argument.field2 == "longterm val 2",
            result.user_prompt_argument.field3 == "user prompt field 3",
            result.user_prompt_argument.field4 == "user prompt field 4",
        ]
    )


class TestRegisterTools:
    def test_register_tools(self, multistep_agent):
        multistep_agent.register_tools([analyze_sales_data])

        assert "analyze_sales_data" in [tool.name for tool in multistep_agent.tools]
        assert len(multistep_agent.tools) == 1

    def test_register_multiple_tools(self, multistep_agent):
        multistep_agent.register_tools([analyze_sales_data, analyze_employee_performance])

        assert "analyze_sales_data" in [tool.name for tool in multistep_agent.tools]
        assert len(multistep_agent.tools) == 2

    def test_register_new_tool_to_existing_tools(self, multistep_agent):
        multistep_agent.register_tools([analyze_sales_data])
        multistep_agent.register_tools([analyze_employee_performance])

        assert "analyze_sales_data" in [tool.name for tool in multistep_agent.tools]
        assert "analyze_employee_performance" in [tool.name for tool in multistep_agent.tools]
        assert len(multistep_agent.tools) == 2

    def test_register_multiple_tools_to_existing_tools(self, multistep_agent):
        multistep_agent.register_tools(analyze_sales_data)
        multistep_agent.register_tools([analyze_employee_performance, generate_budget_projection])

        assert len(multistep_agent.tools) == 3
        assert "analyze_sales_data" in [tool.name for tool in multistep_agent.tools]
        assert "analyze_employee_performance" in [tool.name for tool in multistep_agent.tools]
        assert "generate_budget_projection" in [tool.name for tool in multistep_agent.tools]


#################################
# MultistepAgentFinalizer tests #
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


def create_finalizer_agent(mock_prompt, sample_final_properties):
    return MultistepAgentFinalizer.create(
        name="test_finalizer",
        prompt=mock_prompt,
        topics=[CONDUCTOR_NAME],
        llm_client=AsyncMock(spec=LLMClient),
        tool_executor=AsyncMock(spec=ToolExecutor),
        final_properties=sample_final_properties,
    )


TWO_FINAL_PROPERTIES = ["field1", "field2"]


@pytest.fixture
def finalizer_agent(mock_prompt):
    return create_finalizer_agent(mock_prompt, sample_final_properties=TWO_FINAL_PROPERTIES)


@pytest.fixture
def finalizer_agent_one_property(mock_prompt):
    return create_finalizer_agent(mock_prompt, sample_final_properties=["field1"])


class TestHeuristicAgentFinalizer:
    def test_finalizer_initialization(self, mock_prompt):
        """Test if the finalizer agent initializes correctly with given properties"""
        agent = HeuristicAgentFinalizer.create(
            name="test_finalizer",
            prompt=mock_prompt,
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
    async def test_finalize_conversation_empty_properties(self, mock_prompt, extended_conversation):

        agent = MultistepAgentFinalizer.create(
            name="test_finalizer",
            prompt=mock_prompt,
            topics=[CONDUCTOR_NAME],
            final_properties=TWO_FINAL_PROPERTIES,
            llm_client=AsyncMock(spec=LLMClient),
        )

        agent.invoke = AsyncMock(return_value=extended_conversation)

        await agent.finalize_conversation(extended_conversation)
        assert extended_conversation.final_result == {}
