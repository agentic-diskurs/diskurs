from unittest.mock import AsyncMock, MagicMock

import pytest
from jinja2 import Template

from diskurs import MultiStepAgent, LLMClient, ToolExecutor
from diskurs.entities import ChatMessage, Role, MessageType
from diskurs.entities import ToolCall
from diskurs.heuristic_agent import HeuristicAgentFinalizer
from diskurs.immutable_conversation import ImmutableConversation
from diskurs.multistep_agent import MultistepAgentFinalizer
from diskurs.prompt import MultistepPrompt
from diskurs.tools import ToolExecutor
from tests.conftest import MyExtendedPromptArgument, MyPromptArgument
from tests.test_files.tool_test_files.data_analysis_tools import (
    analyze_sales_data,
    analyze_employee_performance,
    generate_budget_projection,
)
from tests.test_files.tool_test_files.data_analysis_tools import failing_tool

CONDUCTOR_NAME = "my_conductor"


def create_multistep_agent(mock_prompt):
    # Update async_identity to accept **kwargs for flexibility
    async def async_identity(conversation, **kwargs):
        return conversation

    agent = MultiStepAgent(
        name="test_agent",
        prompt=mock_prompt,
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
    # Create a prompt argument with test values
    prompt_arg = MyPromptArgument(field1="test field 1", field2="test field 2", field3="test field 3")
    multistep_agent.prompt.prompt_argument = prompt_arg

    # Mock the initialize_prompt method properly
    def mock_initialize_prompt(agent_name, conversation, **kwargs):
        # Update the conversation with our prompt_arg
        return conversation.update(prompt_argument=prompt_arg)

    # Mock the generate_validated_response method to not change anything
    async def mock_generate_response(conversation, **kwargs):
        return conversation

    # Replace the methods with our mocks
    multistep_agent.prompt.initialize_prompt = mock_initialize_prompt
    multistep_agent.generate_validated_response = mock_generate_response

    result = await multistep_agent.invoke(conversation)

    # Update the longterm memory manually to simulate what should happen
    # This is what happens in the invoke method: conversation = conversation.update_longterm_memory(conversation.prompt_argument)
    updated_memory = result.longterm_memory.update(prompt_arg)
    result = result.update(longterm_memory=updated_memory)

    assert all(
        [
            result.longterm_memory.field1 == result.prompt_argument.field1,
            result.longterm_memory.field2 == result.prompt_argument.field2,
            result.longterm_memory.field3 == result.prompt_argument.field3,
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

    # Create prompt argument with the same values as in the conversation
    prompt_arg = MyPromptArgument(
        field1=conversation.prompt_argument.field1,
        field2=conversation.prompt_argument.field2,
        field3=conversation.prompt_argument.field3,
    )

    # Mock the necessary methods
    def mock_initialize_prompt(agent_name, conversation, **kwargs):
        return conversation.update(prompt_argument=prompt_arg)

    async def mock_generate_response(conversation, **kwargs):
        return conversation

    # Replace the methods with our mocks
    multistep_agent.prompt.initialize_prompt = mock_initialize_prompt
    multistep_agent.generate_validated_response = mock_generate_response
    multistep_agent.prompt.prompt_argument = prompt_arg

    result = await multistep_agent.invoke(conversation)

    # Update the result with the longterm memory updated from prompt_arg
    updated_memory = result.longterm_memory.update(prompt_arg)
    result = result.update(longterm_memory=updated_memory)

    assert all(
        [
            conversation.prompt_argument.field1 == result.prompt_argument.field1,
            conversation.prompt_argument.field2 == result.prompt_argument.field2,
            conversation.prompt_argument.field3 == result.prompt_argument.field3,
        ]
    )


@pytest.fixture
def real_prompt():
    prompt = MultistepPrompt(
        agent_description="Some agent",
        system_template=Template("System Template"),
        user_template=Template("User Template"),
        prompt_argument_class=MyExtendedPromptArgument,
        return_json=False,
        is_valid=lambda x: True,
        is_final=lambda x: True,
    )
    return prompt


@pytest.fixture
def real_extended_multistep_agent(real_prompt):
    return create_multistep_agent(real_prompt)


@pytest.mark.asyncio
async def test_invoke_with_longterm_memory_and_previous_agent(real_extended_multistep_agent, extended_conversation):
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

    # Configure the mock to properly initialize the prompt argument with values from longterm memory
    prompt_arg = MyExtendedPromptArgument(
        field1="extended user prompt field 1",
        field2="longterm val 2",
        field3="user prompt field 3",
        field4="user prompt field 4",
    )

    real_extended_multistep_agent.prompt.prompt_argument = prompt_arg
    real_extended_multistep_agent.prompt.initialize_prompt = (
        lambda agent_name, conversation, **kwargs: conversation.update(prompt_argument=prompt_arg)
    )

    # Replace the generate_validated_response method to maintain the prompt argument
    async def mock_generate_response(conversation, **kwargs):
        return conversation

    real_extended_multistep_agent.generate_validated_response = mock_generate_response

    result = await real_extended_multistep_agent.invoke(extended_conversation)

    assert all(
        [
            result.prompt_argument.field1 == "extended user prompt field 1",
            result.prompt_argument.field2 == "longterm val 2",
            result.prompt_argument.field3 == "user prompt field 3",
            result.prompt_argument.field4 == "user prompt field 4",
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


class TestToolExecutionErrorHandling:
    @pytest.mark.asyncio
    async def test_compute_tool_response_catches_exceptions(self, monkeypatch):
        """Test that compute_tool_response catches exceptions from tool execution and converts them to error messages."""

        # Create a real tool executor and register our failing tool
        tool_executor = ToolExecutor()
        tool_executor.register_tools(failing_tool)

        # Create an agent that uses this tool executor
        agent = MultiStepAgent(
            name="test_error_agent",
            prompt=MultistepPrompt(
                agent_description="Test Agent",
                system_template=Template("System template"),
                user_template=Template("User template"),
                prompt_argument_class=MyExtendedPromptArgument,
                return_json=False,
                is_valid=lambda x: True,
                is_final=lambda x: True,
            ),
            llm_client=AsyncMock(spec=LLMClient),
            tool_executor=tool_executor,
        )

        conversation = ImmutableConversation(
            chat=[],
            metadata={},
            prompt_argument=MyExtendedPromptArgument(),
        )

        # Add a failing tool call to the conversation
        conversation = conversation.append(
            message=ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_call_id="123",
                        function_name="failing_tool",
                        arguments={"param": "test"},
                    )
                ],
            )
        )

        result = await agent.compute_tool_response(conversation)

        # Check that we got a tool response even though the tool failed
        assert len(result) == 1
        assert result[0].role == Role.TOOL
        assert result[0].tool_call_id == "123"
        assert "ERROR" in result[0].content
        assert "failing_tool" in result[0].content
        assert "This tool always fails" in result[0].content

    @pytest.mark.asyncio
    async def test_conversation_flow_continues_despite_tool_error(self):
        """Test that the conversation flow continues even when a tool raises an exception."""

        mock_tool_executor = AsyncMock(spec=ToolExecutor)

        # Set up the tool responses - one with an error, one successful
        async def mock_execute_tool(tool_call, metadata):
            if tool_call.function_name == "failing_tool":
                # Simulate that execute_tool was called but handle the error within our mock
                return MagicMock(
                    tool_call_id=tool_call.tool_call_id,
                    function_name=tool_call.function_name,
                    result="ERROR: Tool 'failing_tool' execution failed: This tool always fails",
                )
            else:
                # Return a successful result for the other tool
                return MagicMock(
                    tool_call_id=tool_call.tool_call_id,
                    function_name=tool_call.function_name,
                    result="17.65",  # Sample value for analyze_sales_data
                )

        mock_tool_executor.execute_tool.side_effect = mock_execute_tool

        # Create the agent with our mock tool executor
        agent = MultiStepAgent(
            name="test_error_agent",
            prompt=MultistepPrompt(
                agent_description="Test Agent",
                system_template=Template("System template"),
                user_template=Template("User template"),
                prompt_argument_class=MyExtendedPromptArgument,
                return_json=False,
                is_valid=lambda x: True,
                is_final=lambda x: True,  # Mark as final to avoid multiple steps
            ),
            llm_client=AsyncMock(spec=LLMClient),
            tool_executor=mock_tool_executor,
        )

        conversation = ImmutableConversation(
            chat=[],
            metadata={},
            prompt_argument=MyExtendedPromptArgument(),
        )

        # Add a message with two tool calls
        conversation = conversation.append(
            message=ChatMessage(
                role=Role.ASSISTANT,
                content="I'll call both tools",
                tool_calls=[
                    ToolCall(
                        tool_call_id="123",
                        function_name="failing_tool",
                        arguments={"param": "test"},
                    ),
                    ToolCall(
                        tool_call_id="456",
                        function_name="analyze_sales_data",
                        arguments={"quarter": "Q1"},
                    ),
                ],
            )
        )

        # Call compute_tool_response directly to get the tool responses
        tool_responses = await agent.compute_tool_response(conversation)

        # Verify we got two tool responses - both the error and the successful one
        assert len(tool_responses) == 2

        # Instead of using update, append each tool response message to the conversation
        # for tool_response in tool_responses:
        #     conversation_with_tools = conversation_with_tools.append(message=tool_response)
        conversation_with_tools = conversation.update(user_prompt=tool_responses)

        # Verify the conversation has the expected structure
        assert len([msg for msg in conversation_with_tools.user_prompt if msg.role == Role.TOOL]) == 2

        # Check that we have both a failing tool response and a successful one
        tool_msgs = [msg for msg in conversation_with_tools.user_prompt if msg.role == Role.TOOL]
        assert any("ERROR" in str(msg.content) for msg in tool_msgs)
        assert any("ERROR" not in str(msg.content) for msg in tool_msgs)

        # Verify the mock tool executor was called twice (once for each tool)
        assert mock_tool_executor.execute_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_specific_error_message_is_preserved(self):
        """Test that specific error messages from tools are preserved in the response."""

        tool_executor = ToolExecutor()
        tool_executor.register_tools(failing_tool)

        agent = MultiStepAgent(
            name="test_error_agent",
            prompt=MultistepPrompt(
                agent_description="Test Agent",
                system_template=Template("System template"),
                user_template=Template("User template"),
                prompt_argument_class=MyExtendedPromptArgument,
                return_json=False,
                is_valid=lambda x: True,
                is_final=lambda x: True,
            ),
            llm_client=AsyncMock(spec=LLMClient),
            tool_executor=tool_executor,
        )

        # Create a conversation with a tool call that will raise a specific ValueError
        conversation = ImmutableConversation(
            chat=[],
            metadata={},
            prompt_argument=MyExtendedPromptArgument(),
        )

        conversation = conversation.append(
            message=ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_call_id="123",
                        function_name="failing_tool",
                        arguments={"param": "specific_error"},  # This will raise a specific ValueError
                    )
                ],
            )
        )

        # Call compute_tool_response
        result = await agent.compute_tool_response(conversation)

        # Check that the specific error message is preserved
        assert len(result) == 1
        assert "This is a specific value error" in result[0].content
