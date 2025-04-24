from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, Mock

import pytest
from jinja2 import Template

from .conftest import MyLongtermMemory
from diskurs import ImmutableConversation, Conversation, PromptValidationError, InputField
from diskurs.conductor_agent import ConductorAgent, validate_finalization
from diskurs.entities import (
    ChatMessage,
    Role,
    MessageType,
    LongtermMemory,
    PromptArgument,
    RoutingRule,
    ToolDescription,
    LockedField,
    OutputField,
    ToolCall,
)
from diskurs.prompt import ConductorPrompt, DefaultConductorPromptArgument
from diskurs.protocols import (
    LLMClient,
    ConversationDispatcher,
    ToolExecutor,
)

FINALIZER_NAME = "finalizer"


# ----- Data Classes for Testing -----


@dataclass
class MyLongTermMemory(LongtermMemory):
    user_query: Optional[str] = ""
    field1: Optional[str] = ""
    field2: Optional[str] = ""
    field3: Optional[str] = ""


@dataclass
class MyPromptArgument(PromptArgument):
    field1: Optional[str] = ""
    field2: Optional[str] = ""
    field3: Optional[str] = ""
    next_agent: Optional[str] = ""


@dataclass
class MockPromptArgument(PromptArgument):
    content: str = ""
    next_agent: str = ""


@dataclass
class DefaultCanFinalizePromptArgument(PromptArgument):
    can_finalize: Optional[bool] = None


# ----- Mock Rules for Rule-Based Routing Tests -----


def rule_always_true(conversation: Conversation) -> bool:
    return True


def rule_always_false(conversation: Conversation) -> bool:
    return False


def rule_raises_exception(conversation: Conversation) -> bool:
    raise ValueError("Test exception")


def rule_content_match(conversation: Conversation) -> bool:
    """Route based on user message content"""
    user_messages = [msg for msg in conversation.chat if msg.role == Role.USER]
    if not user_messages:
        return False
    return "keyword" in user_messages[-1].content.lower()


# ----- Fixtures for Common Test Configuration -----


def create_conductor_prompt(
    has_finalizer: bool = True,
    can_finalize_return_value: bool = True,
    has_can_finalize: bool = True,
):
    def finalize(longterm_memory):
        return {"result": "final result"}

    def can_finalize(longterm_memory):
        return can_finalize_return_value

    prompt = Mock(spec=ConductorPrompt)
    prompt.init_longterm_memory.return_value = MyLongTermMemory()
    prompt.create_prompt_argument.return_value = Mock()

    # Use a regular Mock instead of AsyncMock for finalize
    prompt.finalize = Mock(side_effect=finalize)
    prompt.can_finalize = Mock(side_effect=can_finalize)
    prompt._finalize = finalize if has_finalizer else None
    prompt._can_finalize = can_finalize if has_can_finalize else None
    prompt.fail.return_value = {}
    prompt.is_final = Mock(return_value=True)
    prompt.parse_user_prompt = Mock()

    return prompt


@pytest.fixture
def mock_prompt():
    return create_conductor_prompt()


@pytest.fixture
def mock_prompt_no_finalize():
    return create_conductor_prompt(has_finalizer=False)


@pytest.fixture
def mock_prompt_cannot_finalize():
    return create_conductor_prompt(can_finalize_return_value=False)


@pytest.fixture
def mock_llm_client():
    async def stub_generate_validated_response(conversation, message_type=None, tools=None):
        return conversation.append(
            ChatMessage(role=Role.ASSISTANT, content='{"next_agent": "agent1"}', type=MessageType.CONDUCTOR)
        )

    llm_client = Mock(spec=LLMClient)
    llm_client.generate = AsyncMock(side_effect=stub_generate_validated_response)
    return llm_client


@pytest.fixture
def mock_dispatcher():
    dispatcher = AsyncMock(spec=ConversationDispatcher)
    dispatcher.publish = AsyncMock()
    dispatcher.finalize = AsyncMock()
    dispatcher.publish_final = AsyncMock()
    return dispatcher


@pytest.fixture
def mock_rules():
    return [
        RoutingRule(
            name="test_rule_1",
            description="Rule that always returns true",
            condition=rule_always_true,
            target_agent="agent1",
        ),
        RoutingRule(
            name="test_rule_2",
            description="Rule that always returns false",
            condition=rule_always_false,
            target_agent="agent2",
        ),
        RoutingRule(
            name="test_rule_3",
            description="Rule with content matching",
            condition=rule_content_match,
            target_agent="agent3",
        ),
        RoutingRule(
            name="test_rule_error",
            description="Rule that raises an exception",
            condition=rule_raises_exception,
            target_agent="error_agent",
        ),
    ]


@pytest.fixture
def mock_conversation():
    return ImmutableConversation(prompt_argument=MockPromptArgument()).append(
        ChatMessage(role=Role.USER, content="test message", name="user")
    )


@pytest.fixture
def mock_conversation_with_keyword():
    return ImmutableConversation(prompt_argument=MockPromptArgument()).append(
        ChatMessage(role=Role.USER, content="test message with keyword", name="user")
    )


def create_conductor_agent(
    mock_dispatcher,
    mock_llm_client,
    mock_prompt,
    finalizer=None,
    supervisor=None,
    rules=None,
    fallback_to_llm=True,
    tools=None,
    tool_executor=None,
    init_prompt_arguments_with_longterm_memory=True,
    init_prompt_arguments_with_previous_agent=True,
):
    agent = ConductorAgent(
        name="conductor",
        prompt=mock_prompt,
        llm_client=mock_llm_client,
        topics=["agent1", "agent2", "agent3"],
        locked_fields={},
        finalizer_name=finalizer,
        supervisor=supervisor,
        dispatcher=mock_dispatcher,
        max_trials=5,
        max_dispatches=5,
        rules=rules,
        fallback_to_llm=fallback_to_llm,
        tools=tools,
        tool_executor=tool_executor,
        init_prompt_arguments_with_longterm_memory=init_prompt_arguments_with_longterm_memory,
        init_prompt_arguments_with_previous_agent=init_prompt_arguments_with_previous_agent,
    )
    return agent


@pytest.fixture
def conductor_agent(mock_prompt, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt)


@pytest.fixture
def conductor_cannot_finalize(mock_prompt_cannot_finalize, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt_cannot_finalize)


@pytest.fixture
def conductor_agent_with_supervisor(mock_prompt_no_finalize, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(
        mock_dispatcher,
        mock_llm_client,
        mock_prompt_no_finalize,
        supervisor="supervisor",
    )


@pytest.fixture
def conductor_agent_with_finalizer(mock_prompt_no_finalize, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(
        mock_dispatcher,
        mock_llm_client,
        mock_prompt_no_finalize,
        finalizer=FINALIZER_NAME,
    )


@pytest.fixture
def conductor_agent_with_finalizer_function(mock_prompt, mock_llm_client, mock_dispatcher):
    return create_conductor_agent(mock_dispatcher, mock_llm_client, mock_prompt)


@pytest.fixture
def conductor_agent_with_rules(mock_prompt, mock_llm_client, mock_dispatcher, mock_rules):
    """Create a conductor agent with rules"""
    return create_conductor_agent(
        mock_dispatcher, mock_llm_client, mock_prompt, rules=mock_rules, fallback_to_llm=True
    )


@pytest.fixture
def conductor_agent_rules_only(mock_prompt, mock_dispatcher, mock_rules):
    """Create a rule-only conductor (no LLM fallback)"""
    return create_conductor_agent(
        mock_dispatcher, None, mock_prompt, rules=mock_rules, fallback_to_llm=False  # No LLM client
    )


@pytest.fixture
def mock_tool_executor():
    """Mock tool executor for testing tool handling in ConductorAgent"""
    tool_executor = Mock(spec=ToolExecutor)
    tool_executor.call_tool = AsyncMock(return_value={"result": "tool_result"})
    tool_executor.execute_tool = AsyncMock(return_value=Mock(tool_call_id="test_id", result="tool execution result"))
    return tool_executor


@pytest.fixture
def mock_tool_descriptions():
    """Create mock tool descriptions for testing"""
    return [
        ToolDescription(
            name="test_tool",
            description="Test tool for testing",
            arguments={
                "type": "object",
                "properties": {"test_param": {"type": "string"}},
                "required": ["test_param"],
            },
        )
    ]


@pytest.fixture
def conductor_agent_with_tools(
    mock_prompt, mock_llm_client, mock_dispatcher, mock_tool_executor, mock_tool_descriptions
):
    """Create a conductor agent with tools"""
    return create_conductor_agent(
        mock_dispatcher, mock_llm_client, mock_prompt, tools=mock_tool_descriptions, tool_executor=mock_tool_executor
    )


class MyConductorPromptArgument(PromptArgument):
    agent_descriptions: LockedField[str] = "should never be changed"
    field1: InputField[str] = "input field 1"
    field2: OutputField[str] = "output field 2"


@pytest.fixture
def conductor_agent():
    with open(Path(__file__).parent.parent / "diskurs" / "assets" / "json_formatting.jinja2", encoding="utf-8") as f:
        json_formatting_template = f.read()
        json_formatting_template = Template(json_formatting_template)

    prompt = ConductorPrompt(
        agent_description="Test conductor agent",
        system_template=Template(
            "Test conductor agent description {{ agent_descriptions }}, input field {{ field1 }}, output "
            "field {{ field2 }}"
        ),
        user_template=Template("Test conductor agent"),
        prompt_argument_class=MyConductorPromptArgument,
        json_formatting_template=json_formatting_template,
        longterm_memory=MyLongtermMemory,
    )

    return ConductorAgent(
        name="conductor",
        prompt=prompt,
        llm_client=AsyncMock(spec=LLMClient),
        topics=["agent1", "agent2"],
        locked_fields={
            "agent1": "Test agent 1",
            "agent2": "Test agent 2",
        },
        dispatcher=AsyncMock(spec=ConversationDispatcher),
    )


def test_update_longterm_memory(conductor_agent):
    conversation = ImmutableConversation(
        prompt_argument=MyPromptArgument(field1="value1", field2="value2", field3="value3")
    )

    updated_conversation = conductor_agent.create_or_update_longterm_memory(conversation=conversation)
    longterm_memory = updated_conversation.get_agent_longterm_memory("conductor")

    assert longterm_memory.field1 == "value1"
    assert longterm_memory.field2 == "value2"
    assert longterm_memory.field3 == "value3"


def test_update_longterm_memory_with_overwrite(conductor_agent):
    conversation = ImmutableConversation(prompt_argument=MyPromptArgument(field1="new_value1"))
    longterm_memory = MyLongTermMemory(field1="existing_value1")

    updated_conversation = conductor_agent.create_or_update_longterm_memory(conversation=conversation)
    longterm_memory = updated_conversation.get_agent_longterm_memory("conductor")

    assert longterm_memory.field1 == "new_value1"  # Should be overwritten


@pytest.mark.asyncio
async def test_max_dispatches(conductor_cannot_finalize):
    conductor_cannot_finalize.n_dispatches = 49
    conversation = ImmutableConversation(prompt_argument=MyPromptArgument()).append(
        ChatMessage(Role.ASSISTANT, content='{"next_agent": "agent1"}')
    )
    longterm_memory = MyLongTermMemory()

    conversation = conversation.update_agent_longterm_memory(
        agent_name=conductor_cannot_finalize.name,
        longterm_memory=longterm_memory,
    )

    # Use a regular Mock with a return value instead of an AsyncMock
    fail_result = {"error": "max dispatches reached"}
    conductor_cannot_finalize.prompt.fail = Mock(return_value=fail_result)

    await conductor_cannot_finalize.process_conversation(conversation)

    # Verify the fail mock was called correctly
    conductor_cannot_finalize.prompt.fail.assert_called_once_with(longterm_memory)

    # Check that the conversation's final_result was set correctly
    assert conversation.final_result == fail_result


@pytest.mark.asyncio
async def test_conductor_agent_valid_next_agent(mock_prompt, mock_dispatcher):
    """Test that a conductor agent correctly routes to a valid next agent when one is determined.

    This test verifies the core routing functionality of the ConductorAgent - when a conversation
    is processed and a valid next_agent is determined, the conversation should be published to that agent.
    """
    # Create a minimal ConductorAgent for testing
    agent = ConductorAgent(
        name="test_conductor",
        prompt=mock_prompt,
        llm_client=None,  # Not needed for this test
        topics=["agent1", "agent2", "agent3"],
        locked_fields={},
        dispatcher=mock_dispatcher,
    )

    # Create a simple conversation with a prompt argument that already has next_agent set
    initial_conversation = ImmutableConversation(prompt_argument=MockPromptArgument(next_agent="agent1"))

    # Mock create_or_update_longterm_memory to return the same conversation
    agent.create_or_update_longterm_memory = Mock(return_value=initial_conversation)

    # Mock can_finalize to return False so that the finalize branch isn't taken
    agent.can_finalize = AsyncMock(return_value=False)

    # Setup the mock_prompt to return the same conversation when initialized (no changes needed)
    agent.prompt.init_prompt = Mock(return_value=initial_conversation)

    # Mock invoke to return the same conversation (simulate that routing decision has been made)
    agent.invoke = AsyncMock(return_value=initial_conversation)

    # Process the conversation
    await agent.process_conversation(initial_conversation)

    # Verify the conversation was published to the correct agent
    mock_dispatcher.publish.assert_called_once()
    called_topic = mock_dispatcher.publish.call_args[1]["topic"]
    assert (
        called_topic == "agent1"
    ), f"Expected conversation to be routed to 'agent1', but was routed to '{called_topic}'"


@pytest.mark.asyncio
async def test_conductor_agent_fail_on_max_dispatches(
    conductor_cannot_finalize,
):
    conductor_cannot_finalize.n_dispatches = conductor_cannot_finalize.max_dispatches - 1

    conversation = ImmutableConversation()
    conversation = conversation.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content='{"next_agent": "invalid_agent"}',
            type=MessageType.CONDUCTOR,
        )
    )

    def is_valid(prompt_args):
        raise PromptValidationError(
            f"{prompt_args.next_agent} cannot be routed to from this agent. Valid agents are: {conductor_agent.topics}"
        )

    conductor_cannot_finalize.prompt.is_valid = is_valid

    await conductor_cannot_finalize.process_conversation(conversation)

    conductor_cannot_finalize.prompt.fail.assert_called_once()


@pytest.mark.asyncio
async def test_finalize_return_to_supervisor(conductor_agent_with_supervisor, mock_conversation):
    conductor_agent_with_supervisor.add_routing_message_to_chat = AsyncMock(return_value=mock_conversation)

    await conductor_agent_with_supervisor.finalize(mock_conversation)

    conductor_agent_with_supervisor.dispatcher.publish.assert_called_once_with(
        topic="supervisor", conversation=mock_conversation
    )


@pytest.mark.asyncio
async def test_finalize_call_finalizer(conductor_agent_with_finalizer, mock_conversation):
    conductor_agent_with_finalizer.add_routing_message_to_chat = AsyncMock(return_value=mock_conversation)

    await conductor_agent_with_finalizer.finalize(mock_conversation)

    conductor_agent_with_finalizer.dispatcher.publish_final.assert_called_once_with(
        topic=FINALIZER_NAME, conversation=mock_conversation
    )


@pytest.mark.asyncio
async def test_finalize_call_prompt_function(conductor_agent_with_finalizer_function, mock_conversation):
    await conductor_agent_with_finalizer_function.finalize(mock_conversation)

    # No need to await final_result as it's a direct property
    assert mock_conversation.final_result == {"result": "final result"}
    conductor_agent_with_finalizer_function.dispatcher.publish.assert_not_called()


# ----- Tests for Rule-Based Routing -----


def test_evaluate_rules_first_match(conductor_agent_with_rules, mock_conversation):
    # Should return the first rule's target agent since it always returns True
    next_agent = conductor_agent_with_rules.evaluate_rules(mock_conversation)
    assert next_agent == "agent1"


def test_evaluate_rules_content_match(conductor_agent_with_rules, mock_conversation_with_keyword):
    # Make the first rule return false, so it falls through to the content match rule
    conductor_agent_with_rules.rules[0].condition = rule_always_false
    conductor_agent_with_rules.rules[1].condition = rule_content_match

    next_agent = conductor_agent_with_rules.evaluate_rules(mock_conversation_with_keyword)
    assert next_agent == "agent2"  # The second rule's target agent


def test_evaluate_rules_no_match(conductor_agent_with_rules, mock_conversation):
    # Make all rules return false
    for rule in conductor_agent_with_rules.rules:
        rule.condition = rule_always_false

    next_agent = conductor_agent_with_rules.evaluate_rules(mock_conversation)
    assert next_agent is None  # No rule matched


def test_evaluate_rules_handles_exceptions(conductor_agent_with_rules, mock_conversation):
    # Make all rules throw exceptions
    for rule in conductor_agent_with_rules.rules:
        rule.condition = rule_raises_exception

    # Should return None and not raise exceptions
    next_agent = conductor_agent_with_rules.evaluate_rules(mock_conversation)
    assert next_agent is None


@pytest.mark.asyncio
async def test_invoke_with_rule_match(conductor_agent_with_rules, mock_conversation):
    # Set up the agent to use rule-based routing
    conductor_agent_with_rules.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    # Set up the prompt to properly create a prompt_argument with next_agent
    def mock_create_prompt_arg(**kwargs):
        if "next_agent" in kwargs:
            return MockPromptArgument(next_agent=kwargs["next_agent"])
        return MockPromptArgument()

    conductor_agent_with_rules.prompt.create_prompt_argument = Mock(side_effect=mock_create_prompt_arg)

    # Call invoke
    result = await conductor_agent_with_rules.invoke(mock_conversation, MessageType.CONDUCTOR)

    # Verify a rule was evaluated and next_agent is set
    assert hasattr(result.prompt_argument, "next_agent")
    assert result.prompt_argument.next_agent == "agent1"


@pytest.mark.asyncio
async def test_invoke_fallback_to_llm(conductor_agent_with_rules, mock_conversation):
    # Make all rules return false
    for rule in conductor_agent_with_rules.rules:
        rule.condition = rule_always_false

    conductor_agent_with_rules.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    # Set up LLM to return a response with next_agent
    async def mock_llm_generate(conversation, message_type=None, tools=None):
        return conversation.append(
            ChatMessage(role=Role.ASSISTANT, content='{"next_agent": "llm_agent"}', type=MessageType.CONDUCTOR)
        )

    mock_llm_generate = AsyncMock(side_effect=mock_llm_generate)

    # Mock the prompt parsing to return a valid prompt argument
    conductor_agent_with_rules.prompt.parse_user_prompt.return_value = DefaultConductorPromptArgument(
        next_agent="llm_agent"
    )
    conductor_agent_with_rules.prompt.is_final.return_value = True
    # Call invoke
    result = await conductor_agent_with_rules.invoke(mock_conversation, MessageType.CONDUCTOR)

    # Verify LLM was used and next_agent is set correctly
    mock_llm_generate.assert_called_once()
    assert result.prompt_argument.next_agent == "llm_agent"


@pytest.mark.asyncio
async def test_process_conversation_with_rules(conductor_agent_with_rules, mock_conversation):
    """Test that process_conversation correctly handles rule-based routing"""
    # Setup the agent to use rule-based routing
    conductor_agent_with_rules.create_or_update_longterm_memory = Mock(return_value=mock_conversation)
    conductor_agent_with_rules.can_finalize = AsyncMock(return_value=False)

    # Mock the invoke method to return a conversation with next_agent set in both:
    # 1. prompt_argument
    # 2. A properly formatted JSON message
    async def mock_invoke(conversation, message_type=None):
        updated = conversation.update(prompt_argument=MockPromptArgument(next_agent="agent1"))
        # Add a JSON message that can be parsed
        return updated.append(
            ChatMessage(
                role=Role.ASSISTANT, content='{"next_agent": "agent1"}', name="test_agent", type=MessageType.CONDUCTOR
            )
        )

    conductor_agent_with_rules.invoke = AsyncMock(side_effect=mock_invoke)

    # Call process_conversation
    await conductor_agent_with_rules.process_conversation(mock_conversation)

    # Verify the dispatcher was called with the right agent
    conductor_agent_with_rules.dispatcher.publish.assert_called_once()
    args, kwargs = conductor_agent_with_rules.dispatcher.publish.call_args
    assert kwargs.get("topic") == "agent1" or args[0] == "agent1"


@pytest.mark.asyncio
async def test_rule_only_conductor_no_llm_fallback(conductor_agent_rules_only, mock_conversation):
    """Test that a rule-only conductor doesn't try to use LLM when fallback is disabled"""
    # Make all rules return false
    for rule in conductor_agent_rules_only.rules:
        rule.condition = rule_always_false

    # Clear any existing next_agent value
    clean_conversation = mock_conversation.update(prompt_argument=MockPromptArgument(next_agent=None))
    conductor_agent_rules_only.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    # Also mock prompt.create_prompt_argument to ensure it returns a clean object
    conductor_agent_rules_only.prompt.create_prompt_argument = Mock(return_value=MockPromptArgument(next_agent=None))

    # Call invoke
    result = await conductor_agent_rules_only.invoke(clean_conversation, MessageType.CONDUCTOR)

    # Next agent should be None since no rules matched and fallback is off
    assert not hasattr(result.prompt_argument, "next_agent") or result.prompt_argument.next_agent is None


@pytest.mark.asyncio
async def test_rule_based_routing_with_update(conductor_agent_with_rules, mock_conversation):
    """Test that rule-based routing correctly updates the user prompt argument"""
    # Setup
    conductor_agent_with_rules.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    # Create a prompt argument with existing values to preserve
    prompt_arg = DefaultConductorPromptArgument(next_agent=None)
    mock_conversation = mock_conversation.update(prompt_argument=prompt_arg)

    # Create a new user prompt argument to return
    updated_prompt_arg = DefaultConductorPromptArgument(next_agent="agent1")
    conductor_agent_with_rules.prompt.create_prompt_argument.return_value = updated_prompt_arg

    # Call invoke
    result = await conductor_agent_with_rules.invoke(mock_conversation, MessageType.CONDUCTOR)

    # Verify the user prompt argument was updated with next_agent
    assert result.prompt_argument.next_agent == "agent1"


# ----- Tests for Configuration and Validation -----


def test_validate_finalization_all_none():
    prompt = Mock()
    prompt._finalize = lambda x: x
    prompt._can_finalize = lambda x: True

    # Should not raise or modify prompt
    validate_finalization(None, prompt, None)
    assert hasattr(prompt, "_finalize")
    assert hasattr(prompt, "_can_finalize")


def test_validate_finalization_with_finalizer():
    prompt = Mock()
    prompt._finalize = lambda x: x
    prompt._can_finalize = lambda x: True

    validate_finalization("finalizer", prompt, None)
    assert not hasattr(prompt, "_finalize")
    assert hasattr(prompt, "_can_finalize")


def test_validate_finalization_with_supervisor():
    prompt = Mock()
    prompt._finalize = lambda x: x
    prompt._can_finalize = lambda x: True

    validate_finalization(None, prompt, "supervisor")
    assert not hasattr(prompt, "_finalize")
    assert hasattr(prompt, "_can_finalize")


def test_validate_finalization_with_both_raises():
    prompt = Mock()
    prompt._finalize = lambda x: x

    with pytest.raises(
        AssertionError,
        match="Either finalizer_name or supervisor must be set, but not both",
    ):
        validate_finalization("finalizer", prompt, "supervisor")


def test_validate_finalization_with_both_unset_leaves_finalize():
    prompt = Mock()
    prompt._finalize = lambda x: x
    prompt._can_finalize = lambda x: True

    validate_finalization(finalizer_name=None, prompt=prompt, supervisor=None)
    assert hasattr(prompt, "_finalize")


def test_validate_finalization_with_all_raises():
    prompt = Mock()
    prompt._finalize = lambda x: x
    prompt._can_finalize = lambda x: True

    with pytest.raises(
        AssertionError,
        match="Either finalizer_name or supervisor must be set, but not both",
    ):
        validate_finalization("finalizer", prompt, "supervisor")


@pytest.mark.asyncio
async def test_invoke_rule_match_one_message(conductor_agent_with_rules, mock_conversation):
    """Test that invoke appends only one message when a rule matches."""
    conductor_agent_with_rules.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    # Set up the prompt to properly create a prompt_argument with next_agent
    def mock_create_prompt_arg(**kwargs):
        if "next_agent" in kwargs:
            return MockPromptArgument(next_agent=kwargs["next_agent"])
        return MockPromptArgument()

    conductor_agent_with_rules.prompt.create_prompt_argument = Mock(side_effect=mock_create_prompt_arg)

    initial_message_count = len(mock_conversation.chat)
    result = await conductor_agent_with_rules.invoke(mock_conversation, MessageType.CONDUCTOR)

    # Assert that only one message was appended
    assert len(result.chat) == initial_message_count + 1


@pytest.mark.asyncio
async def test_invoke_no_rule_match_one_message(conductor_agent_with_rules, mock_conversation):
    """Test that invoke appends only one message when no rule matches and LLM is used."""
    # Make all rules return false
    for rule in conductor_agent_with_rules.rules:
        rule.condition = rule_always_false

    conductor_agent_with_rules.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    # Mock generate_validated_response to return a modified conversation
    async def mock_generate_validated_response(conversation, tools=None, message_type=None):
        return conversation.append(
            ChatMessage(role=Role.ASSISTANT, content="LLM response", name="llm", type=MessageType.CONDUCTOR)
        )

    conductor_agent_with_rules.generate_validated_response = AsyncMock(side_effect=mock_generate_validated_response)

    initial_message_count = len(mock_conversation.chat)
    result = await conductor_agent_with_rules.invoke(mock_conversation, MessageType.CONDUCTOR)

    # Assert that generate_validated_response was called
    conductor_agent_with_rules.generate_validated_response.assert_called_once()

    # Assert that only one message was appended
    assert len(result.chat) == initial_message_count + 1


@pytest.mark.asyncio
async def test_invoke_no_rule_no_llm_no_message(conductor_agent_rules_only, mock_conversation):
    """Test that invoke appends no message when no rule matches and no LLM is available."""
    # Make all rules return false
    for rule in conductor_agent_rules_only.rules:
        rule.condition = rule_always_false

    conductor_agent_rules_only.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    initial_message_count = len(mock_conversation.chat)
    result = await conductor_agent_rules_only.invoke(mock_conversation, MessageType.CONDUCTOR)

    # Assert that no message was appended
    assert len(result.chat) == initial_message_count


# ----- Tests for BaseAgent Tool Functionality in ConductorAgent -----


@pytest.mark.asyncio
async def test_handle_tool_call(conductor_agent_with_tools, mock_conversation):
    """Test that tool calls are properly handled using the BaseAgent.handle_tool_call method."""
    # Create a conversation with tool calls
    conversation_with_tool_call = mock_conversation.append(
        ChatMessage(
            role=Role.ASSISTANT,
            content=None,
            name=conductor_agent_with_tools.name,
            tool_calls=[
                ToolCall(
                    tool_call_id="test_id",
                    function_name="test_tool",
                    arguments={"test_param": "test_value"},
                )
            ],
        )
    )

    # Call handle_tool_call
    result = await conductor_agent_with_tools.handle_tool_call(
        conversation_with_tool_call, conversation_with_tool_call
    )

    # Verify the tool was executed and the result was added to the conversation
    conductor_agent_with_tools.tool_executor.execute_tool.assert_called_once()

    # Check there's a tool response message
    tool_responses = [msg for msg in result.user_prompt if msg.role == Role.TOOL]
    assert len(tool_responses) == 1
    assert tool_responses[0].tool_call_id == "test_id"
    assert tool_responses[0].content == "tool execution result"


@pytest.mark.asyncio
async def test_prepare_invoke_with_initialization_flags(mock_dispatcher, mock_llm_client, mock_prompt):
    """Test that the prepare_invoke method correctly initializes conversation based on flags"""
    # Create agent with initialization flags disabled
    conductor = create_conductor_agent(
        mock_dispatcher,
        mock_llm_client,
        mock_prompt,
        init_prompt_arguments_with_longterm_memory=False,
        init_prompt_arguments_with_previous_agent=False,
    )

    # Create mock conversation
    conversation = ImmutableConversation(prompt_argument=MyPromptArgument(field1="original_value"))

    # Mock the longterm memory that would normally be retrieved
    longterm_memory = MyLongTermMemory(field1="memory_value")
    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)

    # Mock init_prompt to return conversation with new prompt argument
    new_prompt_arg = MyPromptArgument(field1="")
    conductor.prompt.create_prompt_argument = Mock(return_value=new_prompt_arg)
    conductor.prompt.init_prompt = Mock(return_value=conversation.update(prompt_argument=new_prompt_arg))

    # Call prepare_invoke
    result = await conductor.prepare_invoke(conversation)

    # Since init flags are disabled, the field1 should remain empty and not be updated
    # from either longterm memory or previous prompt argument
    assert result.prompt_argument.field1 == ""
    assert result.prompt_argument != conversation.prompt_argument


# ----- Tests for Integration of BaseAgent and ConductorAgent Functionality -----


@pytest.mark.asyncio
async def test_integration_rule_based_routing_with_tools(
    conductor_agent_with_tools, conductor_agent_with_rules, mock_conversation
):
    """Test the integration of rule-based routing with tool handling"""
    # Combine tools and rules in one agent
    conductor = create_conductor_agent(
        mock_dispatcher=conductor_agent_with_tools.dispatcher,
        mock_llm_client=conductor_agent_with_tools.llm_client,
        mock_prompt=conductor_agent_with_tools.prompt,
        tools=conductor_agent_with_tools.tools,
        tool_executor=conductor_agent_with_tools.tool_executor,
        rules=conductor_agent_with_rules.rules,
    )

    # Set up the agent to use rule-based routing
    conductor.prompt.init_prompt = lambda agent_name, conversation, message_type, **kwargs: conversation

    # Make all rules return false to force LLM fallback with tool use
    for rule in conductor.rules:
        rule.condition = rule_always_false

    # Setup LLM to return a tool call first, then a regular response
    tool_calls_made = 0

    async def mock_generate_with_tools(conversation, tools=None, message_type=None):
        nonlocal tool_calls_made
        if tool_calls_made == 0:
            tool_calls_made += 1
            return conversation.append(
                ChatMessage(
                    role=Role.ASSISTANT,
                    content=None,
                    name=conductor.name,
                    tool_calls=[
                        {
                            "tool_call_id": "test_id",
                            "function_name": "test_tool",
                            "arguments": '{"test_param": "test_value"}',
                        }
                    ],
                )
            )
        else:
            return conversation.append(
                ChatMessage(role=Role.ASSISTANT, content='{"next_agent": "agent1"}', name=conductor.name)
            )

    # Setup prompt to return a valid prompt argument after tool use
    conductor.llm_client.generate = AsyncMock(side_effect=mock_generate_with_tools)
    prompt_arg = MockPromptArgument(next_agent="agent1")
    conductor.prompt.parse_user_prompt.return_value = prompt_arg
    conductor.prompt.is_final = Mock(return_value=True)

    # Invoke the agent
    result = await conductor.invoke(mock_conversation, MessageType.CONDUCTOR)

    # Verify that tool was called AND routing happened
    conductor.tool_executor.execute_tool.assert_called_once()
    assert result.prompt_argument.next_agent == "agent1"
