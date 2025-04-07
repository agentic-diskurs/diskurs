from dataclasses import dataclass
from typing import Optional
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from diskurs import ImmutableConversation, Conversation, PromptValidationError
from diskurs.conductor_agent import ConductorAgent, validate_finalization
from diskurs.entities import (
    ChatMessage,
    Role,
    MessageType,
    LongtermMemory,
    PromptArgument,
    RoutingRule,
)
from diskurs.prompt import (
    DefaultConductorUserPromptArgument,
)
from diskurs.protocols import (
    LLMClient,
    ConversationDispatcher,
    ConductorPrompt,
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
class MyUserPromptArgument(PromptArgument):
    field1: Optional[str] = ""
    field2: Optional[str] = ""
    field3: Optional[str] = ""
    next_agent: Optional[str] = ""


@dataclass
class TestUserPromptArgument(PromptArgument):
    content: str = ""
    next_agent: str = ""


@dataclass
class DefaultCanFinalizeUserPromptArgument(PromptArgument):
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
    prompt.create_system_prompt_argument.return_value = Mock()
    prompt.create_prompt_argument.return_value = Mock()

    finalize_mock = AsyncMock(side_effect=finalize)
    can_finalize_mock = Mock(side_effect=can_finalize)

    prompt.finalize = finalize_mock
    prompt.can_finalize = can_finalize_mock
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
    return ImmutableConversation(prompt_argument=TestUserPromptArgument()).append(
        ChatMessage(role=Role.USER, content="test message", name="user")
    )


@pytest.fixture
def mock_conversation_with_keyword():
    return ImmutableConversation(prompt_argument=TestUserPromptArgument()).append(
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
):
    agent = ConductorAgent(
        name="conductor",
        prompt=mock_prompt,
        llm_client=mock_llm_client,
        topics=["agent1", "agent2", "agent3"],
        agent_descriptions={},
        finalizer_name=finalizer,
        dispatcher=mock_dispatcher,
        max_trials=5,
        max_dispatches=5,
        supervisor=supervisor,
        rules=rules,
        fallback_to_llm=fallback_to_llm,
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


# ----- Tests for Basic Agent Functionality -----


def test_update_longterm_memory(conductor_agent):
    conversation = ImmutableConversation(
        prompt_argument=MyUserPromptArgument(field1="value1", field2="value2", field3="value3")
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
        prompt_argument=MyUserPromptArgument(field1="value1", field2="value2", field3="value3")
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
    conversation = ImmutableConversation(prompt_argument=MyUserPromptArgument(field1="new_value1"))
    longterm_memory = MyLongTermMemory(field1="existing_value1")

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
    conversation.update_agent_longterm_memory = Mock(return_value=conversation)
    conductor_agent.prompt.init_longterm_memory = Mock()

    updated_conversation = conductor_agent.create_or_update_longterm_memory(conversation, overwrite=True)

    assert longterm_memory.field1 == "new_value1"  # Should be overwritten
    conversation.update_agent_longterm_memory.assert_called_once_with(
        agent_name=conductor_agent.name, longterm_memory=longterm_memory
    )


@pytest.mark.asyncio
async def test_process_conversation_updates_longterm_memory(conductor_agent):
    conversation = ImmutableConversation(
        prompt_argument=MyUserPromptArgument(field1="value1", field2="value2", field3="value3")
    ).append(ChatMessage(Role.ASSISTANT, content='{"next_agent": "agent1"}'))
    longterm_memory = MyLongTermMemory()

    with patch.object(
        ImmutableConversation,
        "update_agent_longterm_memory",
        return_value=conversation,
    ) as mock_update_longterm:
        conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)
        conductor_agent.prompt.init_longterm_memory.return_value = longterm_memory
        conductor_agent.prompt.can_finalize.return_value = False
        conductor_agent.prompt.create_system_prompt_argument.return_value = MyUserPromptArgument(
            field1="sys1", field2="sys2"
        )

        prompt_argument = MyUserPromptArgument(field1="value1", field2="value2")
        conductor_agent.prompt.create_prompt_argument.return_value = prompt_argument

        conductor_agent.generate_validated_response = AsyncMock(return_value=conversation)

        await conductor_agent.process_conversation(conversation)

        assert longterm_memory.field1 == "value1"
        assert longterm_memory.field2 == "value2"
        mock_update_longterm.assert_called_once()


@pytest.mark.asyncio
async def test_process_conversation_finalize(conductor_agent):
    conversation = ImmutableConversation(prompt_argument=MyUserPromptArgument())
    longterm_memory = MyLongTermMemory()

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)

    conductor_agent.prompt.can_finalize.return_value = True
    conductor_agent.prompt.finalize.return_value = {"result": "final result"}
    conductor_agent.dispatcher.finalize = Mock()

    conductor_agent.generate_validated_response = Mock()

    await conductor_agent.process_conversation(conversation)

    assert await conversation.final_result == {"result": "final result"}
    conductor_agent.generate_validated_response.assert_not_called()


@pytest.mark.asyncio
async def test_max_dispatches(conductor_cannot_finalize):
    conductor_cannot_finalize.n_dispatches = 49
    conversation = ImmutableConversation(prompt_argument=MyUserPromptArgument()).append(
        ChatMessage(Role.ASSISTANT, content='{"next_agent": "agent1"}')
    )
    longterm_memory = MyLongTermMemory()

    conversation = conversation.update_agent_longterm_memory(
        agent_name=conductor_cannot_finalize.name,
        longterm_memory=longterm_memory,
    )

    await conductor_cannot_finalize.process_conversation(conversation)

    conductor_cannot_finalize.prompt.fail.assert_called_once_with(longterm_memory)


@pytest.mark.asyncio
async def test_conductor_agent_valid_next_agent(conductor_cannot_finalize, mock_llm_client):
    # Setup initial conversation with longterm memory
    conversation = ImmutableConversation(prompt_argument=MyUserPromptArgument())
    longterm_memory = MyLongTermMemory()
    conversation = conversation.update_agent_longterm_memory(
        agent_name=conductor_cannot_finalize.name, longterm_memory=longterm_memory
    )

    async def stub_generate_validated_response(conversation, message_type=None, tools=None):
        return conversation.append(
            ChatMessage(role=Role.ASSISTANT, content='{"next_agent": "agent1"}', type=MessageType.CONDUCTOR)
        )

    mock_llm_client.generate = AsyncMock(side_effect=stub_generate_validated_response)

    parsed_prompt_argument = DefaultConductorUserPromptArgument(next_agent="agent1")
    conductor_cannot_finalize.prompt.parse_user_prompt.return_value = parsed_prompt_argument
    conductor_cannot_finalize.prompt.can_finalize.return_value = False
    conductor_cannot_finalize.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    await conductor_cannot_finalize.process_conversation(conversation)

    conductor_cannot_finalize.dispatcher.publish.assert_called_once_with(topic="agent1", conversation=ANY)


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
async def test_process_conversation_finalize_with_agent_calls_dispatcher(
    conductor_agent_with_finalizer,
):
    conversation = ImmutableConversation(prompt_argument=MyUserPromptArgument())
    longterm_memory = MyLongTermMemory()

    conversation.get_agent_longterm_memory = Mock(return_value=longterm_memory)

    conductor_agent_with_finalizer.add_routing_message_to_chat = AsyncMock(return_value=conversation)
    conductor_agent_with_finalizer.generate_validated_response = Mock()

    await conductor_agent_with_finalizer.process_conversation(conversation)

    conductor_agent_with_finalizer.add_routing_message_to_chat.assert_called_once_with(ANY, FINALIZER_NAME)

    conductor_agent_with_finalizer.dispatcher.publish_final.assert_called_once_with(
        topic=FINALIZER_NAME, conversation=ANY
    )


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

    assert await mock_conversation.final_result == {"result": "final result"}
    conductor_agent_with_finalizer_function.dispatcher.publish.assert_not_called()


@pytest.mark.asyncio
async def test_process_conversation_calls_can_finalize(mock_conversation, conductor_agent):
    conductor_agent.finalize = AsyncMock()

    await conductor_agent.process_conversation(mock_conversation)
    conductor_agent.finalize.assert_called_once()


@pytest.mark.asyncio
async def test_can_finalize(mock_conversation, conductor_agent):
    conductor_agent.can_finalize_name = "Finalizer_Agent"
    conductor_agent.prompt.can_finalize.return_value = True
    conductor_agent.finalize = AsyncMock()

    parsed_prompt_argument = DefaultCanFinalizeUserPromptArgument(can_finalize=True)
    conductor_agent.prompt.parse_user_prompt.return_value = parsed_prompt_argument

    await conductor_agent.process_conversation(mock_conversation)
    conductor_agent.finalize.assert_called_once()


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
            return TestUserPromptArgument(next_agent=kwargs["next_agent"])
        return TestUserPromptArgument()

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

    conductor_agent_with_rules.llm_client.generate = AsyncMock(side_effect=mock_llm_generate)

    # Mock the prompt parsing to return a valid prompt argument
    conductor_agent_with_rules.prompt.parse_user_prompt.return_value = DefaultConductorUserPromptArgument(
        next_agent="llm_agent"
    )
    conductor_agent_with_rules.prompt.is_final.return_value = True
    # Call invoke
    result = await conductor_agent_with_rules.invoke(mock_conversation, MessageType.CONDUCTOR)

    # Verify LLM was used and next_agent is set correctly
    conductor_agent_with_rules.llm_client.generate.assert_called_once()
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
        updated = conversation.update(prompt_argument=TestUserPromptArgument(next_agent="agent1"))
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
    clean_conversation = mock_conversation.update(prompt_argument=TestUserPromptArgument(next_agent=None))
    conductor_agent_rules_only.prompt.init_prompt = (
        lambda agent_name, conversation, message_type, **kwargs: conversation
    )

    # Also mock prompt.create_prompt_argument to ensure it returns a clean object
    conductor_agent_rules_only.prompt.create_prompt_argument = Mock(
        return_value=TestUserPromptArgument(next_agent=None)
    )

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
    prompt_arg = DefaultConductorUserPromptArgument(next_agent=None)
    mock_conversation = mock_conversation.update(prompt_argument=prompt_arg)

    # Create a new user prompt argument to return
    updated_prompt_arg = DefaultConductorUserPromptArgument(next_agent="agent1")
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
            return TestUserPromptArgument(next_agent=kwargs["next_agent"])
        return TestUserPromptArgument()

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
    async def mock_generate_validated_response(conversation, message_type=None):
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
