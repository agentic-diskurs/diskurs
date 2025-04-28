from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass
from diskurs.entities import InputField, OutputField, LongtermMemory, PromptArgument

import pytest

from .conftest import MyExtendedPromptArgument
from diskurs import Conversation, ToolExecutor
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


@pytest.mark.asyncio
async def test_invoke_basic_functionality():
    """
    Test that the HeuristicAgent's invoke method performs its core responsibilities:
    1. Initialize the prompt
    2. Execute the heuristic sequence
    3. Render and append a template when render_prompt=True
    4. Set the active agent and return the updated conversation
    """

    # Create a minimal implementation of HeuristicPrompt
    class SimpleHeuristicPrompt(HeuristicPrompt):
        def __init__(self):
            self.initialize_prompt_called = False
            self.heuristic_sequence_called = False
            self.render_template_called = False

        def initialize_prompt(self, agent_name, conversation, **kwargs):
            self.initialize_prompt_called = True
            # Return conversation with active_agent set to show it was processed
            return conversation.update(active_agent=agent_name)

        async def heuristic_sequence(self, conversation, call_tool=None, **kwargs):
            self.heuristic_sequence_called = True
            # Simulate adding a message during heuristic sequence
            return conversation.append(ChatMessage(role=Role.ASSISTANT, content="Response from heuristic sequence"))

        def render_user_template(self, agent_name, prompt_args, message_type):
            self.render_template_called = True
            return ChatMessage(role=Role.USER, content="Rendered template content", name=agent_name, type=message_type)

        def create_prompt_argument(self, **kwargs):
            # Simple implementation just to satisfy the interface
            return {}

    # Create a minimal conversation
    from diskurs import ImmutableConversation

    conversation = ImmutableConversation().append(ChatMessage(role=Role.USER, content="Initial user message"))

    # Create the prompt and agent
    prompt = SimpleHeuristicPrompt()
    agent = HeuristicAgent(name="test-agent", prompt=prompt, render_prompt=True)

    # Invoke the agent
    result = await agent.invoke(conversation)

    # Verify core functionality
    assert prompt.initialize_prompt_called, "initialize_prompt should be called"
    assert prompt.heuristic_sequence_called, "heuristic_sequence should be called"
    assert prompt.render_template_called, "render_user_template should be called when render_prompt=True"

    # Verify conversation updates
    assert result.active_agent == "test-agent", "The agent should set itself as the active agent"
    assert (
        len(result.chat) == 3
    ), "Conversation should have 3 messages (original + heuristic sequence + rendered template)"

    # Verify the messages in the conversation
    assert result.chat[0].content == "Initial user message"
    assert result.chat[1].content == "Response from heuristic sequence"
    assert result.chat[2].content == "Rendered template content"


@pytest.mark.asyncio
async def test_invoke_without_rendering():
    """Test that the HeuristicAgent doesn't render a template when render_prompt=False"""

    # Create a minimal implementation of HeuristicPrompt
    class SimpleHeuristicPrompt(HeuristicPrompt):
        def __init__(self):
            self.render_template_called = False

        def initialize_prompt(self, agent_name, conversation, **kwargs):
            return conversation

        async def heuristic_sequence(self, conversation, call_tool=None, **kwargs):
            return conversation

        def render_user_template(self, agent_name, prompt_args, message_type):
            self.render_template_called = True
            return ChatMessage(role=Role.USER, content="This should not be added")

        def create_prompt_argument(self, **kwargs):
            return {}

    # Create a minimal conversation
    from diskurs import ImmutableConversation

    conversation = ImmutableConversation().append(ChatMessage(role=Role.USER, content="Initial user message"))

    # Create an agent with render_prompt=False
    prompt = SimpleHeuristicPrompt()
    agent = HeuristicAgent(name="test-agent", prompt=prompt, render_prompt=False)

    # Invoke the agent
    result = await agent.invoke(conversation)

    # Verify that render_user_template was not called
    assert not prompt.render_template_called, "render_user_template should not be called when render_prompt=False"

    # Verify conversation wasn't changed
    assert len(result.chat) == 1, "No additional messages should be added when render_prompt=False"


@pytest.mark.asyncio
async def test_invoke_with_tool_executor():
    """Test that the HeuristicAgent properly passes the tool executor to the heuristic sequence"""

    # Create a minimal implementation of HeuristicPrompt
    class SimpleHeuristicPrompt(HeuristicPrompt):
        def __init__(self):
            self.tool_executor_passed = None

        def initialize_prompt(self, agent_name, conversation, **kwargs):
            return conversation

        async def heuristic_sequence(self, conversation, call_tool=None, **kwargs):
            self.tool_executor_passed = call_tool
            return conversation

        def render_user_template(self, agent_name, prompt_args, message_type):
            return ChatMessage(role=Role.USER, content="Template")

        def create_prompt_argument(self, **kwargs):
            return {}

    # Create test objects
    from diskurs import ImmutableConversation

    conversation = ImmutableConversation()

    # Create a simple mock tool executor
    async def mock_call_tool(tool_name, **kwargs):
        return {"result": f"Called {tool_name}"}

    tool_executor = MagicMock(spec=ToolExecutor)
    tool_executor.call_tool = mock_call_tool

    # Create the agent with the tool executor
    prompt = SimpleHeuristicPrompt()
    agent = HeuristicAgent(name="test-agent", prompt=prompt, tool_executor=tool_executor, render_prompt=False)

    # Invoke the agent
    await agent.invoke(conversation)

    # Verify that the tool executor was passed to heuristic_sequence
    assert prompt.tool_executor_passed is not None, "Tool executor should be passed to heuristic_sequence"
    assert prompt.tool_executor_passed == tool_executor.call_tool, "Correct tool executor function should be passed"


@pytest.mark.asyncio
async def test_invoke_no_executor():
    # Setup
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    prompt_arg = MagicMock()

    # Set up mock for initialize_prompt
    initialized_conversation = MagicMock(spec=Conversation)
    initialized_conversation.prompt_argument = prompt_arg
    prompt.initialize_prompt.return_value = initialized_conversation

    # Create a mock result for the heuristic_sequence
    mock_result = MagicMock(spec=Conversation)
    mock_result.prompt_argument = prompt_arg
    prompt.heuristic_sequence = AsyncMock(return_value=mock_result)

    # Return a proper ChatMessage for render_user_template
    prompt.render_user_template.return_value = ChatMessage(
        role=Role.USER, content="rendered_template", type=MessageType.CONVERSATION
    )

    agent = HeuristicAgent(
        name="test_agent",
        prompt=prompt,
        llm_client=None,
        topics=["conductor_name"],
        init_prompt_arguments_with_longterm_memory=True,
        render_prompt=True,
    )

    result = await agent.invoke(conversation)

    # Assert initialize_prompt was called with the right parameters
    prompt.initialize_prompt.assert_called_once()
    init_prompt_args = prompt.initialize_prompt.call_args[1]
    assert init_prompt_args["agent_name"] == "test_agent"
    assert init_prompt_args["conversation"] == conversation
    assert init_prompt_args["init_from_longterm_memory"] == True
    # No longer checking for init_from_previous_agent as it's no longer used with the new longterm memory model

    # Assert heuristic_sequence was called with the right parameters
    prompt.heuristic_sequence.assert_awaited_once()
    heuristic_args = prompt.heuristic_sequence.call_args[1]
    assert heuristic_args["conversation"] == initialized_conversation
    assert heuristic_args["call_tool"] is None
    assert "llm_client" in heuristic_args

    # Assert render_user_template was called (since render_prompt=True)
    prompt.render_user_template.assert_called_once_with(
        "test_agent",
        prompt_args=mock_result.prompt_argument,
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
    return create_prompt(MyExtendedPromptArgument())


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


# The test_invoke_with_longterm_memory_and_previous_agent test has been removed as it's
# superseded by test_invoke_with_annotated_fields which properly tests the field
# initialization behavior with InputField and OutputField annotations.


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


@dataclass
class AnnotatedLongtermMemory(LongtermMemory):
    """Longterm memory with annotated fields for testing field inheritance"""

    field1: str = "ltm_field1"
    field2: str = "longterm val 2"  # This should be inherited via InputField
    field3: str = "ltm_field3"
    field5: str = "ltm_field5"  # Additional field not in prompt arguments


@dataclass
class AnnotatedSourcePromptArgument(PromptArgument):
    """Source prompt argument with fields for testing field inheritance.
    Note: OutputField annotations don't matter for update() - the method
    only checks if target fields are not InputField or LockedField."""

    field3: str = "user prompt field 3"  # Source field for inheritance
    field4: str = "user prompt field 4"  # Source field for inheritance
    field6: str = "source_field6"  # Field not in target class


@dataclass
class AnnotatedPromptArgument(PromptArgument):
    """Target prompt argument with properly annotated fields"""

    field1: str = "default field1"  # Regular field, not annotated
    field2: InputField[str] = "default field2"  # Should get value from longterm memory
    field3: str = "default field3"  # Regular field, should get value from source
    field4: str = "default field4"  # Regular field, should get value from source
    field5: InputField[str] = "default field5"  # Should get value from longterm memory if present
    field7: OutputField[str] = "output field7"  # Output field not used in this test


@pytest.fixture
def annotated_extended_conversation():
    """Create a conversation with AnnotatedSourcePromptArgument and AnnotatedLongtermMemory"""
    from diskurs import ImmutableConversation

    conversation = ImmutableConversation(
        prompt_argument=AnnotatedSourcePromptArgument(),
        chat=[ChatMessage(role=Role.USER, content="Initial user message", name="Alice")],
        longterm_memory=AnnotatedLongtermMemory(),  # Changed from dictionary to single object
        active_agent="my_conductor",
        conversation_id="test_conversation_id",
    )
    return conversation


@pytest.fixture
def annotated_heuristic_prompt():
    """Create a minimal HeuristicPrompt implementation for testing annotations"""
    # Create a simple template for the user prompt
    from jinja2 import Template

    user_template = Template("This is a user template with {{ field1 }}, {{ field2 }}, {{ field3 }}, and {{ field4 }}")

    # Create a simple heuristic sequence function that just returns the conversation unchanged
    async def simple_heuristic_sequence(conversation, call_tool=None, llm_client=None):
        return conversation

    # Create a real HeuristicPrompt with minimal dependencies
    prompt = HeuristicPrompt(
        prompt_argument_class=AnnotatedPromptArgument,
        heuristic_sequence=simple_heuristic_sequence,
        user_template=user_template,
        agent_description="Test heuristic agent",
    )

    return prompt


@pytest.fixture
def annotated_heuristic_agent(annotated_heuristic_prompt):
    """Create a HeuristicAgent with the annotated prompt"""
    return HeuristicAgent.create(
        name="test_agent",
        prompt=annotated_heuristic_prompt,
        topics=[CONDUCTOR_NAME],
        render_prompt=True,
    )


@pytest.mark.asyncio
async def test_invoke_with_annotated_fields():
    """
    Test that field annotations correctly control field inheritance during initialization

    This test directly verifies the behavior of the field inheritance methods on PromptArgument.
    We use init() to transfer values between different types of prompt arguments.
    """
    from diskurs import ImmutableConversation

    # Create a conversation with our test prompt argument and longterm memory
    conversation = ImmutableConversation(
        prompt_argument=AnnotatedSourcePromptArgument(),
        longterm_memory=AnnotatedLongtermMemory(),  # Use a single object instead of a dictionary
    )

    # Create a target prompt argument
    target = AnnotatedPromptArgument()

    # 1. Apply values from longterm memory using the real .init() method
    ltm = conversation.longterm_memory  # Get the global longterm memory directly
    # This should only initialize InputField-annotated fields from longterm memory
    target = target.init(ltm)

    # 2. Apply values from source prompt argument using the real .init() method again
    source = conversation.prompt_argument

    # According to guidance, we use init() for transferring between different types
    target = target.init(source)

    # Now verify the behavior of these methods matches our expectations

    # Regular field without annotation should keep its default value
    assert target.field1 == "default field1", "Regular field should keep its default value"

    # Field with InputField annotation should get value from longterm memory
    assert target.field2 == "longterm val 2", "InputField should get value from longterm memory"

    # Regular field should NOT get value from source prompt argument when using init()
    # init() only transfers InputField-annotated fields
    assert target.field3 == "default field3", "Field should keep default value when using init()"
    assert target.field4 == "default field4", "Field should keep default value when using init()"

    # InputField should get value from longterm memory
    assert target.field5 == "ltm_field5", "InputField should get value from longterm memory"

    # OutputField in target should keep its default value
    assert target.field7 == "output field7", "OutputField should keep its default value"


@pytest.mark.asyncio
async def test_invoke_respects_initialization_flags(annotated_heuristic_agent, annotated_extended_conversation):
    """
    Test that the HeuristicAgent respects initialization flags for longterm memory and previous agent.
    """
    # Configure the agent to not initialize from longterm memory or previous agent
    annotated_heuristic_agent.init_prompt_arguments_with_longterm_memory = False
    annotated_heuristic_agent.init_prompt_arguments_with_previous_agent = False

    # Add messages to the conversation
    annotated_extended_conversation = annotated_extended_conversation.append(
        message=ChatMessage(
            content="{'next_agent': 'test_agent'}",
            role=Role.USER,
            type=MessageType.CONDUCTOR,
            name="my_conductor",
        )
    )

    # Invoke the agent
    result = await annotated_heuristic_agent.invoke(annotated_extended_conversation)

    # All fields should have their default values since initialization is disabled
    prompt_arg = result.prompt_argument
    assert prompt_arg.field1 == "default field1", "Field should keep default when init is disabled"
    assert prompt_arg.field2 == "default field2", "InputField should keep default when longterm init is disabled"
    assert prompt_arg.field3 == "default field3", "Field should keep default when previous agent init is disabled"
    assert prompt_arg.field4 == "default field4", "Field should keep default when previous agent init is disabled"
    assert prompt_arg.field5 == "default field5", "InputField should keep default when longterm init is disabled"
    assert prompt_arg.field7 == "output field7", "OutputField should keep its default value"


@pytest.mark.asyncio
async def test_prompt_argument_update_method():
    """
    Test that PromptArgument.update() correctly transfers values between same-type objects
    according to field annotations.

    This test verifies:
    1. Values from source are transferred to non-InputField, non-LockedField fields in target
    2. InputField values in target are not overwritten by update()
    3. LockedField values in target are not overwritten by update()
    """
    from dataclasses import dataclass
    from typing import get_type_hints
    from diskurs.entities import InputField, OutputField, LockedField, PromptArgument

    # Define a test prompt argument class with various field annotations
    @dataclass
    class TestPromptArgument(PromptArgument):
        regular_field: str = "default_regular"
        input_field: InputField[str] = "default_input"
        output_field: OutputField[str] = "default_output"
        locked_field: LockedField[str] = "default_locked"

    # Create source and target instances (same type)
    source = TestPromptArgument(
        regular_field="source_regular",
        input_field="source_input",
        output_field="source_output",
        locked_field="source_locked",
    )

    target = TestPromptArgument()

    # Apply update from source to target
    updated_target = target.update(source)

    # Regular fields should be updated
    assert updated_target.regular_field == "source_regular", "Regular fields should be updated"

    # OutputField should be updated (it's not InputField or LockedField)
    assert updated_target.output_field == "source_output", "OutputField should be updated"

    # InputField should NOT be updated
    assert updated_target.input_field == "default_input", "InputField should not be updated"

    # LockedField should NOT be updated
    assert updated_target.locked_field == "default_locked", "LockedField should not be updated"
