from unittest.mock import Mock

import pytest

from conftest import conversation
from diskurs.protocols import ToolExecutor, HeuristicPrompt, ConversationDispatcher
from test_files.heuristic_agent_test_files.prompt import MyHeuristicPromptArgument


@pytest.fixture
def tool_executor():
    executor = Mock(spec=ToolExecutor)
    return executor


@pytest.fixture
def heuristic_prompt(conversation):
    prompt = Mock(spec=HeuristicPrompt)

    # Create an instance of MyHeuristicPromptArgument to be returned
    prompt_arg_instance = MyHeuristicPromptArgument()

    # Configure create_user_prompt_argument to return the specific instance
    prompt.create_user_prompt_argument.return_value = prompt_arg_instance

    # Side effect function for heuristic_sequence
    def heuristic_sequence_side_effect(user_prompt_argument, metadata, call_tool):
        # Assert that heuristic_sequence is called with the correct Conversation instance
        assert user_prompt_argument == conversation.user_prompt_argument, "Expected correct user_prompt_argument"
        assert metadata == conversation.metadata, "Expected correct metadata"
        # Return the specific instance
        return prompt_arg_instance

    # Set the heuristic_sequence to the side effect function
    prompt.heuristic_sequence.side_effect = heuristic_sequence_side_effect

    return prompt


def dispatcher():
    return Mock(spec=ConversationDispatcher)


def test_heuristic_prompt(heuristic_prompt, conversation):
    result = heuristic_prompt.create_user_prompt_argument()
    assert isinstance(result, MyHeuristicPromptArgument)

    # Test heuristic_sequence behavior and argument validation
    result = heuristic_prompt.heuristic_sequence(
        user_prompt_argument=conversation.user_prompt_argument,
        metadata=conversation.metadata,
        call_tool=lambda x: x,  # Mock or real function as needed
    )
    assert isinstance(result, MyHeuristicPromptArgument)
