from unittest.mock import Mock

import pytest
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from diskurs import ImmutableConversation
from diskurs.entities import ChatMessage, PromptArgument, ToolDescription
from diskurs.entities import Role
from diskurs.llm_client import OpenAILLMClient

load_dotenv()

example_messages = [
    {
        "role": "system",
        "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "New synergies will help drive top-line growth.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Things working well together will increase revenue.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Let's talk later when we're less busy about how to do better.",
    },
    {
        "role": "user",
        "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
]

example_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The customer's order ID.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    }
]

example_messages_with_tools = {"tools": example_tools, "model": "gpt-4-0613", "messages": example_messages}

tool_call = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_62136354",
            "type": "function",
            "function": {"arguments": "{'order_id': 'order_12345'}", "name": "get_delivery_date"},
        }
    ],
}


@pytest.fixture
def llm_client():
    client = Mock(spec=OpenAI)
    model = "gpt-4-0613"
    tokenizer = tiktoken.encoding_for_model(model)
    llm_client = OpenAILLMClient(client=client, model=model, tokenizer=tokenizer, max_tokens=100)
    return llm_client


@pytest.fixture
def init_conversation():
    system_prompt = ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant.")
    user_prompt = ChatMessage(role=Role.USER, content="What is the capital of france?")

    prompt_arg = PromptArgument()

    return ImmutableConversation(system_prompt, user_prompt, prompt_arg)


def test_format_for_llm(init_conversation, llm_client):
    formatted = llm_client.format_conversation_for_llm(init_conversation)

    assert isinstance(formatted, dict)

    assert "messages" in formatted
    assert isinstance(formatted["messages"], list)

    assert all(
        isinstance(message, dict)
        and "role" in message
        and message["role"] in ["user", "assistant", "system"]
        and "content" in message
        and isinstance(message["content"], str)
        for message in formatted["messages"]
    )


def test_count_tokens_in_conversation_tool_calls(llm_client):
    n_tokens = llm_client.count_tokens_in_conversation([tool_call])

    assert n_tokens == 34


def test_count_tokens_in_conversation(llm_client):
    n_tokens = llm_client.count_tokens_in_conversation(example_messages)

    assert n_tokens == 129


def test_count_tokens_of_tool_descriptions(llm_client):
    n_tokens = llm_client.count_tokens_of_tool_descriptions(example_tools)

    assert n_tokens == 72


def test_truncate_chat_history(llm_client):
    truncated_chat = llm_client.truncate_chat_history(messages=example_messages, n_tokens_tool_descriptions=50)

    assert truncated_chat[0] == example_messages[0]
    assert truncated_chat[-1] == example_messages[-1]

    assert len(truncated_chat) < len(example_messages)


def test_truncate_chat_history_with_large_second_message(llm_client):
    large_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Lorem ipsum " * 1000},  # Large message that exceeds token limit
        {"role": "assistant", "content": "How can I help?"},
    ]

    truncated_chat = llm_client.truncate_chat_history(messages=large_messages, n_tokens_tool_descriptions=0)

    # Should keep system prompt and last message
    assert len(truncated_chat) == 2
    assert truncated_chat[0] == large_messages[0]
    assert truncated_chat[-1] == large_messages[-1]


@pytest.fixture
def long_history_conversation():
    # Create conversation with history that exceeds token limit
    long_message = "x" * 10000  # Very long message that will exceed token limit
    conversation = ImmutableConversation(
        prompt_argument=PromptArgument(),
        chat=[
            ChatMessage(role="system", content=long_message),
            ChatMessage(role="user", content=long_message),
            ChatMessage(role="assistant", content=long_message),
        ],
        metadata={"ticket_id": "123"},
    )
    return conversation


def test_truncate_chat_history_with_tools_and_large_content(llm_client):
    # Create a message that with tools would exceed the limit
    large_content = "straw" * 7000  # This plus tools should exceed 8192
    large_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": large_content},
        {"role": "assistant", "content": "How can I help?"},
    ]

    # Add significant tool descriptions
    n_tokens_tools = 1000  # Simulate large tool descriptions
    llm_client.max_tokens = 8192
    truncated_chat = llm_client.truncate_chat_history(
        messages=large_messages, n_tokens_tool_descriptions=n_tokens_tools
    )

    # Calculate total tokens including tools
    total_tokens = llm_client.count_tokens_in_conversation(truncated_chat) + n_tokens_tools

    assert total_tokens < llm_client.max_tokens
    # Ensure we keep system prompt and last message
    assert truncated_chat[0] == large_messages[0]
    assert truncated_chat[-1] == large_messages[-1]


def test_format_conversation_with_multistep_agent_interaction(llm_client):
    # Create a large conversation that simulates multistep agent interaction
    llm_client.max_tokens = 8192
    large_content = (
        "the quick brown fox jumped over the lazy hog" * 500
    )  # Large content that will push us close to limit
    chat_history = [
        ChatMessage(role=Role.USER, content=large_content),
        ChatMessage(role=Role.ASSISTANT, content=large_content),
        ChatMessage(role=Role.USER, content="What should I do next?"),
        ChatMessage(role=Role.TOOL, content=large_content),
    ]

    # Add tool descriptions similar to what MultiStepAgent would use
    tools = [
        ToolDescription(
            name="analyze_data",
            description="Analyze the provided data",
            arguments={
                "data": {"type": "str", "description": "Data to analyze"},
                "format": {"type": "str", "description": "Output format"},
            },
        ),
        ToolDescription(
            name="generate_report",
            description="Generate a report from analysis",
            arguments={
                "analysis": {"type": "str", "description": "Analysis results"},
                "style": {"type": "str", "description": "Report style"},
            },
        ),
    ]

    conversation = ImmutableConversation(
        system_prompt=ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        user_prompt=ChatMessage(role=Role.USER, content="What is the capital of france?"),
        chat=chat_history,
    )

    formatted = llm_client.format_conversation_for_llm(conversation=conversation, tools=tools)

    # Calculate total tokens
    total_tokens = llm_client.count_tokens_in_conversation(
        formatted["messages"]
    ) + llm_client.count_tokens_of_tool_descriptions(formatted.get("tools", []))

    # Verify we stay under limit
    assert total_tokens < llm_client.max_tokens
    # Verify we kept essential messages
    assert formatted["messages"][0]["role"] == "system"  # Keep system prompt
    assert formatted["messages"][-1]["role"] == "user"  # Keep last user


def test_truncate_tool_responses_when_too_long(llm_client):
    from diskurs.entities import ChatMessage, Role

    large_content = "x" * 3000
    llm_client.max_tokens = 200
    messages = [
        ChatMessage(role=Role.TOOL, content=large_content, tool_call_id="tc_1"),
        ChatMessage(role=Role.TOOL, content=large_content, tool_call_id="tc_2"),
    ]

    truncated_messages = llm_client.truncate_tool_responses(messages, total_tokens=3000)

    assert any("[Content truncated due to token limits...]" in msg.content for msg in truncated_messages)
    assert all(len(msg.content) <= len(large_content) for msg in truncated_messages)


def test_truncate_tool_responses_with_large_conversation(llm_client):
    from diskurs.entities import ChatMessage, Role

    llm_client.max_tokens = 100
    large_content = "x" * 3000
    conversation_tokens = 50  # Assume user/system messages use 50 tokens

    # Tool messages that push us over the limit
    messages = [
        ChatMessage(role=Role.TOOL, content=large_content, tool_call_id="tool_call_1"),
        ChatMessage(role=Role.TOOL, content=large_content, tool_call_id="tool_call_2"),
    ]

    # Trigger truncation
    truncated = llm_client.truncate_tool_responses(messages, total_tokens=conversation_tokens)
    assert any("[Content truncated due to token limits...]" in msg.content for msg in truncated)
    assert all(len(msg.content) < len(large_content) for msg in truncated)


def test_truncate_tool_responses_no_truncation_when_under_limit(llm_client):
    from diskurs.entities import ChatMessage, Role

    llm_client.max_tokens = 400
    large_content = "x" * 100
    # Combined tokens < max_tokens => no truncation
    messages = [ChatMessage(role=Role.TOOL, content=large_content, tool_call_id="tc_1")]
    truncated = llm_client.truncate_tool_responses(messages, total_tokens=200)

    # Should remain unchanged
    assert len(truncated) == len(messages)
    assert truncated[0].content == large_content


def test_truncate_tool_responses_partial_truncation_of_multiple_tools(llm_client):
    from diskurs.entities import ChatMessage, Role

    llm_client.max_tokens = 300
    # Increase content size to ensure tokens exceed max_tokens
    content_1 = "x" * 1000
    content_2 = "y" * 1000

    messages = [
        ChatMessage(role=Role.TOOL, content=content_1, tool_call_id="tc_1"),
        ChatMessage(role=Role.TOOL, content=content_2, tool_call_id="tc_2"),
    ]
    truncated = llm_client.truncate_tool_responses(messages, total_tokens=150)

    assert all("[Content truncated due to token limits...]" in msg.content for msg in truncated)
    assert all(len(msg.content) < len(content_1) for msg in truncated)


def test_iterative_truncation_multiple_iterations(llm_client):
    from diskurs.entities import ChatMessage, Role

    # Set a very low max_tokens to force iterative truncation
    llm_client.max_tokens = 150
    original_content_1 = "a" * 2000
    original_content_2 = "b" * 2000

    messages = [
        ChatMessage(role=Role.TOOL, content=original_content_1, tool_call_id="tc_1"),
        ChatMessage(role=Role.TOOL, content=original_content_2, tool_call_id="tc_2"),
    ]
    conversation_tokens = 100  # Simulated token count for conversation

    truncated = llm_client.truncate_tool_responses(messages, total_tokens=conversation_tokens)

    for msg in truncated:
        # Each truncated message should contain the truncation marker
        assert "[Content truncated due to token limits...]" in msg.content
        # And should be shorter than the original message
        assert llm_client.count_tokens(msg.content) < llm_client.count_tokens(original_content_1)

    # Verify that the combined token count (conversation + truncated tool responses) is within the limit
    final_combined = conversation_tokens + sum(llm_client.count_tokens(msg.content) for msg in truncated)
    assert final_combined <= llm_client.max_tokens


def test_format_conversation_truncates_tool_responses(llm_client):
    from diskurs.entities import ChatMessage, Role
    from diskurs import ImmutableConversation

    # Force truncation by setting a low max_tokens
    llm_client.max_tokens = 120
    # Create two tool response messages with large content
    tool_msg1 = ChatMessage(role=Role.TOOL, content="x" * 300, tool_call_id="tool1")
    tool_msg2 = ChatMessage(role=Role.TOOL, content="y" * 300, tool_call_id="tool2")

    chat_history = [
        ChatMessage(role=Role.USER, content="What should I do next?"),
    ]

    conversation = ImmutableConversation(
        system_prompt=ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        user_prompt=[tool_msg1, tool_msg2],
        chat=chat_history,
    )

    formatted = llm_client.format_conversation_for_llm(conversation)

    # Extract tool responses from formatted messages
    tool_responses = [msg for msg in formatted["messages"] if msg["role"] == str(Role.TOOL)]

    # Verify that the tool responses have been truncated (marker is present)
    for msg in tool_responses:
        assert "[Content truncated due to token limits...]" in msg["content"]

    # Ensure that the total token count (without tool descriptions) is within max_tokens
    total_tokens = llm_client.count_tokens_in_conversation(formatted["messages"])
    assert total_tokens < llm_client.max_tokens
