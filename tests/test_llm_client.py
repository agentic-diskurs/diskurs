from unittest.mock import AsyncMock, MagicMock
from unittest.mock import Mock

import pytest
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from diskurs import ImmutableConversation
from diskurs.entities import ChatMessage, PromptArgument, ToolDescription
from diskurs.entities import Role
from diskurs.llm_client import OpenAILLMClient
from diskurs.protocols import LLMClient, Conversation

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

    system_prompt_arg = PromptArgument()
    user_prompt_arg = PromptArgument()

    return ImmutableConversation(system_prompt, user_prompt, system_prompt_arg, user_prompt_arg)


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
        chat=[
            ChatMessage(role="system", content=long_message),
            ChatMessage(role="user", content=long_message),
            ChatMessage(role="assistant", content=long_message),
        ],
        metadata={"ticket_id": "123"},
        user_prompt_argument=PromptArgument(),
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
        chat=chat_history,
        system_prompt=ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        user_prompt=ChatMessage(role=Role.USER, content="What is the capital of france?"),
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
