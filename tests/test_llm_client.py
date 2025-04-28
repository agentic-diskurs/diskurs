from unittest.mock import Mock

import pytest
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from diskurs import ImmutableConversation
from diskurs.entities import ChatMessage, Role
from diskurs.entities import PromptArgument, ToolDescription
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
    return ImmutableConversation(system_prompt=system_prompt, chat=[user_prompt], prompt_argument=prompt_arg)


@pytest.fixture
def tool_response_messages():
    """Create a set of tool response messages with varying sizes for testing."""
    tiny = "tiny" * 2  # 8 chars
    small = "small" * 10  # 50 chars
    medium = "medium" * 50  # 300 chars
    large = "large" * 200  # 1000 chars
    huge = "huge" * 1000  # 4000 chars

    return {
        "tiny": ChatMessage(role=Role.TOOL, content=tiny, tool_call_id="tc_tiny"),
        "small": ChatMessage(role=Role.TOOL, content=small, tool_call_id="tc_small"),
        "medium": ChatMessage(role=Role.TOOL, content=medium, tool_call_id="tc_medium"),
        "large": ChatMessage(role=Role.TOOL, content=large, tool_call_id="tc_large"),
        "huge": ChatMessage(role=Role.TOOL, content=huge, tool_call_id="tc_huge"),
        "single_char": [ChatMessage(role=Role.TOOL, content=c, tool_call_id=f"tc_{c}") for c in ["a", "b", "c"]],
    }


# ----------------- Basic Functionality Tests -----------------


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


def test_token_counting(llm_client):
    """Test various token counting functions."""
    # Test token counting for conversation
    conversation_tokens = llm_client.count_tokens_in_conversation(example_messages)
    assert conversation_tokens == 129

    # Test token counting for tool calls
    tool_call_tokens = llm_client.count_tokens_in_conversation([tool_call])
    assert tool_call_tokens == 34

    # Test token counting for tool descriptions
    tool_description_tokens = llm_client.count_tokens_of_tool_descriptions(example_tools)
    assert tool_description_tokens == 72


# ----------------- Chat History Truncation Tests -----------------


def test_truncate_chat_history_basic(llm_client):
    """Test basic chat history truncation preserves system and last message."""
    truncated_chat = llm_client.truncate_chat_history(messages=example_messages, n_tokens_tool_descriptions=50)

    # Always keep system message and last message
    assert truncated_chat[0] == example_messages[0]
    assert truncated_chat[-1] == example_messages[-1]
    assert len(truncated_chat) < len(example_messages)


def test_truncate_chat_history_with_large_content(llm_client):
    """Test chat history truncation with an oversized message."""
    # Create a message that with tools would exceed the limit
    large_content = "straw" * 7000
    large_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": large_content},
        {"role": "assistant", "content": "How can I help?"},
    ]

    # Add significant tool descriptions
    n_tokens_tools = 1000
    llm_client.max_tokens = 8192
    truncated_chat = llm_client.truncate_chat_history(
        messages=large_messages, n_tokens_tool_descriptions=n_tokens_tools
    )

    # Calculate total tokens including tools
    total_tokens = llm_client.count_tokens_in_conversation(truncated_chat) + n_tokens_tools
    assert total_tokens < llm_client.max_tokens
    assert len(truncated_chat) <= 3  # At most original message count
    # Always preserve system and last message
    assert truncated_chat[0] == large_messages[0]
    assert truncated_chat[-1] == large_messages[-1]


def test_format_conversation_handles_token_limits(llm_client):
    """Test that format_conversation_for_llm properly handles token limits."""
    # Set up a conversation that would exceed token limits
    llm_client.max_tokens = 8192
    large_content = "the quick brown fox jumped over the lazy hog" * 500

    chat_history = [
        ChatMessage(role=Role.USER, content=large_content),
        ChatMessage(role=Role.ASSISTANT, content=large_content),
        ChatMessage(role=Role.USER, content="What should I do next?"),
        ChatMessage(role=Role.TOOL, content=large_content),
    ]

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

    # Verify we stay under limit
    total_tokens = llm_client.count_tokens_in_conversation(
        formatted["messages"]
    ) + llm_client.count_tokens_of_tool_descriptions(formatted.get("tools", []))

    assert total_tokens < llm_client.max_tokens
    # Verify we kept essential messages
    assert formatted["messages"][0]["role"] == "system"  # Keep system prompt
    assert formatted["messages"][-1]["role"] == "user"  # Keep last user message


# ----------------- Tool Response Truncation Tests -----------------


def test_truncate_tool_responses_basic(llm_client, tool_response_messages):
    """Test basic truncation of tool responses."""
    # Set a very low token limit to force truncation
    llm_client.max_tokens = 200

    # Test with two large messages
    messages = [tool_response_messages["large"], tool_response_messages["large"]]

    truncated = llm_client.truncate_tool_responses(messages)

    # Check basic truncation properties
    assert len(truncated) == len(messages)
    assert all("[Response truncated]" in msg.content for msg in truncated)
    assert all(llm_client.count_tokens(msg.content) <= llm_client.max_tokens // 2 for msg in truncated)

    # Test with a single message
    single_message = tool_response_messages["large"]
    truncated_single = llm_client.truncate_tool_responses(single_message)

    assert isinstance(truncated_single, list)
    assert len(truncated_single) == 1
    assert "[Response truncated]" in truncated_single[0].content


def test_truncate_tool_responses_in_conversation(llm_client):
    """Test that tool responses are properly truncated within a conversation."""
    # Force truncation with very low token limit
    llm_client.max_tokens = 120

    # Create tool responses that will need truncation
    tool_msg1 = ChatMessage(role=Role.TOOL, content="lorem ipsum dolor sit amet" * 300, tool_call_id="tool1")
    tool_msg2 = ChatMessage(role=Role.TOOL, content="lorem ipsum dolor sit amet" * 300, tool_call_id="tool2")

    conversation = ImmutableConversation(
        system_prompt=ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        user_prompt=[tool_msg1, tool_msg2],
        chat=[ChatMessage(role=Role.USER, content="What should I do next?")],
    )

    formatted = llm_client.format_conversation_for_llm(conversation)

    # Extract tool responses
    tool_responses = [msg for msg in formatted["messages"] if msg["role"] == str(Role.TOOL)]

    # Verify truncation occurred and total is within limits
    assert all("[Response truncated]" in msg["content"] for msg in tool_responses)
    total_tokens = llm_client.count_tokens_in_conversation(formatted["messages"])
    assert total_tokens < llm_client.max_tokens


def test_no_truncation_when_under_limit(llm_client, tool_response_messages):
    """Test that no truncation occurs when content is under the token limit."""
    llm_client.max_tokens = 1000

    # Use small messages that shouldn't need truncation
    messages = [tool_response_messages["small"]]

    truncated = llm_client.truncate_tool_responses(messages)

    # Content should be unchanged
    assert truncated[0].content == tool_response_messages["small"].content
    assert "[Response truncated]" not in truncated[0].content


# ----------------- Water Filling Algorithm Tests -----------------


def test_water_filling_algorithm_basic_scenarios(llm_client, tool_response_messages):
    """Test the water filling algorithm across different basic scenarios."""

    # Test Case 1: Equal length messages
    equal_length = [tool_response_messages["medium"], tool_response_messages["medium"]]
    token_counts = [llm_client.count_tokens(msg.content) for msg in equal_length]
    responses_with_counts = list(zip(equal_length, token_counts))

    # Allow only half of tokens
    total_allowed = sum(token_counts) // 2
    truncated_equal = llm_client.water_filling_truncate_responses(responses_with_counts, total_allowed)

    # Both should be truncated similarly
    truncated_tokens = [
        llm_client.count_tokens(msg.content.split("[Response truncated]")[0]) for msg in truncated_equal
    ]
    assert abs(truncated_tokens[0] - truncated_tokens[1]) <= 1
    assert sum(truncated_tokens) <= total_allowed

    # Test Case 2: Varying length messages (short, medium, long)
    varying_length = [
        tool_response_messages["small"],
        tool_response_messages["medium"],
        tool_response_messages["large"],
    ]
    token_counts = [llm_client.count_tokens(msg.content) for msg in varying_length]
    responses_with_counts = list(zip(varying_length, token_counts))

    # Allow 40% of tokens
    total_allowed = int(sum(token_counts) * 0.4)
    truncated_varying = llm_client.water_filling_truncate_responses(responses_with_counts, total_allowed)

    # Verify waterfall behavior (shorter preserved more)
    truncated_tokens = [
        (
            llm_client.count_tokens(msg.content.split("[Response truncated]")[0])
            if "[Response truncated]" in msg.content
            else llm_client.count_tokens(msg.content)
        )
        for msg in truncated_varying
    ]

    # Short message preserved better than long message
    assert truncated_tokens[0] / token_counts[0] > truncated_tokens[2] / token_counts[2]
    assert sum(truncated_tokens) <= total_allowed


def test_water_filling_tiny_message_preservation(llm_client, tool_response_messages):
    """Test that the water filling algorithm preserves tiny messages intact."""

    # Test with extreme size differences
    messages = [
        tool_response_messages["huge"],
        tool_response_messages["large"],
        tool_response_messages["small"],
        tool_response_messages["tiny"],
    ]

    token_counts = [llm_client.count_tokens(msg.content) for msg in messages]
    responses_with_counts = list(zip(messages, token_counts))

    # Force significant truncation (20% of total)
    total_allowed = int(sum(token_counts) * 0.2)
    truncated = llm_client.water_filling_truncate_responses(responses_with_counts, total_allowed)

    # The tiny message should remain intact
    assert "[Response truncated]" not in truncated[3].content
    assert truncated[3].content == tool_response_messages["tiny"].content

    # Huge message should be heavily truncated
    assert "[Response truncated]" in truncated[0].content

    # Check total token count is within limits
    truncated_tokens = [
        (
            llm_client.count_tokens(msg.content.split("[Response truncated]")[0])
            if "[Response truncated]" in msg.content
            else llm_client.count_tokens(msg.content)
        )
        for msg in truncated
    ]
    assert sum(truncated_tokens) <= total_allowed


def test_water_filling_edge_cases(llm_client, tool_response_messages):
    """Test edge cases in the water filling algorithm."""

    # Case 1: Single character messages should never be truncated
    messages = [tool_response_messages["large"], *tool_response_messages["single_char"]]

    token_counts = [llm_client.count_tokens(msg.content) for msg in messages]
    responses_with_counts = list(zip(messages, token_counts))

    # Force heavy truncation (just 25% of total)
    total_allowed = int(sum(token_counts) * 0.25)
    truncated = llm_client.water_filling_truncate_responses(responses_with_counts, total_allowed)

    # Single character messages should remain intact
    for i in range(1, len(truncated)):
        assert "[Response truncated]" not in truncated[i].content
        assert truncated[i].content == messages[i].content

    # Large message should be truncated
    assert "[Response truncated]" in truncated[0].content

    # Case 2: Minimal truncation needed - smallest message should be preserved
    messages = [
        tool_response_messages["medium"],
        tool_response_messages["small"],
        tool_response_messages["tiny"],
    ]

    token_counts = [llm_client.count_tokens(msg.content) for msg in messages]
    responses_with_counts = list(zip(messages, token_counts))

    # Just slightly below total (95%)
    total_allowed = int(sum(token_counts) * 0.95)
    truncated = llm_client.water_filling_truncate_responses(responses_with_counts, total_allowed)

    # Tiny message should be preserved
    assert "[Response truncated]" not in truncated[2].content

    # Total tokens should be within limit
    truncated_tokens = [
        (
            llm_client.count_tokens(msg.content.split("[Response truncated]")[0])
            if "[Response truncated]" in msg.content
            else llm_client.count_tokens(msg.content)
        )
        for msg in truncated
    ]
    assert sum(truncated_tokens) <= total_allowed


def test_water_filling_real_world_example(llm_client):
    """Test water filling with realistic tool responses."""

    # Create realistic tool responses
    proxy_logs = """Proxy Host: mu
Timestamp: 2025-04-09 15:08:09
Request Details
Client IP: 213.156.236.175 | Username: - | Policy Group: default | URL: 20067.acme.bigcorp:443
Response Details:
Result: Unknown Domain Name | HTTP Response: 200 OK

Proxy Host: mu
Timestamp: 2025-04-09 15:08:09
Request Details
Client IP: 213.156.236.175 | Username: - | Policy Group: default | URL: https://20067.acme.bigcorp/
Response Details:
Result: Unknown Domain Name | HTTP Response: 503 Service Unavailable"""

    url_info = """URL: https://example.com
Status: Accessible
Response time: 230ms
Content type: text/html
Server: nginx/1.18.0
Redirects: None
Security headers:
- X-Content-Type-Options: nosniff
- Strict-Transport-Security: max-age=31536000
- Content-Security-Policy: default-src 'self'"""

    api_data = "API data: " + "x" * 5000

    messages = [
        ChatMessage(role=Role.TOOL, content=proxy_logs, tool_call_id="tc_proxy", name="get_proxy_logs"),
        ChatMessage(role=Role.TOOL, content=url_info, tool_call_id="tc_url", name="check_url"),
        ChatMessage(role=Role.TOOL, content=api_data, tool_call_id="tc_api", name="api_call"),
    ]

    # Set a low token limit
    llm_client.max_tokens = 500
    truncated = llm_client.truncate_tool_responses(messages, fraction=2)

    # Find the messages in the results by their content patterns rather than positions
    # (since water filling algorithm sorts by size)
    api_msg = next((msg for msg in truncated if msg.content.startswith("API data:")), None)
    proxy_msg = next((msg for msg in truncated if "Proxy Host" in msg.content), None)
    url_msg = next((msg for msg in truncated if "URL: https://example.com" in msg.content), None)

    # Verify API data message was truncated
    assert api_msg is not None
    assert "[Response truncated]" in api_msg.content

    # Check if structured data was preserved better
    assert proxy_msg is not None
    assert url_msg is not None

    # Check if proxy logs contain important information
    if "[Response truncated]" in proxy_msg.content:
        assert "Proxy Host" in proxy_msg.content

    # Total size should be below limit
    total_tokens = sum(llm_client.count_tokens(msg.content) for msg in truncated)
    assert total_tokens <= llm_client.max_tokens // 2
