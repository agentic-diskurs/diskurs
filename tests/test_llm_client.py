import os
from unittest.mock import Mock

import openai
import pytest
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from diskurs.llm_client import OpenAILLMClient, BaseOaiApiLLMClient
from diskurs.entities import Conversation, ChatMessage, Role, PromptArgument

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
            "function": {
                "arguments": "{'order_id': 'order_12345'}",
                "name": "get_delivery_date"
            }
        }
    ]
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

    return Conversation(system_prompt, user_prompt, system_prompt_arg, user_prompt_arg)


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
    assert truncated_chat[1] == example_messages[1]
    assert truncated_chat[-1] == example_messages[-1]

    assert len(truncated_chat) < len(example_messages)
