import os
from unittest.mock import Mock

import openai
import pytest
from dotenv import load_dotenv

from diskurs.llm_client import LLMClient
from diskurs.entities import Conversation, ChatMessage, Role, PromptArgument

load_dotenv()


@pytest.fixture
def dummy_openai():
    """Creates a dummy OpenAI object with the correct type signature."""
    # Create a mock OpenAI object with the required attributes
    dummy_openai = Mock(spec=openai.ChatCompletion)

    # You can configure its attributes to match the expected structure
    dummy_openai.create = Mock(return_value=None)

    return dummy_openai


@pytest.fixture
def init_conversation():
    system_prompt = ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant.")
    user_prompt = ChatMessage(role=Role.USER, content="What is the capital of france?")

    system_prompt_arg = PromptArgument()
    user_prompt_arg = PromptArgument()

    return Conversation(system_prompt, user_prompt, system_prompt_arg, user_prompt_arg)


def test_format_for_llm(init_conversation, dummy_openai):

    llm_client = LLMClient(client=dummy_openai, model="")
    formatted = llm_client.format_conversation_for_llm(init_conversation)

    assert isinstance(formatted, dict)
    print(formatted)


def test_generate(init_conversation):
    llm_client = LLMClient.create(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
    res = llm_client.generate(init_conversation)
    print(res)
