from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from conftest import MyUserPromptArgument
from diskurs import MultiStepAgent, MultistepPrompt, LLMClient, ImmutableConversation
from diskurs.entities import ChatMessage, Role, MessageType, PromptArgument, LongtermMemory

CONDUCTOR_NAME = "my_conductor"


def create_prompt(user_prompt_argument):
    prompt = AsyncMock(spec=MultistepPrompt)
    prompt.create_system_prompt_argument.return_value = AsyncMock()
    prompt.create_user_prompt_argument.return_value = user_prompt_argument
    prompt.render_user_template.return_value = ChatMessage(
        role=Role.USER,
        name="my_multistep",
        content="rendered template",
        type=MessageType.CONVERSATION,
    )
    prompt.is_final.return_value = True
    prompt.user_prompt_argument = user_prompt_argument

    return prompt


@pytest.fixture
def mock_prompt():
    return create_prompt(MyUserPromptArgument())


def create_multistep_agent(mock_prompt):
    async def async_identity(conversation):
        return conversation

    agent = MultiStepAgent(
        name="test_agent",
        prompt=mock_prompt,
        init_prompt_arguments_with_previous_agent=True,
        max_reasoning_steps=3,
        llm_client=AsyncMock(spec=LLMClient),
        topics=[CONDUCTOR_NAME],
    )
    agent.generate_validated_response = async_identity
    return agent


@pytest.fixture
def multistep_agent(mock_prompt):
    return create_multistep_agent(mock_prompt)


@pytest.mark.asyncio
async def test_invoke_with_conductor_as_previous_agent(multistep_agent, conversation):
    conversation = conversation.append(
        message=ChatMessage(
            content="I am a conductor message",
            role=Role.USER,
            type=MessageType.ROUTING,
        )
    )

    # Execute
    result = await multistep_agent.invoke(conversation)
    longterm_memory = conversation.get_agent_longterm_memory(CONDUCTOR_NAME)

    assert all(
        [
            longterm_memory.field1 == result.user_prompt_argument.field1,
            longterm_memory.field2 == result.user_prompt_argument.field2,
            longterm_memory.field3 == result.user_prompt_argument.field3,
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
    result = await multistep_agent.invoke(conversation)
    assert all(
        [
            conversation.user_prompt_argument.field1 == result.user_prompt_argument.field1,
            conversation.user_prompt_argument.field2 == result.user_prompt_argument.field2,
            conversation.user_prompt_argument.field3 == result.user_prompt_argument.field3,
        ]
    )


@dataclass
class MyExtendedLongtermMemory(LongtermMemory):
    field2: str = ""
    field3: str = ""


@dataclass
class MySourceUserPromptArgument(PromptArgument):
    field3: str = ""
    field4: str = ""


@dataclass
class MyExtendedUserPromptArgument(PromptArgument):
    field1: str = "extended user prompt field 1"
    field2: str = "extended user prompt field 2"
    field3: str = "extended user prompt field 3"
    field4: str = "extended user prompt field 4"


@pytest.fixture
def mock_extended_prompt():
    return create_prompt(MyExtendedUserPromptArgument())


@pytest.fixture
def extended_multistep_agent(mock_extended_prompt):
    return create_multistep_agent(mock_extended_prompt)


@pytest.fixture
def extended_conversation():
    conversation = ImmutableConversation(
        conversation_id="my_conversation_id",
        user_prompt_argument=MySourceUserPromptArgument(
            field3="user prompt field 3",
            field4="user prompt field 4",
        ),
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
        longterm_memory={
            "my_conductor": MyExtendedLongtermMemory(
                field2="longterm val 2",
                field3="longterm val 3",
                user_query="longterm user query",
            ),
        },
        active_agent="my_conductor",
    )
    return conversation


@pytest.mark.asyncio
async def test_invoke_with_longterm_memory_and_previous_agent(extended_multistep_agent, extended_conversation):
    conversation = extended_conversation.append(
        message=ChatMessage(
            content="I am a multistep agent message",
            role=Role.USER,
            type=MessageType.CONVERSATION,
        )
    )
    result = await extended_multistep_agent.invoke(conversation)
    assert isinstance(result.user_prompt_argument, MyExtendedUserPromptArgument)
    assert all(
        [
            result.user_prompt_argument.field1 == "extended user prompt field 1",
            result.user_prompt_argument.field2 == "longterm val 2",
            result.user_prompt_argument.field3 == "user prompt field 3",
            result.user_prompt_argument.field4 == "user prompt field 4",
        ]
    )
