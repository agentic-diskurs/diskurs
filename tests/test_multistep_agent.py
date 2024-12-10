from unittest.mock import AsyncMock

import pytest

from conftest import MyUserPromptArgument
from diskurs import MultiStepAgent, MultistepPrompt, LLMClient
from diskurs.entities import ChatMessage, Role, MessageType

CONDUCTOR_NAME = "my_conductor"


@pytest.fixture
def mock_prompt():
    prompt = AsyncMock(spec=MultistepPrompt)
    prompt.create_system_prompt_argument.return_value = AsyncMock()
    prompt.create_user_prompt_argument.return_value = MyUserPromptArgument()
    prompt.render_user_template.return_value = ChatMessage(
        role=Role.USER,
        name="my_multistep",
        content="rendered template",
        type=MessageType.CONVERSATION,
    )
    prompt.is_final.return_value = True

    prompt.user_prompt_argument = MyUserPromptArgument()
    return prompt


@pytest.fixture
def multistep_agent(mock_prompt, conversation):
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
            content="I am a conductor message",
            role=Role.USER,
            type=MessageType.CONVERSATION,
        )
    )
    result = await multistep_agent.invoke(conversation)
    conversation.user_prompt_argument.field1 = result.user_prompt_argument.field1
