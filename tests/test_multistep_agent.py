from dataclasses import asdict
from unittest.mock import MagicMock

from diskurs import MultiStepAgent
from diskurs.azure_llm_client import AzureOpenAIClient
from diskurs.config import MultistepAgentConfig
from diskurs.prompt import MultistepPrompt


def test_create():
    multistep_mock = MagicMock(speck=MultistepPrompt)

    conf = MultistepAgentConfig(
        name="test",
        topics=["test"],
        max_trials=25,
        llm = "test",
        prompt=multistep_mock,
        max_reasoning_steps=25

    )

    agent = MultiStepAgent.create(
        llm_client=None,
        **asdict(conf)
    )

    assert agent.max_trials == 25
    assert agent.max_reasoning_steps == 25
