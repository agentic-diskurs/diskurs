import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from diskurs import ForumFactory
from diskurs.prompt import ConductorPrompt, MultistepPrompt


def test_create_agent_max_values():
    # Create a temporary directory for the config file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Write the minimal config.yaml file
        config_content = '''
first_contact: "conductor_agent"
toolExecutorType: "default"
dispatcherType: "synchronous"
# customModules:
agents:
  - name: "conductor_agent"
    type: "conductor"
    llm: "gpt-4-o-openai"
    maxTrials: 10
    prompt:
      type: "conductor_prompt"
      location: "agents/conductor_agent"
      userPromptArgumentClass: "ConductorUserPromptArgument"
      systemPromptArgumentClass: "ConductorSystemPromptArgument"
      longtermMemoryClass: "MyConductorLongtermMemory"
      canFinalizeName: "can_finalize"
    topics:
      - "first_agent"
      - "second_agent"
    finalizerName: "finalizer_agent"
  - name: "first_agent"
    type: "multistep"
    llm: "gpt-4-o-openai"
    maxReasoningSteps: 25
    prompt:
      type: "multistep_prompt"
      location: "agents/first_agent"
      systemPromptArgumentClass: "FirstSystemPromptArgument"
      userPromptArgumentClass: "FirstUserPromptArgument"
      isValidName: "is_valid"
      isFinalName: "is_final"
    topics:
      - "conductor_agent"
  - name: "second_agent"
    type: "multistep"
    llm: "gpt-4-o-openai"
    prompt:
      type: "multistep_prompt"
      location: "agents/second_agent"
      systemPromptArgumentClass: "SecondSystemPromptArgument"
      userPromptArgumentClass: "SecondUserPromptArgument"
      isValidName: "is_valid"
      isFinalName: "is_final"
    topics:
        - "conductor_agent"

llms:
  - name: "gpt-4-o-openai"
    type: "openai"
    modelName: "gpt-4o"
    modelMaxTokens: 8192
    apiKey: "${OPENAI_API_KEY}"
'''
        config_path = temp_path / "config.yaml"
        config_path.write_text(config_content)

        # TODO: refactor forum, for it to be better testable
        with patch('openai.OpenAI') as MockOpenAIClient, \
                patch.object(ConductorPrompt, 'create', return_value=MagicMock()) as mock_create_conductor_prompt,\
                patch.object(MultistepPrompt, 'create', return_value=MagicMock()) as mock_create_multistep_prompt:
            # Initialize the ForumFactory and create the foru
            factory = ForumFactory(config_path=config_path, base_path=temp_path)
            forum = factory.create_forum()

            # Retrieve the agents
            conductor_agent = next((agent for agent in forum.agents if agent.name == 'conductor_agent'), None)
            first_agent = next((agent for agent in forum.agents if agent.name == 'first_agent'), None)

            # Assertions for conductor_agent
            assert conductor_agent is not None, "Agent 'conductor_agent' was not created."
            assert conductor_agent.max_trials == 10, f"Expected max_trials to be 10, got {conductor_agent.max_trials}."
            assert conductor_agent.prompt == mock_create_conductor_prompt.return_value, "Conductor agent's prompt was not set correctly."

            # Assertions for first_agent
            assert first_agent is not None, "Agent 'first_agent' was not created."
            assert first_agent.max_reasoning_steps == 25, f"Expected max_reasoning_steps to be 25, got {first_agent.max_reasoning_steps}."
            assert first_agent.prompt == mock_create_multistep_prompt.return_value, "First agent's prompt was not set correctly."
