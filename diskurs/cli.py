from enum import Enum
from pathlib import Path

import click
import inquirer
import yaml


class AgentType(str, Enum):
    MULTISTEP = "MultistepAgent"
    CONDUCTOR = "ConductorAgent"
    HEURISTIC = "HeuristicAgent"


def create_prompt_config(name: str, agent_type: AgentType) -> dict:
    """Create prompt config based on agent type."""
    base_config = {
        "location": f"agents/{name}/prompt.py",
        "userPromptArgumentClass": "UserPromptArgument",
        "systemPromptArgumentClass": "SystemPromptArgument",
    }

    configs = {
        AgentType.MULTISTEP: {
            **base_config,
            "type": "multistep_prompt",
            "isValidName": "is_valid",
            "isFinalName": "is_final",
        },
        AgentType.CONDUCTOR: {
            **base_config,
            "type": "conductor_prompt",
            "longtermMemoryClass": "LongTermMemory",
            "canFinalizeName": "can_finalize",
            "failName": "fail",
        },
        AgentType.HEURISTIC: {**base_config, "type": "heuristic_prompt", "heuristicSequenceName": "execute_sequence"},
    }
    return configs[agent_type]


def create_agent_config(name: str, agent_type: AgentType) -> dict:
    """Create agent config based on type."""
    base_config = {"name": name, "llm": "default"}

    configs = {
        AgentType.MULTISTEP: {
            **base_config,
            "type": "multistep",
            "prompt": create_prompt_config(name, agent_type),
            "maxReasoningSteps": 5,
            "maxTrials": 5,
            "initPromptArgumentsWithLongtermMemory": True,
            "initPromptArgumentsWithPreviousAgent": True,
        },
        AgentType.CONDUCTOR: {
            **base_config,
            "type": "conductor",
            "prompt": create_prompt_config(name, agent_type),
            "maxDispatches": 50,
            "maxTrials": 5,
        },
        AgentType.HEURISTIC: {
            **base_config,
            "type": "heuristic",
            "prompt": create_prompt_config(name, agent_type),
            "renderPrompt": True,
        },
    }
    return configs[agent_type]


def create_agent_files(name: str, agent_dir: Path) -> None:
    """Create required agent files in the specified directory."""

    PROMPT_TEMPLATE = """from typing import Dict, Any, Optional
    from diskurs.prompts import UserPromptArgument, SystemPromptArgument

    def get_prompt_args() -> Dict[str, Any]:
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.95
        }
    
    def is_valid(response: str) -> bool:
        return len(response.strip()) > 0
    
    def is_final(response: str) -> bool:
        return True
    """

    SYSTEM_TEMPLATE = """You are a helpful AI assistant.
    Please process the following request carefully."""

    USER_TEMPLATE = """{{ user_input }}"""

    # Create directory structure
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Write files
    files = {"prompt.py": PROMPT_TEMPLATE, "system_template": SYSTEM_TEMPLATE, "user_template": USER_TEMPLATE}

    for filename, content in files.items():
        file_path = agent_dir / filename
        file_path.write_text(content)


def update_config(name: str, config_path: Path) -> None:
    """Add agent entry to config.yaml."""
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
    else:
        config = {"agents": {}}  # Change to dict instead of list

    # Add new agent config as dict entry
    agent_config = create_agent_config(name)
    config["agents"][name] = agent_config

    config_path.write_text(yaml.dump(config, default_flow_style=False))


@click.group()
def cli():
    """Diskurs CLI tool."""
    pass


@cli.command()
@click.argument("name")
def create_agent(name: str):
    """Create a new agent with the given name."""
    agent_dir = Path("agents") / name
    config_path = Path("config.yaml")

    if agent_dir.exists():
        click.echo(f"Agent {name} already exists.")
        return

    questions = [
        inquirer.List(
            "agent_type",
            message="Select agent type",
            choices=[t.value for t in AgentType],
            default=AgentType.MULTISTEP.value,
        )
    ]

    answers = inquirer.prompt(questions)
    agent_type = answers["agent_type"]

    agent_dir.mkdir(parents=True, exist_ok=True)
    create_agent_files(name, agent_dir)

    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
    else:
        config = {"agents": {}}

    config["agents"][name] = create_agent_config(name, AgentType(agent_type))
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    click.echo(f"Agent {name} created successfully as {agent_type}")


if __name__ == "__main__":
    cli()
