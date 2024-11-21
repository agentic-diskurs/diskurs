[![Documentation Status](https://readthedocs.org/projects/diskurs/badge/?version=latest&style=flat-square)](https://diskurs.readthedocs.io/en/latest/)

# Diskurs
*Please note: diskurs is currently under development (meaning early alpha state) and not yet ready for production use.*

Diskurs is a hackable and extendable Python framework for developing LLM-based multi-agentic systems to tackle complex workflow automation tasks. It allows developers to set up agent interactions using customizable configurations.

## Features

- **Multi-Agent System**: Define and configure multiple agents with specific roles and interactions.
- **Configurable Architecture**: Use YAML configuration files to customize agents, tools, and dependencies.
- **Extensible Tools**: Integrate custom tools and modules to extend functionality.
- **OpenAI and Azure OpenAI Integration**: Supports integration with Azure OpenAI services for language models. (to be extended)

## Installation

Diskurs requires **Python 3.12** or higher.

### Using Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management. To install Diskurs and its dependencies:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/agentic-diskurs/diskurs.git
   cd diskurs
   ```

2. **Install Poetry** (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   For more details, refer to the [official Poetry installation guide](https://python-poetry.org/docs/#installation).

3. **Install dependencies**:

   ```bash
   poetry install
   ```
   
4. **Build the documentation**:

   ```bash
   poetry run sphinx-build docs/source docs/build
   ```

5. **Initialize the hook folder**:
To setup the hook folder to the custom hook folder run `setup_hooks.sh`:
```bash
./setup_hooks.sh
```

## Usage

Below is an example of how to launch Diskurs in your project:

```python
import logging
from pathlib import Path
from dotenv import load_dotenv

from diskurs import create_forum_from_config, DiskursInput

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

ticket_content = """
Hello team, I cannot reach http://www.test.com:8080/ from my machine.
I tried to access it around 04.09.2024 at 1 AM. Please can you check what is wrong?
"""

load_dotenv()

diskurs_input = DiskursInput(
    user_query=ticket_content,
    metadata={"company_id": "3929", "proxy_instance": "academy-prx002-ch-zur-1"},
)

def main(config: Path):
    forum = create_forum_from_config(config_path=config, base_path=Path(__file__).parent)
    res = forum.ama(diskurs_input)
    print(res)

if __name__ == "__main__":
    main(Path(__file__).parent / "config.yaml")
```

### Steps to Run

1. **Set Up Configuration**: Create a `config.yaml` file in your project directory. This file contains the necessary configurations for your agents, tools, and dependencies.

2. **Load Environment Variables**: Create a `.env` file or set environment variables in your system. This includes API keys and other sensitive information required by Diskurs.

   Example `.env` file:

   ```dotenv
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   ```

3. **Run the Script**: Execute your Python script to start the forum interaction.

   ```bash
   poetry run python your_script.py
   ```

## Configuration

The `config.yaml` file is used to customize your agents, tools, and dependencies. Below is a condensed example configuration to help you set up your own:

### Language Models

Define the language models your agents will use:

```yaml
llms:
  - name: "gpt-4-base"
    type: "azure"
    modelName: "gpt-4-0613"
    endpoint: "https://your-azure-endpoint.openai.azure.com"
    apiVersion: "2023-03-15-preview"
    apiKey: "${AZURE_OPENAI_API_KEY}"
```

### Agents

Define your agents, specifying their names, types, language models, prompts, tools they use, and the agents they interact with:

```yaml
agents:
  - name: "Conductor_Agent"
    type: "conductor"
    llm: "gpt-4-base"
    prompt:
      type: "conductor_prompt"
      location: "agents/Conductor_Agent"
      userPromptArgumentClass: "ConductorUserPromptArgument"
      systemPromptArgumentClass: "ConductorSystemPromptArgument"
      longtermMemoryClass: "ConductorLongtermMemory"
      canFinalizeName: "can_finalize"
    topics:
      - "Agent_A"
      - "Agent_B"

  - name: "Agent_A"
    type: "multistep"
    llm: "gpt-4-base"
    prompt:
      type: "multistep_prompt"
      location: "agents/Agent_A"
      systemPromptArgumentClass: "AgentASystemPrompt"
      userPromptArgumentClass: "AgentAUserPrompt"
      isValidName: "is_valid"
      isFinalName: "is_final"
    tools:
      - "tool_x"
    topics:
      - "Conductor_Agent"
```

### Tools

List the tools your agents will use and implement the tool functions in the specified `modulePath`:

```yaml
tools:
  - name: "tool_x"
    functionName: "function_x"
    modulePath: "tools/custom_tools.py"
    configs:
      param1: "value1"
      param2: "value2"
```

### External Dependencies

Specify external dependencies required by the tools:

```yaml
toolDependencies:
  - type: "external_service"
    name: "service_x"
    url: "http://service-x-url"
    port: 8080
```

### Setting Up Your Own Configuration

1. **Define Language Models**: In the `llms` section, configure the language models your agents will use. Replace the `endpoint` and `apiKey` with your own Azure OpenAI details.

2. **Create Agents**: In the `agents` section, define your agents. Specify their names, types, language models, prompts, tools they use, and the agents they interact with.

3. **Implement Prompts**: For each agent, create the prompt templates and argument classes as specified in the `prompt` section. These should be located in the paths you provide under `location`.

4. **Add Tools**: In the `tools` section, list the tools your agents will use. Implement the tool functions in the specified `modulePath`.

5. **Configure Dependencies**: If your tools rely on external services or databases, specify them in the `toolDependencies` section.

6. **Environment Variables**: Use environment variables for sensitive information like API keys. Reference them in your `config.yaml` using `${VARIABLE_NAME}`.

## Dependencies

- **Python 3.12** or higher
- **Poetry** for dependency management
- **Required Python packages** (specified in `pyproject.toml`):
  - `diskurs`
  - `python-dotenv`
  - Other dependencies as specified in the repository

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository** on GitHub.

   ```bash
   git clone https://github.com/agentic-diskurs/diskurs.git
   ```

2. **Create a new branch** for your feature or bug fix.

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit your changes** with clear messages.

   ```bash
   git commit -m "Add new feature: description"
   ```

4. **Push your branch** to your forked repository.

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a pull request** detailing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or suggestions, please open an issue on the [GitHub repository](https://github.com/agentic-diskurs/diskurs/issues).

---

*Happy coding with Diskurs!*