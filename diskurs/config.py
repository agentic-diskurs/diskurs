import os
import re
from dataclasses import asdict, is_dataclass, field
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, get_args, get_origin
from typing import Type

import yaml

from diskurs.entities import ToolDescription
from diskurs.utils import load_module_from_path

T = TypeVar("T", bound="YamlSerializable")


class YamlSerializable:
    """
    A base class for dataclasses that provides methods to load from YAML
    and serialize back to YAML with key conversion between snake_case
    (used in Python) and camelCase (used in YAML).

    This class provides utility methods to convert between the two formats
    when working with nested data structures like dictionaries and lists.
    """

    @classmethod
    def load_from_yaml(cls: Type[T], yaml_content: str, base_path: Optional[Path] = None) -> T:
        """
        Load a YAML string with camelCase keys and convert it into an instance
        of the dataclass, mapping keys to snake_case.

        :param yaml_content: A YAML formatted string with camelCase keys.
        :param base_path: Optional base path for resolving relative paths.
        :return: An instance of the dataclass with values loaded from the YAML.
        """
        data = yaml.safe_load(yaml_content)
        data = resolve_env_vars(data)  # Replace placeholders with environment variables
        snake_case_data = cls._convert_keys_to_snake_case(data)
        return dataclass_loader(cls, snake_case_data, base_path=base_path)

    def serialize_to_yaml(self) -> str:
        """
        Serialize the dataclass instance into a YAML string with camelCase keys.

        :return: A YAML formatted string with camelCase keys.
        """
        instance_dict = asdict(self)
        camel_case_dict = self._convert_keys_to_camel_case(instance_dict)
        return yaml.dump(camel_case_dict, default_flow_style=False)

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """
        Convert a string from camelCase to snake_case.
        """
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @staticmethod
    def _snake_to_camel(name: str) -> str:
        """
        Convert a string from snake_case to camelCase.
        """
        components = name.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    @classmethod
    def _convert_keys_to_snake_case(cls, d: Any) -> Any:
        """
        Recursively convert all keys in a dictionary (or list) from camelCase
        to snake_case.
        """
        if isinstance(d, dict):
            return {cls._camel_to_snake(k): cls._convert_keys_to_snake_case(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [cls._convert_keys_to_snake_case(i) for i in d]
        else:
            return d

    @classmethod
    def _convert_keys_to_camel_case(cls, d: Any) -> Any:
        """
        Recursively convert all keys in a dictionary (or list) from snake_case
        to camelCase.
        """
        if isinstance(d, dict):
            return {cls._snake_to_camel(k): cls._convert_keys_to_camel_case(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [cls._convert_keys_to_camel_case(i) for i in d]
        else:
            return d


class Registrable:
    registry: dict[str, Type["Registrable"]] = {}
    discriminator: str = "type"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        key = getattr(cls, cls.discriminator, None)
        if key:
            cls.registry[key] = cls

    @classmethod
    def get_subclass(cls, key: str) -> Type["Registrable"]:
        return cls.registry.get(key, cls)


@dataclass(kw_only=True)
class PromptConfig(YamlSerializable, Registrable):
    """
    Represents the prompt configuration for an agent.
    """

    type: str
    location: Path
    user_prompt_argument_class: str
    system_prompt_argument_class: str
    json_formatting_template: Optional[Path] = None


@dataclass(kw_only=True)
class MultistepPromptConfig(PromptConfig):
    """
    Represents the prompt configuration for an agent.
    """

    type: str = "multistep_prompt"
    is_valid_name: str
    is_final_name: str



@dataclass(kw_only=True)
class ConductorPromptConfig(PromptConfig):
    """
    Represents the prompt configuration for an agent.
    """

    type: str = "conductor_prompt"
    user_prompt_argument_class: Optional[str] = None
    system_prompt_argument_class: Optional[str] = None
    longterm_memory_class: str
    can_finalize_name: str


@dataclass(kw_only=True)
class AgentConfig(YamlSerializable, Registrable):
    """
    Represents an agent configuration.
    """

    type: str
    name: str
    topics: list[str]
    max_trials: Optional[int] = 5


@dataclass(kw_only=True)
class MultistepAgentConfig(AgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "multistep"
    llm: str
    prompt: PromptConfig
    tools: Optional[list[str]] = None
    init_prompt_arguments_with_longterm_memory: Optional[bool] = True
    max_reasoning_steps: Optional[int] = 5


@dataclass(kw_only=True)
class ConductorAgentConfig(AgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "conductor"
    llm: str
    prompt: PromptConfig


@dataclass(kw_only=True)
class LLMConfig(YamlSerializable, Registrable):
    """
    Represents the LLM configuration.
    """

    type: str
    name: str
    model_max_tokens: int


@dataclass(kw_only=True)
class AzureLLMConfig(LLMConfig):
    """
    Represents the LLM configuration.
    """

    type: str = "azure"
    api_key: str
    api_version: str
    model_name: str
    endpoint: str
    use_entra_id: bool = False


@dataclass(kw_only=True)
class OpenAILLMConfig(LLMConfig):
    """
    Represents the LLM configuration.
    """

    type: str = "openai"
    api_key: str
    model_name: str


@dataclass
class ToolConfig(YamlSerializable):
    """
    Represents a tool configuration.
    """

    name: str
    function_name: str
    module_path: Path
    configs: Optional[dict] = None
    dependencies: Optional[list[str]] = None


@dataclass
class ToolDependency(YamlSerializable, Registrable):
    type: str
    name: str


@dataclass
class ForumConfig(YamlSerializable):
    """
    Represents the entire config file structure.
    """

    dispatcher_type: str
    first_contact: str
    tool_executor_type: str
    agents: list[AgentConfig]
    llms: list[LLMConfig]
    tools: Optional[list[ToolConfig]] = None
    custom_modules: list[str] = field(default_factory=list)
    tool_dependencies: list[ToolDependency] = field(default_factory=dict)


def resolve_env_vars(data):
    if isinstance(data, dict):
        return {k: resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Replace placeholders of the form ${VAR_NAME:default_value}
        pattern = re.compile(r"\$\{([^}:]+)(?::([^}]+))?\}")
        matches = pattern.findall(data)
        for var, default in matches:
            env_value = os.getenv(var, default)
            if env_value is None:
                raise ValueError(f"Environment variable '{var}' is not set and no default value provided.")
            data = data.replace(f'${{{var}{":" + default if default else ""}}}', env_value)
        return data
    else:
        return data


def get_dataclass_subclass(base_class, data):
    if issubclass(base_class, Registrable):
        discriminator = base_class.discriminator
        if discriminator in data:
            key = data[discriminator]
            subclass = base_class.get_subclass(key)
            if subclass:
                return subclass
            else:
                raise ValueError(f"Unknown {base_class.__name__} type: {key}")
        else:
            raise ValueError(f"Discriminator '{discriminator}' not found in data for {base_class.__name__}")
    else:
        return base_class


def dataclass_loader(dataclass_type, data, base_path: Optional[Path] = None):
    """Recursively loads YAML data into the appropriate dataclass."""
    if is_dataclass(dataclass_type) and isinstance(data, dict):
        # Get the correct subclass based on data
        dataclass_type = get_dataclass_subclass(dataclass_type, data)
        field_types = {field.name: field.type for field in dataclass_type.__dataclass_fields__.values()}
        return dataclass_type(
            **{
                key: dataclass_loader(field_types[key], value, base_path=base_path)
                for key, value in data.items()
                if key in field_types
            }
        )
    elif get_origin(dataclass_type) == list and isinstance(data, list):
        # Handle lists by extracting the type of the list elements
        list_type = get_args(dataclass_type)[0]
        return [dataclass_loader(list_type, item, base_path=base_path) for item in data]
    elif dataclass_type == Path and isinstance(data, str):
        # Resolve paths relative to the base path
        path = Path(data)
        if not path.is_absolute() and base_path is not None:
            path = base_path / path
        return path
    else:
        # If it's neither a dict nor a list, assume it's a primitive type and return as-is
        return data


def pre_load_custom_modules(yaml_data, base_path: Path):
    custom_modules = yaml_data.get("custom_modules", [])
    for module_name in custom_modules:
        module_full_path = (base_path / f"{module_name.replace('.', '/')}.py").resolve()
        load_module_from_path(module_full_path.stem, module_full_path)


def load_config_from_yaml(config: str | Path, base_path: Optional[Path] = None) -> ForumConfig:
    """
    Loads the complete configuration from YAML content and maps it
    to the ForumConfig dataclass.

    :param config: If type string then loads from YAML string, if Path loads the file containing the configs.
    :param base_path: Base path to resolve relative paths.
    :return: An instance of the ForumConfig dataclass.
    """
    if isinstance(config, Path):
        with open(config) as f:
            config_content = f.read()
        if base_path is None:
            base_path = config.parent.resolve()
    else:
        config_content = config
    yaml_data = yaml.safe_load(config_content)
    yaml_data = resolve_env_vars(yaml_data)
    snake_case_data = YamlSerializable._convert_keys_to_snake_case(yaml_data)

    pre_load_custom_modules(snake_case_data, base_path)

    return dataclass_loader(ForumConfig, snake_case_data, base_path=base_path)
