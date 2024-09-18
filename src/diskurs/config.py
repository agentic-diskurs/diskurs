import os
import re
import yaml
from dataclasses import asdict, dataclass, is_dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, get_args, get_origin

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
    def load_from_yaml(cls: Type[T], yaml_content: str) -> T:
        """
        Load a YAML string with camelCase keys and convert it into an instance
        of the dataclass, mapping keys to snake_case.

        :param yaml_content: A YAML formatted string with camelCase keys.
        :return: An instance of the dataclass with values loaded from the YAML.
        """
        data = yaml.safe_load(yaml_content)
        data = resolve_env_vars(data)  # Replace placeholders with environment variables
        snake_case_data = cls._convert_keys_to_snake_case(data)
        return dataclass_loader(cls, snake_case_data)

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

        :param name: The camelCase string to be converted.
        :return: The converted snake_case string.
        """
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @staticmethod
    def _snake_to_camel(name: str) -> str:
        """
        Convert a string from snake_case to camelCase.

        :param name: The snake_case string to be converted.
        :return: The converted camelCase string.
        """
        components = name.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    @classmethod
    def _convert_keys_to_snake_case(cls, d: Any) -> Any:
        """
        Recursively convert all keys in a dictionary (or list) from camelCase
        to snake_case.

        :param d: A dictionary or list with camelCase keys.
        :return: A dictionary or list with snake_case keys.
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

        :param d: A dictionary or list with snake_case keys.
        :return: A dictionary or list with camelCase keys.
        """
        if isinstance(d, dict):
            return {cls._snake_to_camel(k): cls._convert_keys_to_camel_case(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [cls._convert_keys_to_camel_case(i) for i in d]
        else:
            return d


@dataclass
class PromptConfig(YamlSerializable):
    """
    Represents the prompt configuration for an agent.
    """

    prompt_assets: Path
    user_prompt_argument_class: str
    system_prompt_argument_class: str
    type: Optional[str] = "prompt"


@dataclass
class AgentConfig(YamlSerializable):
    """
    Represents an agent configuration.
    """

    name: str
    type: str
    llm: str
    prompt: PromptConfig
    additional_arguments: Optional[dict] = field(default=None, repr=False)


@dataclass
class LLMConfig(YamlSerializable):
    """
    Represents the LLM configuration.
    """

    name: str
    type: str
    additional_arguments: Optional[dict] = field(default=None, repr=False)


@dataclass
class ToolConfig(YamlSerializable):
    """
    Represents a tool configuration.
    """

    name: str
    function_name: str
    module_path: Path  # Changed to Path type
    configs: Optional[dict] = None


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
    tools: list[ToolConfig]
    custom_modules: list[str] = field(default_factory=list)


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


def dataclass_loader(dataclass_type, data, base_path: Optional[Path] = None):
    """Recursively loads YAML data into the appropriate dataclass."""
    if is_dataclass(dataclass_type) and isinstance(data, dict):
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
    return dataclass_loader(ForumConfig, snake_case_data, base_path=base_path)
