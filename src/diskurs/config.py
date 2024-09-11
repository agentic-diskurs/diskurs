import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Type, TypeVar, Any, Optional

import yaml

T = TypeVar('T', bound='YamlSerializable')


# TODO: figure how to handle default values for optional fields

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
        snake_case_data = cls._convert_keys_to_snake_case(data)
        return cls(**snake_case_data)

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
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @staticmethod
    def _snake_to_camel(name: str) -> str:
        """
        Convert a string from snake_case to camelCase.

        :param name: The snake_case string to be converted.
        :return: The converted camelCase string.
        """
        components = name.split('_')
        return components[0] + ''.join(x.title() for x in components[1:])

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
class Prompt(YamlSerializable):
    """
    Represents the prompt configuration for an agent.
    """
    prompt_assets: str
    user_prompt_argument_class: str
    system_prompt_argument_class: str


@dataclass
class Agent(YamlSerializable):
    """
    Represents an agent configuration.
    """
    name: str
    type: str
    llm: str
    prompt: Prompt
    tools: list[str]


@dataclass
class LLM(YamlSerializable):
    """
    Represents the LLM configuration.
    """
    name: str
    model_name: str
    type: str
    endpoint: str
    api_version: str


@dataclass
class ToolConfig(YamlSerializable):
    """
    Represents optional tool-specific configurations.
    """
    foo: Optional[str] = None
    baz: Optional[str] = None


@dataclass
class Tool(YamlSerializable):
    """
    Represents a tool configuration.
    """
    name: str
    function_name: str
    module_name: str
    configs: Optional[ToolConfig] = None


@dataclass
class Config(YamlSerializable):
    """
    Represents the entire config file structure.
    """
    agents: list[Agent]
    llms: list[LLM]
    tools: list[Tool]


def load_config_from_yaml(yaml_content: str | Path) -> Config:
    """
    Loads the complete configuration from YAML content and maps it
    to the Config dataclass.

    :param yaml_content: If type string then loads from YAML string, if Path loads the file containing the configs.
    :return: An instance of the Config dataclass.
    """
    if isinstance(yaml_content, Path):
        with open(yaml_content) as f:
            yaml_content = f.read()
    return Config.load_from_yaml(yaml_content)
