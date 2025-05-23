import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, get_args, get_origin

import yaml

from diskurs.logger_setup import get_logger
from diskurs.utils import load_module_from_path

logger = get_logger(f"diskurs.{__name__}")

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
    prompt_argument_class: str
    json_formatting_template: Optional[Path] = None


@dataclass(kw_only=True)
class MultistepPromptConfig(PromptConfig):
    """
    Represents the prompt configuration for an agent.
    """

    type: str = "multistep_prompt"
    is_valid_name: Optional[str] = None
    is_final_name: Optional[str] = None


@dataclass(kw_only=True)
class ConductorPromptConfig(PromptConfig):
    """
    Represents the prompt configuration for an agent.
    """

    type: str = "conductor_prompt"
    prompt_argument_class: Optional[str] = None
    can_finalize_name: Optional[str] = None
    fail_name: str


@dataclass(kw_only=True)
class HeuristicPromptConfig(PromptConfig):
    """
    Represents the prompt configuration for an agent.
    """

    type: str = "heuristic_prompt"
    heuristic_sequence_name: str
    prompt_argument_class: str


@dataclass(kw_only=True)
class LLMCompilerPromptConfig(PromptConfig):
    """
    Represents the prompt configuration for an agent.
    """

    type: str = "llm_compiler_prompt"
    location: Optional[Path] = None
    prompt_argument_class: Optional[str] = None
    is_valid_name: Optional[str] = None
    is_final_name: Optional[str] = None


@dataclass(kw_only=True)
class AgentConfig(YamlSerializable, Registrable):
    """
    Represents an agent configuration.
    """

    type: str
    name: str
    topics: Optional[list[str]] = None
    locked_fields: Optional[dict[str, Any]] = None


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
    max_trials: Optional[int] = 5


@dataclass(kw_only=True)
class ParallelMultistepAgentConfig(MultistepAgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "parallel_multistep"
    location: Path
    invoke_on_final: bool = True
    branch_conversation_name: Optional[str] = None
    join_conversations_name: Optional[str] = None


@dataclass(kw_only=True)
class LLMCompilerAgentConfig(AgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "llm_compiler"
    llm: str
    prompt: PromptConfig
    tools: Optional[list[str]] = None
    init_prompt_arguments_with_longterm_memory: Optional[bool] = True
    max_reasoning_steps: Optional[int] = 5
    max_trials: Optional[int] = 5


@dataclass(kw_only=True)
class MultistepAgentPredicateConfig(MultistepAgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "multistep_predicate"


@dataclass(kw_only=True)
class MultistepAgentFinalizerConfig(MultistepAgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "multistep_finalizer"
    final_properties: Optional[list[str]] = None


@dataclass(kw_only=True)
class ConductorAgentConfig(AgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "conductor"
    llm: Optional[str] = None
    prompt: ConductorPromptConfig = None
    agent_descriptions: dict[str, str] = field(default_factory=dict)
    finalizer_name: Optional[str] = None
    supervisor: Optional[str] = None
    can_finalize_name: Optional[str] = None
    max_dispatches: int = 50
    rules: Optional[list["RuleConfig"]] = None  # Add rules field
    fallback_to_llm: bool = True  # Add fallback field


@dataclass(kw_only=True)
class HeuristicAgentConfig(AgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "heuristic"
    prompt: PromptConfig
    llm: Optional[str] = None
    tools: Optional[list[str]] = None
    init_prompt_arguments_with_longterm_memory: Optional[bool] = True
    render_prompt: Optional[bool] = True


@dataclass(kw_only=True)
class HeuristicAgentFinalizerConfig(HeuristicAgentConfig):
    """
    Represents an agent configuration.
    """

    type: str = "heuristic_finalizer"
    final_properties: Optional[list[str]] = None


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
    api_key: str = ""
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
    module_name: str
    configs: Optional[dict] = None
    dependencies: Optional[list[str]] = None


@dataclass
class ToolDependencyConfig(YamlSerializable):
    name: str
    module_name: str
    class_name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationStoreConfig(YamlSerializable, Registrable):
    type: str
    is_persistent: bool = False


@dataclass(kw_only=True)
class FilesystemConversationStoreConfig(ConversationStoreConfig):
    type: str = "filesystem"
    base_path: Optional[Path] = None


@dataclass
class LongtermMemoryConfig(YamlSerializable):
    """Configuration for global longterm memory"""

    type: str
    location: Optional[Path] = None


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
    longterm_memory: LongtermMemoryConfig
    tools: list[ToolConfig] = field(default_factory=list)
    custom_modules: list[dict] = field(default_factory=dict)
    tool_dependencies: list[ToolDependencyConfig] = field(default_factory=list)
    conversation_type: str = "immutable_conversation"
    conversation_store: ConversationStoreConfig = field(
        default_factory=lambda: FilesystemConversationStoreConfig(base_path=Path(__file__).parent / "conversations")
    )


@dataclass(kw_only=True)
class RuleConfig(YamlSerializable):
    """Configuration for a routing rule"""

    name: str
    description: str
    condition_module: str
    condition_name: str
    target_agent: str


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


def dataclass_loader(dataclass_type, data, base_path=None):
    """Recursively loads YAML data into the appropriate dataclass."""
    if is_dataclass(dataclass_type) and isinstance(data, dict):
        # Get the correct subclass based on data
        dataclass_type = get_dataclass_subclass(dataclass_type, data)
        field_types = {field.name: field.type for field in dataclass_type.__dataclass_fields__.values()}

        # Create kwargs dict including both data fields and base_path if needed
        kwargs = {
            key: dataclass_loader(field_types[key], value, base_path=base_path)
            for key, value in data.items()
            if key in field_types
        }

        # If the class accepts base_path, include it in kwargs
        if "base_path" in dataclass_type.__dataclass_fields__:
            kwargs["base_path"] = base_path

        return dataclass_type(**kwargs)

    elif get_origin(dataclass_type) == list and isinstance(data, list):
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
    logger.info(f"Pre-loading custom modules: {custom_modules}")

    for module in custom_modules:
        module_full_path = (base_path / module["location"]).resolve()
        load_module_from_path(module_full_path)


def load_config_from_yaml(config: str | Path, base_path: Optional[Path] = None) -> ForumConfig:
    """
    Loads the complete configuration from YAML content and maps it
    to the ForumConfig dataclass.

    :param config: If type string then loads from YAML string, if Path loads the file containing the configs.
    :param base_path: Base path to resolve relative paths.
    :return: An instance of the ForumConfig dataclass.
    """
    if isinstance(config, Path):
        logger.info(f"Loading config from file: {config}")
        with open(base_path / config) as f:
            config_content = f.read()
        if base_path is None:
            base_path = config.parent.resolve()
    else:
        logger.info("Loading config from string")
        config_content = config
    yaml_data = yaml.safe_load(config_content)
    yaml_data = resolve_env_vars(yaml_data)
    snake_case_data = YamlSerializable._convert_keys_to_snake_case(yaml_data)

    pre_load_custom_modules(snake_case_data, base_path)

    return dataclass_loader(ForumConfig, snake_case_data, base_path=base_path)
