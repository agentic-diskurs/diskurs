import json
import logging
import re
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Self, Type, TypeVar, Union, get_type_hints, get_origin, get_args

from jinja2 import Environment, FileSystemLoader, Template

from diskurs import LockedField
from diskurs.entities import (
    ChatMessage,
    LongtermMemory,
    MessageType,
    OutputField,
    PromptArgument,
    Role,
    PromptField,
)
from diskurs.errors import PromptValidationError
from diskurs.protocols import CallTool
from diskurs.protocols import ConductorPrompt as ConductorPromptProtocol
from diskurs.protocols import Conversation
from diskurs.protocols import HeuristicPrompt as HeuristicPromptProtocol
from diskurs.protocols import HeuristicSequence, LLMClient
from diskurs.protocols import MultistepPrompt as MultistepPromptProtocol
from diskurs.protocols import Prompt as PromptProtocol
from diskurs.registry import register_prompt
from diskurs.utils import load_module_from_path, load_template_from_package, safe_load_symbol

logger = logging.getLogger(__name__)

PromptArg = TypeVar("PromptArg", bound=PromptArgument)


def always_true(*args, **kwargs) -> bool:
    return True


def always_false(*args, **kwargs) -> bool:
    return False


def _initialize_prompt(
    prompt_argument: PromptArgument,
    conversation: "Conversation",
    locked_fields: Optional[dict[str, Any]] = None,
    init_from_longterm_memory: bool = True,
) -> PromptArgument:
    """
    Initialize a prompt argument with values from the global longterm memory.

    This utility function centralizes the logic for initializing prompt arguments, which is used
    by multiple agent types (MultiStepAgent, HeuristicAgent, LLMCompilerAgent, ConductorAgent, etc.).

    :param prompt_argument: The prompt argument to initialize
    :param conversation: The current conversation
    :param locked_fields: Dictionary of fields to lock with specific values
    :param init_from_longterm_memory: Whether to initialize from longterm memory
    :return: initialized prompt_argument
    """

    if init_from_longterm_memory:
        # Use the global longterm memory
        longterm_memory = conversation.longterm_memory

        if longterm_memory and prompt_argument:
            prompt_argument = prompt_argument.init(longterm_memory)

    if locked_fields:
        for field_name, field_value in locked_fields.items():
            if hasattr(prompt_argument, field_name):
                setattr(prompt_argument, field_name, field_value)

    return prompt_argument


def load_template(location: Path) -> Template:
    """
    Loads a Jinja2 template from the provided file path.

    :param location: Path to the template file.
    :return: Jinja2 Template object.
    :raises FileNotFoundError: If the template file does not exist.
    """
    logger.info(f"Loading template: {location}")
    if not location.exists():
        raise FileNotFoundError(f"Template not found: {location}")

    file_loader = FileSystemLoader(location.parent)
    env = Environment(loader=file_loader)
    template = env.get_template(location.name)

    return template


GenericDataclass = TypeVar("GenericDataclass", bound=dataclass)


def escape_newlines_in_json_string(text: str) -> str:
    """
    Iterate through the text and, while inside a JSON string literal,
    replace literal newline characters with the escape sequence "\\n".
    This prevents JSON from choking on unescaped newline characters
    that may occur within a string value.
    """
    result = []
    in_string = False  # Are we currently inside a JSON string?
    escape = False  # Was the previous character a backslash?
    for ch in text:
        # Toggle string state if we see an unescaped double quote.
        if ch == '"' and not escape:
            in_string = not in_string
            result.append(ch)
            continue

        # If we are inside a string and see a literal newline, escape it.
        if in_string and ch == "\n":
            result.append("\\n")
            continue

        # Track backslash escapes.
        if ch == "\\" and not escape:
            escape = True
            result.append(ch)
            continue
        else:
            escape = False

        result.append(ch)
    return "".join(result)


def clean_json_string(text: str) -> str:
    """
    Clean and sanitize a JSON string from common LLM response issues.

    The following steps are applied:

      1. Remove triple-backtick code fences (optionally with "json").
      2. Normalize curly quotes to straight quotes.
      3. Remove trailing commas in objects/arrays.
      4. Remove extraneous text before the first '{' or '[' and after the last '}' or ']'.
      5. Remove carriage returns.
      6. Escape literal newlines found within JSON string values.

    :param text: Raw JSON string (possibly wrapped in code fences and containing unescaped newlines).
    :return: A cleaned JSON string.
    """
    text = text.strip()

    # 1. Remove triple-backtick code fences.
    #    This handles both "```json" and "```" blocks.
    text = re.sub(r"(?s)```(?:json)?\s*(.*?)\s*```", r"\1", text)

    # 2. Normalize curly quotes.
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)

    # 3. Remove trailing commas from objects/arrays.
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # 4. Remove extraneous text outside of the JSON braces/brackets.
    text = re.sub(r"^[^{\[]+", "", text)  # Remove text before the first '{' or '['.
    text = re.sub(r"[^}\]]+$", "", text)  # Remove text after the last '}' or ']'.

    # 5. Remove carriage returns (if any).
    text = text.replace("\r", "")

    # 6. Escape literal newline characters inside JSON string values.
    text = escape_newlines_in_json_string(text)

    return text.strip()


def validate_json(llm_response: str, max_depth: int = 5, max_size: int = 1_000_000) -> Union[dict, list]:
    """
    Parse and validate the LLM response as JSON, applying several cleaning steps
    to handle common formatting issues.

    If json.loads fails, the function cleans the response and tries parsing again,
    up to a maximum recursion depth.

    :param llm_response: Raw text response from the LLM.
    :param max_depth: Maximum recursion depth for nested cleaning/parsing attempts.
    :param max_size: Maximum allowed size (in characters) for the JSON string.
    :return: A parsed Python dictionary or list.
    :raises PromptValidationError: If the response is too large, too deeply nested,
                                   or ultimately invalid.
    """
    if len(llm_response) > max_size:
        raise PromptValidationError(f"JSON response exceeds maximum size of {max_size} characters.")

    def _parse_with_depth(json_str: str, current_depth: int) -> Union[dict, list]:
        if current_depth > max_depth:
            raise PromptValidationError("Maximum JSON nesting depth exceeded.")
        try:
            parsed = json.loads(json_str)
            # Accept both dict and list as valid top-level JSON structures
            if isinstance(parsed, (dict, list)):
                return parsed
            # If we parsed a string, it might be double-encoded JSON. Try again.
            if isinstance(parsed, str):
                return _parse_with_depth(parsed, current_depth + 1)
            # Otherwise, we require a top-level JSON object or array.
            raise PromptValidationError(f"Expected a JSON object or array, got {type(parsed).__name__}.")
        except json.JSONDecodeError as e:
            # Try cleaning the string and parsing again.
            cleaned = clean_json_string(json_str)
            if cleaned and cleaned != json_str:
                return _parse_with_depth(cleaned, current_depth + 1)
            error_msg = (
                f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}. "
                "Please ensure the response is valid JSON."
            )
            raise PromptValidationError(error_msg)

    return _parse_with_depth(llm_response, 1)


def validate_dataclass(parsed_response: dict[str, Any], prompt_argument: Type[GenericDataclass]) -> GenericDataclass:
    """
    Validate and convert a dictionary to a dataclass instance with proper type coercion.
    """
    if not is_dataclass(prompt_argument):
        raise TypeError(f"{prompt_argument} is not a valid dataclass")

    hints = get_type_hints(prompt_argument, include_extras=True)
    dc_fields = {f.name: f for f in fields(prompt_argument)}

    # Determine excluded fields (Input or Locked)
    def is_excluded(name: str) -> bool:
        t = hints.get(name)
        if hasattr(t, "__metadata__"):
            return any(isinstance(m, PromptField) and (m.is_input() or m.is_locked()) for m in t.__metadata__)
        return False

    excluded = {name for name in dc_fields if is_excluded(name)}
    includable = set(dc_fields) - excluded

    # Check required and extra fields
    required = {
        n for n, f in dc_fields.items() if n in includable and f.default is MISSING and f.default_factory is MISSING
    }
    missing = required - parsed_response.keys()
    extra = set(parsed_response) - set(dc_fields)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"Missing required fields: {', '.join(sorted(missing))}.")
        if extra:
            parts.append(f"Extra fields provided: {', '.join(sorted(extra))}. Please remove them.")
        # Preserve original dataclass field order for valid fields
        valid = ", ".join([name for name in dc_fields if name in includable])
        parts.append(f"Valid fields are: {valid}.")
        raise PromptValidationError(" ".join(parts))

    # Helper to strip Optional/Annotated metadata
    def resolve_type(tp):
        origin = get_origin(tp)
        if origin is Union:
            args = [a for a in get_args(tp) if a is not type(None)]
            return resolve_type(args[0]) if args else Any
        if origin is list or origin is dict:
            return tp
        if origin is getattr(__import__("typing"), "Annotated", None):
            return resolve_type(get_args(tp)[0])
        return tp

    # Recursive field parsing
    def parse_field(value, tp):
        rt = resolve_type(tp)
        if value is None:
            return None
        if rt is Any:
            return value
        if rt is bool:
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes")
            return bool(value)
        origin = get_origin(rt)
        if origin is list:
            elem = get_args(rt)[0]
            return [parse_field(v, elem) for v in value]
        if is_dataclass(rt) and isinstance(value, dict):
            return validate_dataclass(value, rt)
        try:
            return rt(value)
        except Exception as e:
            raise PromptValidationError(
                f"Type conversion failed for field: expected {rt}, got {type(value)} ({value})"
            )

    # Build with defaults then override
    default_inst = prompt_argument()
    data = asdict(default_inst)
    for name in includable & parsed_response.keys():
        data[name] = parse_field(parsed_response[name], dc_fields[name].type)

    try:
        return prompt_argument(**data)
    except TypeError as e:
        raise PromptValidationError(f"Error creating {prompt_argument.__name__}: {e}")


class BasePrompt(PromptProtocol):
    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        prompt_argument_class: Type[PromptArg],
        return_json: bool = True,
        json_formatting_template: Optional[Template] = None,
        is_valid: Optional[Callable[[Any], bool]] = None,
        is_final: Optional[Callable[[Any], bool]] = None,
    ):
        self.agent_description = agent_description
        self.prompt_argument = prompt_argument_class
        self.system_template = system_template
        self.user_template = user_template
        self.return_json = return_json
        self.json_formatting_template = json_formatting_template
        self._is_valid = is_valid
        self._is_final = is_final

    @classmethod
    def create_default_is_valid(cls, **kwargs) -> Callable[[PromptArg], bool]:
        raise NotImplementedError

    @classmethod
    def create_default_is_final(cls, **kwargs) -> Callable[[PromptArg], bool]:
        raise NotImplementedError

    @classmethod
    def safe_load_predicates(cls, is_final_name, is_valid_name, module, **kwargs):
        is_final_name = is_final_name or "is_final"
        is_valid_name = is_valid_name or "is_valid"

        is_valid: Callable[[PromptArg], bool] = safe_load_symbol(
            symbol_name=is_valid_name, module=module, default_factory=cls.create_default_is_valid, **kwargs
        )
        is_final: Callable[[PromptArg], bool] = safe_load_symbol(
            symbol_name=is_final_name, module=module, default_factory=cls.create_default_is_final, **kwargs
        )

        return is_final, is_valid

    @classmethod
    def load_additional_resources(cls, module: Any, kwargs: dict) -> dict:
        """Load additional resources specific to subclasses. Override in subclasses if needed."""
        return {}

    @classmethod
    def load_templates(
        cls, location: Path, system_template_filename: str, user_template_filename: str, kwargs: dict
    ) -> tuple[Template, Template]:
        """Load templates with default fallback options. Override in subclasses if needed."""
        system_template = load_template(location / system_template_filename)
        user_template = load_template(location / user_template_filename)
        return system_template, user_template

    @classmethod
    def create(cls, location: Path, **kwargs) -> Self:
        prompt_argument_class: str = kwargs.get("prompt_argument_class")
        agent_description_filename: str = kwargs.get("agent_description_filename", "agent_description.txt")
        code_filename: str = kwargs.get("code_filename", "prompt.py")
        is_valid_name: str = kwargs.get("is_valid_name", "is_valid")
        is_final_name: str = kwargs.get("is_final_name", "is_final")
        user_template_filename: str = kwargs.get("user_template_filename", "user_template.jinja2")
        system_template_filename: str = kwargs.get("system_template_filename", "system_template.jinja2")
        json_formatting_filename = kwargs.get("json_formatting_filename", "")

        module_path = location / code_filename

        agent_description = ""

        if agent_description_filename:
            with open(location / agent_description_filename, "r") as f:
                agent_description = f.read()

        module = load_module_from_path(module_path=module_path)

        prompt_argument_class: Type[PromptArg] = safe_load_symbol(symbol_name=prompt_argument_class, module=module)

        is_final, is_valid = cls.safe_load_predicates(
            is_final_name=is_final_name, is_valid_name=is_valid_name, module=module, topics=kwargs.get("topics", [])
        )

        system_template, user_template = cls.load_templates(
            location=location,
            system_template_filename=system_template_filename,
            user_template_filename=user_template_filename,
            kwargs=kwargs,
        )

        json_render_template: Template = (
            load_template(location / json_formatting_filename)
            if json_formatting_filename
            else load_template_from_package("diskurs.assets", "json_formatting.jinja2")
        )

        additional_resources = cls.load_additional_resources(module, kwargs)

        base_args = {
            "agent_description": agent_description,
            "system_template": system_template,
            "user_template": user_template,
            "json_formatting_template": json_render_template,
            "prompt_argument_class": prompt_argument_class,
            "is_valid": is_valid,
            "is_final": is_final,
        }

        all_args = {**base_args, **additional_resources}

        return cls(**all_args)

    def render_system_template(
        self, name: str, prompt_argument: PromptArgument, return_json: bool = True
    ) -> ChatMessage:
        content = self.system_template.render(**asdict(prompt_argument))

        if self.return_json:
            content += "\n" + self.render_json_formatting_prompt(prompt_argument)

        return ChatMessage(role=Role.SYSTEM, name=name, content=content)

    def parse_user_prompt(
        self,
        name: str,
        llm_response: str,
        old_prompt_argument: PromptArgument,
        message_type: MessageType = MessageType.CONDUCTOR,
    ) -> PromptArgument | ChatMessage:
        """
        Parse the text returned from the LLM into a structured prompt argument.
        First validate the text, then parse it into the prompt argument.
        If the text is not valid, raise a PromptValidationError, and generate a user prompt with the error message,
        for the LLM to correct its output.

        :param name: Name of the agent.
        :param llm_response: Response from the LLM.
        :param old_prompt_argument: The previous user prompt argument.
        :param message_type: Type of message to be created.
        :return: Validated prompt argument or a ChatMessage with an error message.
        :raises PromptValidationError: If the text is not valid.
        """
        logger.debug("Parsing llm response into user prompt arguments")
        try:
            parsed_response = validate_json(llm_response)
            validated_response = validate_dataclass(parsed_response, self.prompt_argument)
            updated_prompt_argument = old_prompt_argument.update(validated_response)
            self.is_valid(updated_prompt_argument)

            # Return the updated prompt argument so partial updates retain existing values
            return updated_prompt_argument
        except PromptValidationError as e:
            return ChatMessage(role=Role.USER, name=name, content=str(e), type=message_type)

    def is_final(self, prompt_argument: PromptArgument) -> bool:
        return self._is_final(prompt_argument)

    def is_valid(self, prompt_argument: PromptArgument) -> bool:
        return self._is_valid(prompt_argument)

    def create_prompt_argument(self, **prompt_args: dict) -> PromptArg:
        return self.prompt_argument(**prompt_args)

    def render_json_formatting_prompt(self, prompt_argument: PromptArgument) -> str:
        """
        Render the JSON formatting template with included fields.

        This method analyzes the structure of the prompt_argument class and generates
        a representative JSON example that shows the expected structure, including nested dataclasses
        and lists of dataclasses.

        :param prompt_args: Dictionary of prompt arguments to be filtered and rendered.

        :returns: The rendered JSON formatting template.

        :raises: ValueError: If json_formatting_template is not set.
        """
        if self.json_formatting_template is None:
            raise ValueError("json_formatting_template is not set.")

        schema = self._generate_json_schema(
            {key: get_type_hints(self.prompt_argument).get(key, None) for key in prompt_argument.get_output_fields()}
        )

        return self.json_formatting_template.render(schema=schema)

    def _generate_json_schema(self, field_dict: dict) -> dict:
        """
        Generate a JSON schema representation of the dataclass structure.

        :param field_dict: Dictionary mapping field names to their type hints
        :return: Dict representing the structure with example values
        """
        schema = {}

        for field_name, field_type in field_dict.items():
            # Handle None type (shouldn't happen in practice)
            if field_type is None:
                schema[field_name] = ""
                continue

            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                field_types = field_type.__args__
                if type(None) in field_types:  # Optional field
                    field_type = next(t for t in field_types if t != type(None))

            # Handle different types
            if field_type == str:
                schema[field_name] = ""
            elif field_type == int:
                schema[field_name] = 0
            elif field_type == float:
                schema[field_name] = 0.0
            elif field_type == bool:
                schema[field_name] = False
            elif field_type == dict or (hasattr(field_type, "__origin__") and field_type.__origin__ is dict):
                schema[field_name] = {}
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                # Handle lists
                element_type = field_type.__args__[0]
                if is_dataclass(element_type):
                    # It's a list of dataclasses, generate example items
                    element_fields = {f.name: f.type for f in fields(element_type)}
                    schema[field_name] = [self._generate_json_schema(element_fields)]
                    # Add a second example item for clarity
                    if not all(value == "" for value in schema[field_name][0].values()):
                        schema[field_name].append(self._generate_json_schema(element_fields))
                else:
                    # Simple list type
                    schema[field_name] = []
            elif is_dataclass(field_type):
                # Handle nested dataclasses
                nested_fields = {f.name: f.type for f in fields(field_type)}
                schema[field_name] = self._generate_json_schema(nested_fields)
            else:
                # Default for unknown types
                schema[field_name] = ""

        return schema

    def render_user_template(
        self,
        name: str,
        prompt_args: PromptArgument,
        message_type: MessageType = MessageType.CONVERSATION,
    ) -> ChatMessage:
        try:
            if self.is_valid(prompt_args):
                content = self.user_template.render(**asdict(prompt_args))
                return ChatMessage(
                    role=Role.USER,
                    name=name,
                    content=content,
                    type=message_type,
                )
        except PromptValidationError as e:
            return ChatMessage(role=Role.USER, name=name, content=str(e), type=message_type)
        except Exception as e:
            return ChatMessage(role=Role.USER, name=name, content=f"An error occurred: {str(e)}")

    def initialize_prompt(
        self,
        agent_name: str,
        conversation: "Conversation",
        locked_fields: Optional[dict[str, Any]] = None,
        init_from_longterm_memory: bool = True,
        reset_prompt: bool = True,
        message_type: MessageType = MessageType.CONVERSATION,
        render_system_prompt: bool = True,
    ) -> Conversation:
        """
        Initialize prompt arguments from global longterm memory and set up the conversation.

        This method handles initialization for all agent types:
        - Regular agents with system and user prompts (default)
        - Conductor agents (using MessageType.CONDUCTOR)
        - Heuristic agents (without system prompts)

        :param agent_name: The name of the agent
        :param conversation: The conversation to initialize
        :param locked_fields: Dictionary of fields that should be locked to specific values
        :param init_from_longterm_memory: Whether to initialize from longterm memory
        :param reset_prompt: Whether to reset the prompt or use conversation's existing prompt
        :param message_type: The message type to use (CONVERSATION or CONDUCTOR)
        :param render_system_prompt: Whether to render a system prompt
        :return: Initialized conversation
        """
        updated_prompt_argument = (
            _initialize_prompt(
                prompt_argument=self.create_prompt_argument(),
                conversation=conversation,
                locked_fields=locked_fields,
                init_from_longterm_memory=init_from_longterm_memory,
            )
            if reset_prompt
            else conversation.prompt_argument
        )

        # Create update parameters
        update_params = {
            "active_agent": agent_name,
            "prompt_argument": updated_prompt_argument,
            "user_prompt": self.render_user_template(
                name=agent_name, prompt_args=updated_prompt_argument, message_type=message_type
            ),
        }

        # Only add system prompt if required (standard agents, but not heuristic agents)
        if render_system_prompt:
            update_params["system_prompt"] = self.render_system_template(
                name=agent_name, prompt_argument=updated_prompt_argument
            )

        return conversation.update(**update_params)


@register_prompt("multistep_prompt")
class MultistepPrompt(BasePrompt, MultistepPromptProtocol):
    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        prompt_argument_class: Type[PromptArg],
        return_json: bool = True,
        json_formatting_template: Optional[Template] = None,
        is_valid: Optional[Callable[[Any], bool]] = None,
        is_final: Optional[Callable[[Any], bool]] = None,
    ):
        super().__init__(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            prompt_argument_class=prompt_argument_class,
            return_json=return_json,
            json_formatting_template=json_formatting_template,
            is_valid=is_valid,
            is_final=is_final,
        )

    @classmethod
    def create_default_is_valid(cls, **kwargs) -> Callable[[PromptArg], bool]:
        return always_true

    @classmethod
    def create_default_is_final(cls, **kwargs) -> Callable[[PromptArg], bool]:
        return always_true


@dataclass
class DefaultConductorPromptArgument(PromptArgument):
    agent_descriptions: LockedField[Optional[dict[str, str]]] = None
    next_agent: OutputField[str] = ""


@register_prompt("conductor_prompt")
class ConductorPrompt(BasePrompt, ConductorPromptProtocol):
    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        prompt_argument_class: Type[PromptArg],
        json_formatting_template: Optional[Template] = None,
        is_valid: Optional[Callable[[Any], bool]] = None,
        is_final: Optional[Callable[[Any], bool]] = None,
        can_finalize: Callable[[Any], bool] = None,
        finalize: Callable[[Any], Any] = None,
        fail: Callable[[Any], Any] = None,
    ):
        super().__init__(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            prompt_argument_class=prompt_argument_class,
            json_formatting_template=json_formatting_template,
            is_valid=is_valid,
            is_final=is_final,
        )

        self._can_finalize = can_finalize
        self._finalize = finalize
        self._fail = fail

    @staticmethod
    def create_default_is_valid(**kwargs) -> Callable[[PromptArg], bool]:
        topics: list[str] = kwargs.get("topics", [])

        def is_valid(prompt_args: PromptArg) -> bool:
            if not prompt_args.next_agent:
                return True
            if prompt_args.next_agent in topics:
                return True
            else:
                raise PromptValidationError(
                    f"{prompt_args.next_agent} cannot be routed to from this agent. Valid agents are: {topics}"
                )

        return is_valid

    @staticmethod
    def create_default_is_final(**kwargs) -> Callable[[PromptArg], bool]:
        topics: list[str] = kwargs.get("topics", [])

        def is_final(prompt_args: PromptArg) -> bool:
            if not prompt_args.next_agent:
                return False
            else:
                return prompt_args.next_agent in topics

        return is_final

    @classmethod
    def load_templates(
        cls, location: Path, system_template_filename: str, user_template_filename: str, kwargs: dict
    ) -> tuple[Template, Template]:
        system_template = (
            load_template(location / system_template_filename)
            if (location / system_template_filename).exists()
            else load_template_from_package("diskurs.assets", "conductor_system_template.jinja2")
        )
        user_template = (
            load_template(location / user_template_filename)
            if (location / user_template_filename).exists()
            else load_template_from_package("diskurs.assets", "conductor_user_template.jinja2")
        )
        return system_template, user_template

    @classmethod
    def load_additional_resources(cls, module: Any, kwargs: dict) -> dict:
        """Modified to provide default implementations for conductor resources."""

        def load_symbol_if_provided(default_name: str, name_key: str) -> Optional[Callable[..., bool]]:
            if name_key in kwargs and kwargs[name_key] is not None:
                symbol_name = kwargs[name_key]
            else:
                symbol_name = default_name
            return safe_load_symbol(symbol_name, module) if symbol_name else None

        def default_finalize(ltm):
            return asdict(ltm) if hasattr(ltm, "__dict__") else ltm

        def default_fail(ltm):
            if hasattr(ltm, "__dict__"):
                result = asdict(ltm)
                result["error"] = "Failed to finalize conversation"
                return result
            return {"error": "Failed to finalize conversation"}

        # Load functions with defaults
        can_finalize = load_symbol_if_provided("can_finalize", "can_finalize_name")
        finalize = load_symbol_if_provided("finalize", "finalize_name")
        fail = load_symbol_if_provided("fail", "fail_name")

        # Use defaults if not provided
        if can_finalize is None:
            can_finalize = always_false
        if finalize is None:
            finalize = default_finalize
        if fail is None:
            fail = default_fail

        return {
            "can_finalize": can_finalize,
            "finalize": finalize,
            "fail": fail,
        }

    @classmethod
    def create(cls, location: Path, **kwargs) -> Self:
        """Override BasePrompt.create to use default conductor prompt argument classes when needed."""
        prompt_argument_class: str = kwargs.get("prompt_argument_class")
        agent_description_filename: str = kwargs.get("agent_description_filename", "agent_description.txt")
        code_filename: str = kwargs.get("code_filename", "prompt.py")
        is_valid_name: str = kwargs.get("is_valid_name", "is_valid")
        is_final_name: str = kwargs.get("is_final_name", "is_final")
        user_template_filename: str = kwargs.get("user_template_filename", "user_template.jinja2")
        system_template_filename: str = kwargs.get("system_template_filename", "system_template.jinja2")
        json_formatting_filename = kwargs.get("json_formatting_filename", "")

        module_path = location / code_filename

        # Read agent description
        agent_description = ""
        try:
            if agent_description_filename and (location / agent_description_filename).exists():
                with open(location / agent_description_filename, "r") as f:
                    agent_description = f.read()
        except Exception as e:
            logger.warning(f"Error reading agent description: {e}")

        if module_path.exists():
            module = load_module_from_path(module_path=module_path)
        else:
            module = None
            logger.warning(f"Module file {module_path} not found, using default values")

        user_prompt_class = None

        if prompt_argument_class and module:
            user_prompt_class = safe_load_symbol(symbol_name=prompt_argument_class, module=module)

        if not user_prompt_class:
            user_prompt_class = DefaultConductorPromptArgument
            logger.info("Using ConductorPromptArgument")

        is_final, is_valid = cls.safe_load_predicates(
            is_final_name=is_final_name, is_valid_name=is_valid_name, module=module, topics=kwargs.get("topics", [])
        )

        # Load templates
        system_template, user_template = cls.load_templates(
            location=location,
            system_template_filename=system_template_filename,
            user_template_filename=user_template_filename,
            kwargs=kwargs,
        )

        # Load JSON formatting template
        json_render_template = None
        if json_formatting_filename and (location / json_formatting_filename).exists():
            try:
                json_render_template = load_template(location / json_formatting_filename)
            except Exception:
                json_render_template = load_template_from_package("diskurs.assets", "json_formatting.jinja2")
        else:
            json_render_template = load_template_from_package("diskurs.assets", "json_formatting.jinja2")

        # Load additional resources specific to conductor prompt
        additional_resources = cls.load_additional_resources(module, kwargs)

        base_args = {
            "agent_description": agent_description,
            "system_template": system_template,
            "user_template": user_template,
            "json_formatting_template": json_render_template,
            "prompt_argument_class": user_prompt_class,
            "is_valid": is_valid,
            "is_final": is_final,
        }

        all_args = {**base_args, **additional_resources}

        return cls(**all_args)

    def can_finalize(self, longterm_memory: LongtermMemory) -> bool:
        return self._can_finalize(longterm_memory)

    def finalize(self, longterm_memory: LongtermMemory) -> LongtermMemory:
        return self._finalize(longterm_memory)

    def fail(self, longterm_memory: LongtermMemory) -> LongtermMemory:
        return self._fail(longterm_memory)


@register_prompt("heuristic_prompt")
class HeuristicPrompt(BasePrompt, HeuristicPromptProtocol):
    def __init__(
        self,
        prompt_argument_class: Type[PromptArgument],
        heuristic_sequence: HeuristicSequence,
        user_template: Optional[Template] = None,
        agent_description: Optional[str] = "",
    ):
        self.prompt_argument = prompt_argument_class
        self.user_template = user_template
        self.agent_description = agent_description
        self._heuristic_sequence = heuristic_sequence

    @classmethod
    def create(cls, location: Path, **kwargs) -> Self:
        prompt_argument_class: str = kwargs.get("prompt_argument_class")
        agent_description_filename: str = kwargs.get("agent_description_filename", "agent_description.txt")
        code_filename: str = kwargs.get("code_filename", "prompt.py")
        user_template_filename: str = kwargs.get("user_template_filename", "user_template.jinja2")
        heuristic_sequence_name: str = kwargs.get("heuristic_sequence_name", "heuristic_sequence")

        module_path = location / code_filename

        agent_description = ""

        if agent_description_filename:
            with open(location / agent_description_filename, "r") as f:
                agent_description = f.read()

        module = load_module_from_path(module_path=module_path)

        prompt_argument_class: Type[PromptArg] = safe_load_symbol(symbol_name=prompt_argument_class, module=module)

        user_template = None
        template_location = location / user_template_filename
        if template_location.exists():
            user_template = load_template(location / user_template_filename)

        heuristic_sequence: HeuristicSequence = safe_load_symbol(symbol_name=heuristic_sequence_name, module=module)

        assert heuristic_sequence, "Heuristic sequence not found"

        return cls(
            prompt_argument_class=prompt_argument_class,
            heuristic_sequence=heuristic_sequence,
            user_template=user_template,
            agent_description=agent_description,
        )

    async def heuristic_sequence(
        self, conversation: Conversation, call_tool: Optional[CallTool], llm_client: LLMClient
    ) -> Conversation:
        return await self._heuristic_sequence(conversation=conversation, call_tool=call_tool, llm_client=llm_client)

    def create_prompt_argument(self, **prompt_args) -> PromptArg:
        return self.prompt_argument(**prompt_args)

    def render_user_template(
        self,
        name: str,
        prompt_args: PromptArgument,
        message_type: MessageType = MessageType.CONVERSATION,
    ) -> ChatMessage:
        content = self.user_template.render(**asdict(prompt_args)) if self.user_template else ""
        return ChatMessage(
            role=Role.USER,
            name=name,
            content=content,
            type=message_type,
        )
