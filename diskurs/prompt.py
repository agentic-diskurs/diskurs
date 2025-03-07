import json
import logging
import re
from dataclasses import dataclass, asdict, is_dataclass, fields, MISSING
from pathlib import Path
from typing import Type, TypeVar, Optional, Callable, Self, Any, Union, get_type_hints

from jinja2 import Template, FileSystemLoader, Environment

from diskurs.entities import MessageType, ChatMessage, Role, PromptArgument, PromptField
from diskurs.protocols import (
    Prompt as PromptProtocol,
    MultistepPrompt as MultistepPromptProtocol,
    ConductorPrompt as ConductorPromptProtocol,
    HeuristicPrompt as HeuristicPromptProtocol,
    HeuristicSequence,
    Conversation,
    CallTool,
    LLMClient,
)
from diskurs.registry import register_prompt
from diskurs.utils import load_template_from_package, load_module_from_path

logger = logging.getLogger(__name__)

UserPromptArg = TypeVar("UserPromptArg", bound=PromptArgument)
SystemPromptArg = TypeVar("SystemPromptArg", bound=PromptArgument)


class PromptValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def always_true(*args, **kwargs) -> bool:
    return True


def safe_load_symbol(symbol_name: str, module: Any, default_factory: Optional[Callable] = None, **kwargs) -> Any:
    """
    Safely loads a symbol from a module with an optional default factory function.

    param: symbol_name: Name of the symbol to load
    param: module: Module to load the symbol from
    param: default_factory: Optional factory function to create a default value if symbol isn't found
    param: kwargs: Additional arguments passed to the default_factory

    return: The loaded symbol or the default value
    """
    try:
        symbol = getattr(module, symbol_name)
        return symbol
    except AttributeError as e:
        logger.warning(f"Missing attribute {symbol_name} in {module.__name__}: {e}\nloading defaults")
        return default_factory(**kwargs) if default_factory else None


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


def validate_json(llm_response: str, max_depth: int = 5, max_size: int = 1_000_000) -> dict:
    """
    Parse and validate the LLM response as JSON, applying several cleaning steps
    to handle common formatting issues.

    If json.loads fails, the function cleans the response and tries parsing again,
    up to a maximum recursion depth.

    :param llm_response: Raw text response from the LLM.
    :param max_depth: Maximum recursion depth for nested cleaning/parsing attempts.
    :param max_size: Maximum allowed size (in characters) for the JSON string.
    :return: A parsed Python dictionary.
    :raises PromptValidationError: If the response is too large, too deeply nested,
                                   or ultimately invalid.
    """
    if len(llm_response) > max_size:
        raise PromptValidationError(f"JSON response exceeds maximum size of {max_size} characters.")

    def _parse_with_depth(json_str: str, current_depth: int) -> dict:
        if current_depth > max_depth:
            raise PromptValidationError("Maximum JSON nesting depth exceeded.")
        try:
            parsed = json.loads(json_str)
            # If we successfully parse a dict, return it.
            if isinstance(parsed, dict):
                return parsed
            # If we parsed a string, it might be double-encoded JSON. Try again.
            if isinstance(parsed, str):
                return _parse_with_depth(parsed, current_depth + 1)
            # Otherwise, we require a top-level JSON object.
            raise PromptValidationError(f"Expected a JSON object, got {type(parsed).__name__}.")
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


def validate_dataclass(
    parsed_response: dict[str, Any],
    user_prompt_argument: Type[GenericDataclass],
) -> GenericDataclass:
    """
    Validate and convert a dictionary to a dataclass instance with proper type coercion.

    :param parsed_response: Dictionary from parsed JSON/response
    :param user_prompt_argument: Target dataclass type

    :return: An instance of the target dataclass

    :raises TypeError: If user_prompt_argument is not a dataclass
    :raises PromptValidationError: If validation or conversion fails
    """
    if not is_dataclass(user_prompt_argument):
        raise TypeError(f"{user_prompt_argument} is not a valid dataclass")

    # Validate field presence
    dataclass_fields = {f.name: f for f in fields(user_prompt_argument)}
    required_fields = {
        f.name for f in dataclass_fields.values() if f.default is MISSING and f.default_factory is MISSING
    }

    missing_required = required_fields - parsed_response.keys()
    extra_fields = parsed_response.keys() - dataclass_fields.keys()

    if missing_required or extra_fields:
        error_parts = []
        if missing_required:
            error_parts.append(f"Missing required fields: {', '.join(missing_required)}.")
        if extra_fields:
            error_parts.append(f"Extra fields provided: {', '.join(extra_fields)}. Please remove them.")

        valid_fields = ", ".join(dataclass_fields.keys())
        error_parts.append(f"Valid fields are: {valid_fields}.")

        raise PromptValidationError(" ".join(error_parts))

    # Convert and validate fields
    converted = {}
    for field_name, field in dataclass_fields.items():
        if field_name not in parsed_response:
            continue

        value = parsed_response[field_name]
        if value is None:
            converted[field_name] = None
            continue

        field_type = field.type
        # Handle Optional types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            field_types = field_type.__args__
            if type(None) in field_types:  # Optional field
                field_type = next(t for t in field_types if t != type(None))

        try:
            if field_type == bool and isinstance(value, str):
                converted[field_name] = value.lower() == "true"
            elif isinstance(value, list) and hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                element_type = field_type.__args__[0]
                converted[field_name] = [element_type(item) for item in value]
            else:
                converted[field_name] = field_type(value)
        except (ValueError, TypeError) as e:
            raise PromptValidationError(
                f"Type conversion failed for field '{field_name}': expected {field_type}, got {type(value)}"
            )

    try:
        return user_prompt_argument(**converted)
    except TypeError as e:
        raise PromptValidationError(f"Error creating {user_prompt_argument.__name__}: {e}")


class BasePrompt(PromptProtocol):
    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        system_prompt_argument_class: Type[SystemPromptArg],
        user_prompt_argument_class: Type[UserPromptArg],
        json_formatting_template: Optional[Template] = None,
        is_valid: Optional[Callable[[Any], bool]] = None,
        is_final: Optional[Callable[[Any], bool]] = None,
    ):
        self.agent_description = agent_description
        self.system_prompt_argument = system_prompt_argument_class
        self.user_prompt_argument = user_prompt_argument_class
        self.system_template = system_template
        self.user_template = user_template
        self.json_formatting_template = json_formatting_template
        self._is_valid = is_valid
        self._is_final = is_final

    @classmethod
    def create_default_is_valid(cls, **kwargs) -> Callable[[UserPromptArg], bool]:
        raise NotImplementedError

    @classmethod
    def create_default_is_final(cls, **kwargs) -> Callable[[UserPromptArg], bool]:
        raise NotImplementedError

    @classmethod
    def safe_load_predicates(cls, is_final_name, is_valid_name, module, **kwargs):
        is_final_name = is_final_name or "is_final"
        is_valid_name = is_valid_name or "is_valid"

        is_valid: Callable[[UserPromptArg], bool] = safe_load_symbol(
            symbol_name=is_valid_name, module=module, default_factory=cls.create_default_is_valid, **kwargs
        )
        is_final: Callable[[UserPromptArg], bool] = safe_load_symbol(
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
        system_prompt_argument_class: str = kwargs.get("system_prompt_argument_class")
        user_prompt_argument_class: str = kwargs.get("user_prompt_argument_class")
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

        system_prompt_argument_class: Type[SystemPromptArg] = safe_load_symbol(
            symbol_name=system_prompt_argument_class, module=module
        )
        user_prompt_argument_class: Type[UserPromptArg] = safe_load_symbol(
            symbol_name=user_prompt_argument_class, module=module
        )

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
            "system_prompt_argument_class": system_prompt_argument_class,
            "user_prompt_argument_class": user_prompt_argument_class,
            "is_valid": is_valid,
            "is_final": is_final,
        }

        all_args = {**base_args, **additional_resources}

        return cls(**all_args)

    def render_system_template(self, name: str, prompt_args: PromptArgument, return_json: bool = True) -> ChatMessage:
        content = self.system_template.render(**asdict(prompt_args))

        if return_json:
            content += "\n" + self.render_json_formatting_prompt(asdict(self.user_prompt_argument()))

        return ChatMessage(role=Role.SYSTEM, name=name, content=content)

    def parse_user_prompt(
        self,
        name: str,
        llm_response: str,
        old_user_prompt_argument: PromptArgument,
        message_type: MessageType = MessageType.CONDUCTOR,
    ) -> PromptArgument | ChatMessage:
        """
        Parse the text returned from the LLM into a structured prompt argument.
        First validate the text, then parse it into the prompt argument.
        If the text is not valid, raise a PromptValidationError, and generate a user prompt with the error message,
        for the LLM to correct its output.

        :param name: Name of the agent.
        :param llm_response: Response from the LLM.
        :param old_user_prompt_argument: The previous user prompt argument.
        :param message_type: Type of message to be created.
        :return: Validated prompt argument or a ChatMessage with an error message.
        :raises PromptValidationError: If the text is not valid.
        """
        logger.debug("Parsing llm response into user prompt arguments")
        try:
            parsed_response = validate_json(llm_response)
            merged_arguments = {**vars(old_user_prompt_argument), **parsed_response}
            validated_response = validate_dataclass(merged_arguments, self.user_prompt_argument)

            return validated_response
        except PromptValidationError as e:
            return ChatMessage(role=Role.USER, name=name, content=str(e), type=message_type)

    def is_final(self, user_prompt_argument: PromptArgument) -> bool:
        return self._is_final(user_prompt_argument)

    def is_valid(self, user_prompt_argument: PromptArgument) -> bool:
        return self._is_valid(user_prompt_argument)

    def create_system_prompt_argument(self, **prompt_args: dict) -> SystemPromptArg:
        return self.system_prompt_argument(**prompt_args)

    def create_user_prompt_argument(self, **prompt_args: dict) -> UserPromptArg:
        return self.user_prompt_argument(**prompt_args)

    def render_json_formatting_prompt(self, prompt_args: dict[str, Any]) -> str:
        """
        Render the JSON formatting template with included fields.

        This method filters the prompt arguments based on PromptField annotations
        and renders only the fields that should be included in the prompt.

        :param prompt_args: Dictionary of prompt arguments to be filtered and rendered.

        :returns: The rendered JSON formatting template.

        :raises: ValueError: If json_formatting_template is not set.
        """
        if self.json_formatting_template is None:
            raise ValueError("json_render_template is not set.")

        hints = get_type_hints(self.user_prompt_argument, include_extras=True)
        included_keys = [
            key
            for key in prompt_args.keys()
            if not hasattr(hints.get(key, None), "__metadata__")
            or any(isinstance(m, PromptField) and m.should_include() for m in hints[key].__metadata__)
        ]

        return self.json_formatting_template.render(keys=included_keys)

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


@register_prompt("multistep_prompt")
class MultistepPrompt(BasePrompt, MultistepPromptProtocol):
    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        system_prompt_argument_class: Type[SystemPromptArg],
        user_prompt_argument_class: Type[UserPromptArg],
        json_formatting_template: Optional[Template] = None,
        is_valid: Optional[Callable[[Any], bool]] = None,
        is_final: Optional[Callable[[Any], bool]] = None,
    ):

        super().__init__(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            system_prompt_argument_class=system_prompt_argument_class,
            user_prompt_argument_class=user_prompt_argument_class,
            json_formatting_template=json_formatting_template,
            is_valid=is_valid,
            is_final=is_final,
        )

    @classmethod
    def create_default_is_valid(cls, **kwargs) -> Callable[[UserPromptArg], bool]:
        return always_true

    @classmethod
    def create_default_is_final(cls, **kwargs) -> Callable[[UserPromptArg], bool]:
        return always_true


@dataclass
class DefaultConductorUserPromptArgument(PromptArgument):
    next_agent: Optional[str] = None


GenericConductorLongtermMemory = TypeVar("GenericConductorLongtermMemory", bound="ConductorLongtermMemory")


@register_prompt("conductor_prompt")
class ConductorPrompt(BasePrompt, ConductorPromptProtocol):
    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        system_prompt_argument_class: Type[SystemPromptArg],
        user_prompt_argument_class: Type[UserPromptArg],
        json_formatting_template: Optional[Template] = None,
        is_valid: Optional[Callable[[Any], bool]] = None,
        is_final: Optional[Callable[[Any], bool]] = None,
        can_finalize: Callable[[GenericConductorLongtermMemory], bool] = None,
        finalize: Callable[[GenericConductorLongtermMemory], GenericConductorLongtermMemory] = None,
        fail: Callable[[GenericConductorLongtermMemory], GenericConductorLongtermMemory] = None,
        longterm_memory: Type[GenericConductorLongtermMemory] = None,
    ):
        super().__init__(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            system_prompt_argument_class=system_prompt_argument_class,
            user_prompt_argument_class=user_prompt_argument_class,
            json_formatting_template=json_formatting_template,
            is_valid=is_valid,
            is_final=is_final,
        )

        self._can_finalize = can_finalize
        self._finalize = finalize
        self._fail = fail
        self.longterm_memory = longterm_memory

    @staticmethod
    def create_default_is_valid(**kwargs) -> Callable[[UserPromptArg], bool]:
        topics: list[str] = kwargs.get("topics", [])

        def is_valid(prompt_args: UserPromptArg) -> bool:
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
    def create_default_is_final(**kwargs) -> Callable[[UserPromptArg], bool]:
        topics: list[str] = kwargs.get("topics", [])

        def is_final(prompt_args: UserPromptArg) -> bool:
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
            if system_template_filename
            else load_template_from_package("diskurs.assets", "conductor_system_template.jinja2")
        )
        user_template = (
            load_template(location / user_template_filename)
            if user_template_filename
            else load_template_from_package("diskurs.assets", "conductor_user_template.jinja2")
        )
        return system_template, user_template

    @classmethod
    def load_additional_resources(cls, module: Any, kwargs: dict) -> dict:
        def load_symbol_if_provided(default_name: str, name_key: str) -> Optional[Callable[..., bool]]:
            symbol_name = kwargs.get(name_key, default_name)
            return safe_load_symbol(symbol_name, module) if symbol_name else None

        assert "longterm_memory_class" in kwargs, "Longterm memory class not provided"

        can_finalize = load_symbol_if_provided("can_finalize", "can_finalize_name")
        finalize = load_symbol_if_provided("finalize", "finalize_name")
        fail = load_symbol_if_provided("fail", "fail_name")

        longterm_memory_class: Type[GenericConductorLongtermMemory] = safe_load_symbol(
            kwargs["longterm_memory_class"],
            module,
        )

        return {
            "can_finalize": can_finalize,
            "finalize": finalize,
            "fail": fail,
            "longterm_memory": longterm_memory_class,
        }

    def can_finalize(self, longterm_memory: GenericConductorLongtermMemory) -> bool:
        return self._can_finalize(longterm_memory)

    def finalize(self, longterm_memory: GenericConductorLongtermMemory) -> GenericConductorLongtermMemory:
        return self._finalize(longterm_memory)

    def fail(self, longterm_memory: GenericConductorLongtermMemory) -> GenericConductorLongtermMemory:
        return self._fail(longterm_memory)

    def init_longterm_memory(self, **kwargs) -> GenericConductorLongtermMemory:
        return self.longterm_memory(**kwargs)


@register_prompt("heuristic_prompt")
class HeuristicPrompt(HeuristicPromptProtocol):
    def __init__(
        self,
        user_prompt_argument_class: Type[PromptArgument],
        heuristic_sequence: HeuristicSequence,
        user_template: Optional[Template] = None,
        agent_description: Optional[str] = "",
    ):
        self.user_prompt_argument = user_prompt_argument_class
        self.user_template = user_template
        self.agent_description = agent_description
        self._heuristic_sequence = heuristic_sequence

    @classmethod
    def create(cls, location: Path, **kwargs) -> Self:
        user_prompt_argument_class: str = kwargs.get("user_prompt_argument_class")
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

        user_prompt_argument_class: Type[UserPromptArg] = safe_load_symbol(
            symbol_name=user_prompt_argument_class, module=module
        )

        user_template = load_template(location / user_template_filename)

        heuristic_sequence: HeuristicSequence = safe_load_symbol(symbol_name=heuristic_sequence_name, module=module)

        assert heuristic_sequence, "Heuristic sequence not found"

        return cls(
            user_prompt_argument_class=user_prompt_argument_class,
            heuristic_sequence=heuristic_sequence,
            user_template=user_template,
            agent_description=agent_description,
        )

    async def heuristic_sequence(
        self, conversation: Conversation, call_tool: Optional[CallTool], llm_client: LLMClient
    ) -> Conversation:
        return await self._heuristic_sequence(conversation=conversation, call_tool=call_tool, llm_client=llm_client)

    def create_user_prompt_argument(self, **prompt_args) -> UserPromptArg:
        return self.user_prompt_argument(**prompt_args)

    def render_user_template(
        self,
        name: str,
        prompt_args: PromptArgument,
        message_type: MessageType = MessageType.CONVERSATION,
    ) -> ChatMessage:
        content = self.user_template.render(**asdict(prompt_args))
        return ChatMessage(
            role=Role.USER,
            name=name,
            content=content,
            type=message_type,
        )
