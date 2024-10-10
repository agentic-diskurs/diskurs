from pathlib import Path
import json
import logging
from dataclasses import dataclass, is_dataclass, fields, MISSING, asdict
from typing import Optional, Callable, Any, Type, TypeVar, Self

from jinja2 import Environment, FileSystemLoader, Template

from diskurs.entities import (
    PromptArgument,
    ChatMessage,
    Role,
    GenericConductorLongtermMemory,
    MessageType,
)
from diskurs.protocols import MultistepPromptProtocol, ConductorPromptProtocol
from diskurs.registry import register_prompt
from diskurs.utils import load_module_from_path, load_template_from_package

IS_VALID_DEFAULT_VALUE_NAME = "is_valid"
IS_FINAL_DEFAULT_VALUE_NAME = "is_final"
CAN_FINALIZE_DEFAULT_VALUE_NAME = "can_finalize"
FINALIZE_DEFAULT_VALUE_NAME = "finalize"

logger = logging.getLogger(__name__)


class PromptValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


@dataclass
class DefaultConductorSystemPromptArgument(PromptArgument):
    agent_descriptions: dict[str, str]


@dataclass
class DefaultConductorUserPromptArgument(PromptArgument):
    next_agent: Optional[str] = None


class PromptParserMixin:
    """
    Mixin to handle parsing and validation of LLM responses into dataclasses.
    """

    @classmethod
    def validate_dataclass(
            cls,
            parsed_response: dict[str, Any],
            user_prompt_argument: Type[dataclass],
            strict: bool = False,
    ) -> dataclass:
        """
        Validate that the JSON fields match the target dataclass.
        In strict mode, all fields must be present. If not strict, all required fields (without default values)
        must be present at minimum.

        :param parsed_response: Dictionary representing the LLM's response.
        :param user_prompt_argument: The dataclass type to validate against.
        :param strict: If True, all fields must be present. If False, required fields must be present.
        :return: An instance of the dataclass if validation succeeds.
        :raises: LLMResponseParseError if validation fails.
        """
        if not is_dataclass(user_prompt_argument):
            raise TypeError(f"{user_prompt_argument} is not a valid dataclass")

        dataclass_fields = {f.name: f for f in fields(user_prompt_argument)}
        required_fields = {
            f.name
            for f in dataclass_fields.values()
            if f.default is MISSING and f.default_factory is MISSING
        }

        missing_fields = (
            (dataclass_fields.keys() - parsed_response.keys())
            if strict
            else (required_fields - parsed_response.keys())
        )
        extra_fields = parsed_response.keys() - dataclass_fields.keys()

        if missing_fields or extra_fields:
            error_message = []
            if missing_fields:
                error_message.append(
                    f"Missing required fields: {', '.join(missing_fields)}."
                )
            if extra_fields:
                error_message.append(
                    f"Extra fields provided: {', '.join(extra_fields)}. Please remove them."
                )
            valid_fields = ", ".join(dataclass_fields.keys())
            error_message.append(f"Valid fields are: {valid_fields}.")
            raise PromptValidationError(" ".join(error_message))

        try:
            return user_prompt_argument(**parsed_response)
        except TypeError as e:
            raise PromptValidationError(
                f"Error constructing {user_prompt_argument.__name__}: {e}"
            )

    @classmethod
    def validate_json(cls, llm_response: str) -> dict:
        """
        Parse and validate the LLM response as JSON, handling nested JSON strings.

        :param llm_response: The raw text response from the LLM.
        :return: Parsed dictionary if valid JSON.
        :raises PromptValidationError: If the response is not valid JSON.
        """
        try:
            parsed = json.loads(llm_response)
            # Recursively parse if the result is a string
            while isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, dict):
                return parsed
            else:
                raise PromptValidationError("Parsed response is not a JSON object.")
        except json.JSONDecodeError as e:
            error_message = (
                f"LLM response is not valid JSON. Error: {e.msg} at line {e.lineno}, column {e.colno}. "
                "Please ensure the response is valid JSON and follows the correct format."
            )
            raise PromptValidationError(error_message)

    def parse_user_prompt(
            self, llm_response: str, message_type: MessageType = MessageType.ROUTING
    ) -> PromptArgument | ChatMessage:
        """
        Parse the text returned from the LLM into a structured prompt argument.
        First validate the text, then parse it into the prompt argument.
        If the text is not valid, raise a PromptValidationError, and generate a user prompt with the error message,
        for the LLM to correct its output.

        :param llm_response: Response from the LLM.
        :param message_type: Type of message to be created.
        :return: Validated prompt argument or a ChatMessage with an error message.
        :raises PromptValidationError: If the text is not valid.
        """
        try:
            parsed_response = self.validate_json(
                llm_response
            )  # Use the parse_json method
            validated_response = self.validate_dataclass(
                parsed_response, self.user_prompt_argument
            )
            return validated_response
        except PromptValidationError as e:
            return ChatMessage(role=Role.USER, content=str(e), type=message_type)


UserPromptArg = TypeVar("UserPromptArg")
SystemPromptArg = TypeVar("SystemPromptArg")


# Mixins providing shared behavior
class PromptRendererMixin:
    """
    Mixin to handle the rendering of system and user templates, as well as validation
    This class can be reused across prompts.
    """

    def __init__(
            self,
            system_prompt_argument_class: Type[SystemPromptArg],
            user_prompt_argument_class: Type[UserPromptArg],
            system_template: Template,
            user_template: Template,
            json_formatting_template: Optional[Template] = None,
            is_valid: Optional[Callable[[UserPromptArg], bool]] = None,
            is_final: Optional[Callable[[UserPromptArg], bool]] = None,
    ):
        self.system_prompt_argument = system_prompt_argument_class
        self.user_prompt_argument = user_prompt_argument_class
        self.system_template = system_template
        self.user_template = user_template
        self.json_formatting_template = json_formatting_template
        self._is_valid = is_valid or (lambda x: True)
        self._is_final = is_final or (lambda x: True)

    def is_final(self, user_prompt_argument: PromptArgument) -> bool:
        # TODO: make is_valid and is_final optional (case single step agents)
        return self._is_final(user_prompt_argument)

    def is_valid(self, user_prompt_argument: PromptArgument) -> bool:
        return self._is_valid(user_prompt_argument)

    def create_system_prompt_argument(self, **prompt_args: dict) -> SystemPromptArg:
        return self.system_prompt_argument(**prompt_args)

    def create_user_prompt_argument(self, **prompt_args: dict) -> UserPromptArg:
        return self.user_prompt_argument(**prompt_args)

    def render_json_formatting_prompt(self, prompt_args: dict) -> str:
        if self.json_formatting_template is None:
            raise ValueError("json_render_template is not set.")
        keys = prompt_args.keys()
        rendered_prompt = self.json_formatting_template.render(keys=keys)
        return rendered_prompt

    def render_system_template(
            self, name: str, prompt_args: PromptArgument, return_json: bool = True
    ) -> ChatMessage:
        content = self.system_template.render(**asdict(prompt_args))

        if return_json:
            content += "\n" + self.render_json_formatting_prompt(
                asdict(self.user_prompt_argument())
            )

        return ChatMessage(role=Role.SYSTEM, name=name, content=content)

    def render_user_template(
            self,
            name: str,
            prompt_args: PromptArgument,
            message_type: MessageType = MessageType.CONVERSATION,
    ) -> ChatMessage:
        raise NotImplementedError


class PromptLoaderMixin:
    @classmethod
    def prepare_create(
            cls,
            agent_description_filename,
            code_filename,
            kwargs,
            location,
            system_prompt_argument_class,
            system_template_filename,
            user_prompt_argument_class,
            user_template_filename,
    ):
        logger.info(f"Loading templates from: {location}")
        agent_description, loaded_module, system_template, user_template = (
            cls.load_user_assets(
                agent_description_filename,
                code_filename,
                location,
                system_template_filename,
                user_template_filename,
            )
        )

        if json_formatting_filename := kwargs.get("json_formatting_filename", None):
            json_render_template = cls.load_template(
                location / json_formatting_filename
            )
        else:
            json_render_template = load_template_from_package(
                "diskurs.assets", "json_formatting.jinja2"
            )

        prompt_functions = cls.load_prompt_functions(
            system_prompt_argument_class,
            user_prompt_argument_class,
            loaded_module,
            kwargs,
        )

        return (
            agent_description,
            prompt_functions,
            system_template,
            user_template,
            json_render_template,
        )

    @classmethod
    def load_prompt_functions(
            cls,
            system_prompt_argument_class,
            user_prompt_argument_class,
            loaded_module,
            kwargs,
    ) -> dict[str, Callable]:
        raise NotImplementedError

    @classmethod
    def load_user_assets(
            cls,
            agent_description_filename,
            code_filename,
            location,
            system_template_filename,
            user_template_filename,
    ):
        with open(location / agent_description_filename, "r") as f:
            agent_description = f.read()

        if not system_template_filename:
            system_template_filename = (
                "system_template.jinja2"
                if (location / "system_template.jinja2").exists()
                else None
            )
        system_template = (
            cls.load_template(location / system_template_filename)
            if system_template_filename
            else None
        )
        user_template = (
            cls.load_template(location / user_template_filename)
            if user_template_filename
            else None
        )
        module_path = location / code_filename
        loaded_module = load_module_from_path(
            module_name=module_path.stem, module_path=module_path
        )
        return agent_description, loaded_module, system_template, user_template

    @staticmethod
    def load_symbol(symbol_name, loaded_module):
        try:
            symbol = getattr(loaded_module, symbol_name)
            return symbol
        except AttributeError as e:
            logger.error(
                f"Missing expected attribute {symbol_name} in {loaded_module.__name__}: {e}"
            )
            raise AttributeError(
                f"Required attribute {symbol_name} not found in {loaded_module.__name__}"
            )

    @classmethod
    def load_template(cls, location: Path) -> Template:
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


@register_prompt("multistep_prompt")
class MultistepPrompt(
    PromptRendererMixin, PromptParserMixin, PromptLoaderMixin, MultistepPromptProtocol
):
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
            system_prompt_argument_class=system_prompt_argument_class,
            user_prompt_argument_class=user_prompt_argument_class,
            system_template=system_template,
            user_template=user_template,
            json_formatting_template=json_formatting_template,
            is_valid=is_valid,
            is_final=is_final,
        )
        self.agent_description = agent_description

    @classmethod
    def create(
            cls,
            location: Path,
            system_prompt_argument_class: str,
            user_prompt_argument_class: str,
            agent_description_filename: str = "agent_description.txt",
            code_filename: str = "prompt.py",
            user_template_filename: str = "user_template.jinja2",
            system_template_filename: str = "system_template.jinja2",
            **kwargs,
    ) -> Self:
        """
        Factory method to create a Prompt object. Loads templates and code dynamically
        based on the provided directory and filenames.

        :param location: Base path where prompt.py and templates are located.
        :param system_prompt_argument_class: Name of the class that specifies the placeholders of the system prompt template
        :param user_prompt_argument_class:  Name of the class that specifies the placeholders of the user prompt template
        :param agent_description_filename: location of the text file containing the agent's description
        :param code_filename: Name of the file containing PromptArguments and validation logic.
        :param user_template_filename: Filename of the user template (Jinja2 format).
        :param system_template_filename: Filename of the system template (Jinja2 format).

        :return: An instance of the Prompt class.
        """
        (
            agent_description,
            prompt_functions,
            system_template,
            user_template,
            json_formatting_template,
        ) = cls.prepare_create(
            agent_description_filename,
            code_filename,
            kwargs,
            location,
            system_prompt_argument_class,
            system_template_filename,
            user_prompt_argument_class,
            user_template_filename,
        )

        return cls(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            json_formatting_template=json_formatting_template,
            **prompt_functions,
        )

    @classmethod
    def load_prompt_functions(
            cls,
            system_prompt_argument_class,
            user_prompt_argument_class,
            loaded_module,
            kwargs,
    ) -> dict[str, Callable]:
        return {
            "is_valid": cls.load_symbol(
                kwargs.get("is_valid_name", IS_VALID_DEFAULT_VALUE_NAME), loaded_module
            ),
            "is_final": cls.load_symbol(
                kwargs.get("is_final_name", IS_FINAL_DEFAULT_VALUE_NAME), loaded_module
            ),
            "system_prompt_argument_class": cls.load_symbol(
                system_prompt_argument_class, loaded_module
            ),
            "user_prompt_argument_class": cls.load_symbol(
                user_prompt_argument_class, loaded_module
            ),
        }

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
            return ChatMessage(
                role=Role.USER, name=name, content=str(e), type=message_type
            )
        except Exception as e:
            return ChatMessage(role=Role.USER, content=f"An error occurred: {str(e)}")


@register_prompt("conductor_prompt")
class ConductorPrompt(
    PromptRendererMixin, PromptParserMixin, PromptLoaderMixin, ConductorPromptProtocol
):
    def __init__(
            self,
            agent_description: str,
            system_template: Template,
            user_template: Template,
            system_prompt_argument_class: Type[SystemPromptArg],
            user_prompt_argument_class: Type[UserPromptArg],
            json_formatting_template: Optional[Template] = None,
            longterm_memory_class: Type[GenericConductorLongtermMemory] = None,
            can_finalize: Callable[[GenericConductorLongtermMemory], bool] = None,
            finalize: Callable[[GenericConductorLongtermMemory], bool] = None,
    ):
        super().__init__(
            system_prompt_argument_class=system_prompt_argument_class,
            user_prompt_argument_class=user_prompt_argument_class,
            system_template=system_template,
            user_template=user_template,
            json_formatting_template=json_formatting_template,  # Pass the JSON template
        )
        self.agent_description = agent_description
        self.longterm_memory = longterm_memory_class
        self._can_finalize = can_finalize
        self._finalize = finalize

    @classmethod
    def create(
            cls,
            location: Path,
            system_prompt_argument_class: str,
            user_prompt_argument_class: Optional[str] = None,
            agent_description_filename: Optional[str] = "agent_description.txt",
            code_filename: str = "prompt.py",
            user_template_filename: Optional[str] = None,
            system_template_filename: Optional[str] = None,
            **kwargs,
    ) -> "ConductorPrompt":
        (
            agent_description,
            prompt_functions,
            system_template,
            user_template,
            json_formatting_template,
        ) = cls.prepare_create(
            agent_description_filename,
            code_filename,
            kwargs,
            location,
            system_prompt_argument_class,
            system_template_filename,
            user_prompt_argument_class,
            user_template_filename,
        )

        system_template = system_template or load_template_from_package(
            "diskurs.assets", "conductor_system_template.jinja2"
        )
        user_template = user_template or load_template_from_package(
            "diskurs.assets", "conductor_user_template.jinja2"
        )

        return cls(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            json_formatting_template=json_formatting_template,  # Pass the JSON template
            **prompt_functions,
        )

    @classmethod
    def load_prompt_functions(
            cls,
            system_prompt_argument_class,
            user_prompt_argument_class,
            loaded_module,
            kwargs,
    ) -> dict[str, Callable]:
        return {
            "system_prompt_argument_class": (
                cls.load_symbol(system_prompt_argument_class, loaded_module)
                if system_prompt_argument_class
                else DefaultConductorSystemPromptArgument
            ),
            "user_prompt_argument_class": (
                cls.load_symbol(user_prompt_argument_class, loaded_module)
                if user_prompt_argument_class
                else DefaultConductorUserPromptArgument
            ),
            "can_finalize": cls.load_symbol(
                kwargs.get("can_finalize_name", CAN_FINALIZE_DEFAULT_VALUE_NAME),
                loaded_module,
            ),
            "finalize": cls.load_symbol(
                kwargs.get("finalize_name", FINALIZE_DEFAULT_VALUE_NAME), loaded_module
            ),
            "longterm_memory_class": cls.load_symbol(
                kwargs.get("longterm_memory_class"), loaded_module
            ),
        }

    def can_finalize(self, longterm_memory: GenericConductorLongtermMemory) -> bool:
        return self._can_finalize(longterm_memory)

    def finalize(
            self, longterm_memory: GenericConductorLongtermMemory
    ) -> GenericConductorLongtermMemory:
        return self._finalize(longterm_memory)

    def init_longterm_memory(self, **kwargs) -> GenericConductorLongtermMemory:
        return self.longterm_memory(**kwargs)

    def render_user_template(
            self,
            name: str,
            prompt_args: PromptArgument,
            message_type: MessageType = MessageType.ROUTING,
    ) -> ChatMessage:
        content = self.user_template.render(**asdict(prompt_args))
        return ChatMessage(
            role=Role.USER,
            name=name,
            content=content,
            type=message_type,
        )
