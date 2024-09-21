from pathlib import Path
import json
import logging
from dataclasses import dataclass, is_dataclass, fields, MISSING, asdict
from typing import Optional, Callable, Any, Type, TypeVar

from jinja2 import Environment, FileSystemLoader, Template

from entities import (
    PromptArgument,
    ChatMessage,
    Role,
    GenericConductorLongtermMemory,
)
from registry import register_prompt
from utils import load_module_from_path

logger = logging.getLogger(__name__)


class PromptValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


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
            f.name for f in dataclass_fields.values() if f.default is MISSING and f.default_factory is MISSING
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
                error_message.append(f"Missing required fields: {', '.join(missing_fields)}.")
            if extra_fields:
                error_message.append(f"Extra fields provided: {', '.join(extra_fields)}. Please remove them.")
            valid_fields = ", ".join(dataclass_fields.keys())
            error_message.append(f"Valid fields are: {valid_fields}.")
            raise PromptValidationError(" ".join(error_message))

        try:
            return user_prompt_argument(**parsed_response)
        except TypeError as e:
            raise PromptValidationError(f"Error constructing {user_prompt_argument.__name__}: {e}")

    @classmethod
    def validate_json(cls, llm_response: str) -> dict:
        """
        Parse and validate the LLM response as JSON.

        :param llm_response: The raw text response from the LLM.
        :return: Parsed dictionary if valid JSON.
        :raises PromptValidationError: If the response is not valid JSON.
        """
        try:
            return json.loads(llm_response)
        except json.JSONDecodeError as e:
            error_message = (
                f"LLM response is not valid JSON. Error: {e.msg} at line {e.lineno}, column {e.colno}. "
                "Please ensure the response is valid JSON and follows the correct format."
            )
            raise PromptValidationError(error_message)

    def parse_user_prompt(self, llm_response: str) -> PromptArgument | ChatMessage:
        """
        Parse the text returned from the LLM into a structured prompt argument.
        First validate the text, then parse it into the prompt argument.
        If the text is not valid, raise a PromptValidationError, and generate a user prompt with the error message,
        for the LLM to correct its output.

        :param llm_response: Response from the LLM.
        :return: Validated prompt argument or a ChatMessage with an error message.
        :raises PromptValidationError: If the text is not valid.
        """
        try:
            parsed_response = self.validate_json(llm_response)  # Use the parse_json method
            validated_response = self.validate_dataclass(parsed_response, self.user_prompt_argument)
            return validated_response
        except PromptValidationError as e:
            return ChatMessage(role=Role.USER, content=str(e))


UserPromptArg = TypeVar("UserPromptArg")
SystemPromptArg = TypeVar("SystemPromptArg")

IS_VALID_DEFAULT_VALUE_NAME = "is_valid"
IS_FINAL_DEFAULT_VALUE_NAME = "is_final"


def is_valid_default(prompt_arguments: UserPromptArg) -> bool:
    return True


def is_final_default(prompt_arguments: UserPromptArg) -> bool:
    return True


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
        is_valid: Optional[Callable[[UserPromptArg], bool]] = None,
        is_final: Optional[Callable[[UserPromptArg], bool]] = None,
    ):
        self.system_prompt_argument = system_prompt_argument_class
        self.user_prompt_argument = user_prompt_argument_class
        self.system_template = system_template
        self.user_template = user_template
        self.is_valid = is_valid or (lambda x: True)
        self.is_final = is_final or (lambda x: True)

    def create_system_prompt_argument(self, **prompt_args: dict) -> SystemPromptArg:
        return self.system_prompt_argument(**prompt_args)

    def create_user_prompt_argument(self, **prompt_args: dict) -> UserPromptArg:
        return self.user_prompt_argument(**prompt_args)

    def render_system_template(self, name: str, prompt_args: PromptArgument) -> ChatMessage:
        return ChatMessage(role=Role.SYSTEM, name=name, content=self.system_template.render(**asdict(prompt_args)))

    def render_user_template(self, name: str, prompt_args: PromptArgument) -> ChatMessage:
        return ChatMessage(
            role=Role.USER,
            name=name,
            content=self.user_template.render(**asdict(prompt_args)),
        )

    def validate_prompt(self, name: str, prompt_args: PromptArgument) -> ChatMessage:
        try:
            if self.is_valid(prompt_args):
                return self.render_user_template(name, prompt_args)
        except PromptValidationError as e:  # Handle only validation errors
            return ChatMessage(role=Role.USER, content=str(e))
        except Exception as e:  # Handle other unforeseen errors separately
            return ChatMessage(role=Role.USER, content=f"An error occurred: {str(e)}")


@register_prompt("multistep_prompt")
class MultistepPrompt(PromptRendererMixin, PromptParserMixin):
    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        system_prompt_argument_class: Type[SystemPromptArg],
        user_prompt_argument_class: Type[UserPromptArg],
        is_valid: Optional[Callable[[Any], bool]] = None,
        is_final: Optional[Callable[[Any], bool]] = None,
    ):
        # Initialize mixins
        PromptRendererMixin.__init__(
            self,
            system_prompt_argument_class,
            user_prompt_argument_class,
            system_template,
            user_template,
            is_valid,
            is_final,
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
    ) -> "MultistepPrompt":
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

        logger.info(f"Loading templates from: {location}")
        logger.info(f"Loading code from: {location / code_filename}")

        with open(location / agent_description_filename, "r") as f:
            agent_description = f.read()

        system_template = cls.load_template(location / system_template_filename)
        user_template = cls.load_template(location / user_template_filename)

        module_path = location / code_filename
        loaded_module = load_module_from_path(module_name=module_path.stem, module_path=module_path)

        system_prompt_arg, user_prompt_arg = cls.load_prompt_arguments(
            loaded_module, system_prompt_argument_class, user_prompt_argument_class
        )

        predicates = cls._prepare_predicates(loaded_module, kwargs)

        return cls(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            system_prompt_argument_class=system_prompt_arg,
            user_prompt_argument_class=user_prompt_arg,
            **predicates,
        )

    @classmethod
    def _prepare_predicates(cls, loaded_module, kwargs):
        predicate_info = {
            "is_valid_name": kwargs.get("is_valid_name", "is_valid"),
            "is_final_name": kwargs.get("is_final_name", "is_final"),
        }

        is_final, is_valid = cls.load_prompt_predicates(loaded_module=loaded_module, **predicate_info)
        predicates = {"is_valid": is_valid, "is_final": is_final}
        return predicates

    @classmethod
    def load_prompt_predicate(cls, predicate_name, default_name, loaded_module):
        try:
            predicate = getattr(loaded_module, predicate_name)
        except AttributeError as e:
            if predicate_name != default_name:
                logger.error(f"Missing expected attribute {predicate_name} in {loaded_module.__name__}: {e}")
                raise AttributeError(f"Required attribute {predicate_name} not found in {loaded_module.__name__}")
            else:

                predicate = None
        return predicate

    @classmethod
    def load_prompt_predicates(cls, is_final_name, is_valid_name, loaded_module):
        is_valid = cls.load_prompt_predicate(
            predicate_name=is_valid_name, default_name=IS_VALID_DEFAULT_VALUE_NAME, loaded_module=loaded_module
        )
        is_final = cls.load_prompt_predicate(
            predicate_name=is_final_name, default_name=IS_FINAL_DEFAULT_VALUE_NAME, loaded_module=loaded_module
        )

        if is_valid is None:
            is_valid = is_valid_default
            logger.warning(f"Predicate function is_valid not found in {loaded_module.__name__}. Using default.")

        if is_final is None:
            is_final = is_final_default
            logger.warning(f"Predicate function is_final not found in {loaded_module.__name__}. Using default.")

        return is_final, is_valid

    @classmethod
    def load_prompt_arguments(cls, loaded_module, system_prompt_arg_name, user_prompt_arg_name):
        try:
            system_prompt_argument = getattr(loaded_module, system_prompt_arg_name)
            user_prompt_argument = getattr(loaded_module, user_prompt_arg_name)
        except AttributeError as e:
            logger.error(f"Missing expected prompt argument classes in {loaded_module.__name__}: {e}")
            raise AttributeError(
                f"Required: ({system_prompt_arg_name}, {user_prompt_arg_name}) "
                f"not found in {loaded_module.__name__}"
            )
        return system_prompt_argument, user_prompt_argument

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


@register_prompt("conductor_prompt")
class ConductorPrompt(MultistepPrompt):
    def __init__(
        self,
        agent_description: str,
        system_template: Template,
        user_template: Template,
        system_prompt_argument_class: Type[SystemPromptArg],
        user_prompt_argument_class: Type[UserPromptArg],
        longterm_memory_class: Type[GenericConductorLongtermMemory],
        can_finalize: Callable[[GenericConductorLongtermMemory], bool],
    ):
        super().__init__(
            agent_description, system_template, user_template, system_prompt_argument_class, user_prompt_argument_class
        )
        self.longterm_memory = longterm_memory_class
        self._can_finalize = can_finalize

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
    ) -> "ConductorPrompt":
        """
        Factory method to create a Prompt object. Loads templates and code dynamically
        based on the provided directory and filenames.

        :param location: Base path where prompt.py and templates are located.
        :param system_prompt_argument_class: Name of the class that specifies the placeholders of the system prompt template
        :param user_prompt_argument_class:  Name of the class that specifies the placeholders of the user prompt template
        :param location: Optional path to a separate template directory. If not provided, will use `location`.
        :param agent_description_filename: location of the text file containing the agent's description
        :param code_filename: Name of the file containing PromptArguments and validation logic.
        :param user_template_filename: Filename of the user template (Jinja2 format).
        :param system_template_filename: Filename of the system template (Jinja2 format).
        :param kwargs:
            is_valid_name: The name of the function to be used for prompt validation.
            is_final_name: The name of the function to check if a prompt's requirements are satisfied.

        :return: An instance of the Prompt class.
        """
        longterm_memory_class_name = kwargs.get("longterm_memory_class")

        logger.info(f"Loading templates from: {location}")
        logger.info(f"Loading code from: {location / code_filename}")

        with open(location / agent_description_filename, "r") as f:
            agent_description = f.read()

        system_template = cls.load_template(location / system_template_filename)
        user_template = cls.load_template(location / user_template_filename)

        module_path = location / code_filename
        loaded_module = load_module_from_path(module_name=module_path.stem, module_path=module_path)

        system_prompt_arg, user_prompt_arg = cls.load_prompt_arguments(
            loaded_module, system_prompt_argument_class, user_prompt_argument_class
        )

        conductor_longterm_memory = cls.load_longterm_memory(loaded_module, longterm_memory_class_name)

        predicates = cls._prepare_predicates(loaded_module, kwargs)

        return cls(
            agent_description=agent_description,
            system_template=system_template,
            user_template=user_template,
            system_prompt_argument_class=system_prompt_arg,
            user_prompt_argument_class=user_prompt_arg,
            longterm_memory_class=conductor_longterm_memory,
            **predicates,
        )

    @classmethod
    def _prepare_predicates(cls, loaded_module, kwargs):
        can_finalize_name = kwargs.get("can_finalize_name", "can_finalize")

        predicate = cls.load_prompt_predicate(
            predicate_name=can_finalize_name, default_name="can_finalize", loaded_module=loaded_module
        )
        return {"can_finalize": predicate}

    def can_finalize(self, longterm_memory: GenericConductorLongtermMemory) -> bool:
        return self._can_finalize(longterm_memory)

    def init_longterm_memory(self) -> GenericConductorLongtermMemory:
        return self.longterm_memory()

    @classmethod
    def load_longterm_memory(cls, loaded_module, longterm_memory_class_name):
        try:
            longterm_memory_class = getattr(loaded_module, longterm_memory_class_name)
        except AttributeError as e:
            logger.error(f"Missing expected longterm memory argument classes in {loaded_module.__name__}: {e}")
            raise AttributeError(
                f"Required: ({longterm_memory_class_name}, {longterm_memory_class_name}) "
                f"not found in {loaded_module.__name__}"
            )
        return longterm_memory_class
