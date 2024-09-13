import importlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, is_dataclass, fields, MISSING
from pathlib import Path
from typing import Callable, Type, Optional, Any

from jinja2 import Environment, FileSystemLoader, Template

from entities import (
    PromptArgument,
    ChatMessage,
    Role,
    GenericUserPromptArg,
    GenericSystemPromptArg,
)

logger = logging.getLogger(__name__)


class PromptValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class Prompt:

    def __init__(
            self,
            system_template: Template,
            user_template: Template,
            system_prompt_argument_class: Type[GenericSystemPromptArg],
            user_prompt_argument_class: Type[GenericUserPromptArg],
            is_valid: Callable[[GenericUserPromptArg], bool],
            is_final: Callable[[GenericUserPromptArg], bool]
    ):
        self.system_template = system_template
        self.user_template = user_template
        self.user_prompt_argument = user_prompt_argument_class
        self.system_prompt_argument = system_prompt_argument_class
        self.is_valid = is_valid
        self.is_final = is_final

    @classmethod
    def create(
            cls,
            location: Path,
            system_prompt_argument_class: str,
            user_prompt_argument_class: str,
            template_dir: Optional[Path] = None,
            code_filename: str = "prompt.py",
            user_template_filename: str = "user_template.jinja2",
            system_template_filename: str = "system_template.jinja2",
            is_valid_name="is_valid",
            is_final_name="is_final",
    ) -> "Prompt":
        """
        Factory method to create a Prompt object. Loads templates and code dynamically
        based on the provided directory and filenames.

        :param location: Base path where prompt.py and templates are located.
        :param system_prompt_argument_class: Name of the class that specifies the placeholders of the system prompt template
        :param user_prompt_argument_class:  Name of the class that specifies the placeholders of the user prompt template
        :param template_dir: Optional path to a separate template directory. If not provided, will use `location`.
        :param code_filename: Name of the file containing PromptArguments and validation logic.
        :param user_template_filename: Filename of the user template (Jinja2 format).
        :param system_template_filename: Filename of the system template (Jinja2 format).
        :param is_valid_name: The name of the function to be used for prompt validation.
        :param is_final_name: The name of the function to check if a prompt's requirements are satisfied.

        :return: An instance of the Prompt class.
        """
        template_dir = template_dir or location

        logger.info(f"Loading templates from: {template_dir}")
        logger.info(f"Loading code from: {location / code_filename}")

        system_template = cls.load_template(template_dir / system_template_filename)
        user_template = cls.load_template(template_dir / user_template_filename)

        try:
            system_prompt_arg, user_prompt_arg, is_valid, is_final = cls.load_code(
                module_path=location / code_filename,
                system_prompt_arg_name=system_prompt_argument_class,
                user_prompt_arg_name=user_prompt_argument_class,
                is_valid_name=is_valid_name,
                is_final_name=is_final_name,
            )
        except FileNotFoundError as e:
            logger.error(f"Error loading code: {e}")
            raise FileNotFoundError(
                f"Could not load the code from {location / code_filename}"
            )

        return cls(
            system_template=system_template,
            user_template=user_template,
            system_prompt_argument_class=system_prompt_arg,
            user_prompt_argument_class=user_prompt_arg,
            is_valid=is_valid,
            is_final=is_final,
        )

    @classmethod
    def load_code(
            cls,
            module_path: Path,
            system_prompt_arg_name: str,
            user_prompt_arg_name: str,
            is_valid_name: str,
            is_final_name: str,
    ) -> tuple[
        Type[PromptArgument],
        Type[PromptArgument],
        Callable[[PromptArgument], bool],
        Callable[[PromptArgument], bool],
    ]:
        """
        Dynamically loads the module and extracts required attributes (PromptArgument classes and validation logic)
        from the specified module file path.

        :param module_path: Path to the Python module (.py file) to load.
        :param system_prompt_arg_name: Name of the class for the system prompt argument.
        :param user_prompt_arg_name: Name of the class for the user prompt argument.
        :param is_valid_name: The name of the function to be used for prompt validation.
        :param is_final_name: The name of the function to check if a prompt's requirements are satisfied.

        :return: Tuple containing the SystemPromptArgument class, UserPromptArgument class, and validation callables.
        """

        # TODO: try to refactor to use utils.load_module_from_path
        if not module_path.exists():
            raise FileNotFoundError(f"Module file not found: {module_path}")

        logger.info(f"Loading module from path: {module_path}")

        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None:
            raise ImportError(f"Could not create a spec for module from {module_path}")

        loaded_module = importlib.util.module_from_spec(spec)
        sys.modules[module_path.stem] = loaded_module
        spec.loader.exec_module(loaded_module)

        try:
            system_prompt_argument = getattr(loaded_module, system_prompt_arg_name)
            user_prompt_argument = getattr(loaded_module, user_prompt_arg_name)
            is_valid = getattr(loaded_module, is_valid_name)
            is_final = getattr(loaded_module, is_final_name)
        except AttributeError as e:
            logger.error(f"Missing expected attributes in {module_path.stem}: {e}")
            raise AttributeError(
                f"Required attributes (SystemPromptArgument, UserPromptArgument, is_valid, is_final) "
                f"not found in {module_path.stem}"
            )

        return system_prompt_argument, user_prompt_argument, is_valid, is_final

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

    def render_system_template(self, prompt_args: PromptArgument) -> ChatMessage:
        return ChatMessage(
            role=Role.SYSTEM, content=self.system_template.render(**asdict(prompt_args))
        )

    def render_user_template(
            self, name: str, prompt_arg: PromptArgument
    ) -> ChatMessage:
        return ChatMessage(
            role=Role.USER,
            name=name,
            content=self.user_template.render(**asdict(prompt_arg)),
        )

    def validate_prompt(self, name: str, prompt_args: PromptArgument) -> ChatMessage:
        try:
            if self.is_valid(prompt_args):
                return self.render_user_template(name, prompt_args)
        except PromptValidationError as e:  # Handle only validation errors
            return ChatMessage(role=Role.USER, content=str(e))
        except Exception as e:  # Handle other unforeseen errors separately
            return ChatMessage(role=Role.USER, content=f"An error occurred: {str(e)}")

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

    def parse_prompt(self, llm_response: str) -> PromptArgument | ChatMessage:
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
            parsed_response = self.validate_json(
                llm_response
            )  # Use the parse_json method
            validated_response = self.validate_dataclass(
                parsed_response, self.user_prompt_argument
            )
            return validated_response
        except PromptValidationError as e:
            return ChatMessage(role=Role.USER, content=str(e))

    @classmethod
    def from_config(cls, prompt, param):
        pass
