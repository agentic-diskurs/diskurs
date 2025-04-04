import importlib.util
import logging
import sys
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import jinja2
from jinja2 import Template

logger = logging.getLogger(__name__)


def load_module_from_path(module_path: Path):
    module_name = module_path.stem
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Module file '{module_path}' not found.")

    logger.info(f"Loading module from path: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None:
        raise ImportError(f"Could not create a spec for module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Ensure the module is added to sys.modules
    spec.loader.exec_module(module)

    return module


def load_template_from_package(package_name: str, template_name: str) -> Template:
    """
    Loads a Jinja2 template from the specified package's assets folder.

    :param package_name: The full package name where the assets are located (e.g., 'diskurs.assets').
    :param template_name: The name of the template file (e.g., 'json_formatting_prompt.j2').
    :return: Jinja2 Template object.
    :raises FileNotFoundError: If the template file does not exist.
    """
    logger.info(f"Loading template '{template_name}' from package '{package_name}'")
    try:
        with resources.open_text(package_name, template_name) as f:
            template_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template '{template_name}' not found in package '{package_name}'")

    template = jinja2.Template(template_content)
    return template


T = TypeVar("T", bound=dataclass)


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


def get_fields_as_dict(dataclass_obj: T, fields_to_get: list[str]) -> dict[str, Any]:
    """
    Takes a dataclass instance and a list of fields i.e. properties contained in that dataclass and returns a
    dictionary containing only the fields specified in the fields_to_get

    :param dataclass_obj: the dataclass instance to extract the fields from
    :param fields_to_get: the list of property names to extract from the dataclass
    :return: a dictionary containing the subset of fields as specified in fields_to_get
    """
    return {k: v for k, v in asdict(dataclass_obj).items() if k in fields_to_get}
