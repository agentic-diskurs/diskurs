import importlib.util
import logging
import sys
from importlib import resources
from pathlib import Path

import jinja2
from jinja2 import Template

logger = logging.getLogger(__name__)


def load_module_from_path(module_name: str, module_path: Path):
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
