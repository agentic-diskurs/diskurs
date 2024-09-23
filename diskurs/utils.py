import importlib.util
import logging
import sys
from pathlib import Path

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
