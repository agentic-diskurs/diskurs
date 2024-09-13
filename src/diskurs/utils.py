import importlib.util
from pathlib import Path


def load_module_from_path(module_name: str, file_path: str):
    # Resolve the full path and check if the file exists
    module_path = Path(file_path).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Module file '{module_path}' not found.")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
