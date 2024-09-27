import inspect
import inspect
import logging
import re
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Any

from diskurs import ToolDependency
from diskurs.config import ToolConfig
from diskurs.entities import ToolCallResult, ToolCall
from diskurs.registry import register_tool_executor
from diskurs.utils import load_module_from_path

logger = logging.getLogger(__name__)


def map_python_type_to_json(python_type: str) -> str:
    type_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }
    return type_mapping.get(python_type, "string")


def tool(func):
    docstring = inspect.getdoc(func) or ""

    # Initialize metadata
    metadata = {"name": func.__name__, "description": "", "args": {}, "metadata": {}, "invisible_args": {}}

    param_descriptions = {}
    current_param = None
    invisible_params = set()

    # Process docstring using regular expressions for :param and :return:
    param_regex = re.compile(r":param\s+(\w+):\s*(.+)")
    return_regex = re.compile(r":return:\s*(.+)")
    description_lines = []
    return_description = None

    for line in map(str.strip, docstring.splitlines()):
        if param_match := param_regex.match(line):
            current_param, param_description = param_match.groups()
            if "[invisible]" in param_description:
                invisible_params.add(current_param)
                param_description = param_description.replace("[metadata]", "").strip()
            param_descriptions[current_param] = param_description
        elif return_match := return_regex.match(line):
            return_description = return_match.group(1)
        elif current_param:
            # Append to current param description
            param_descriptions[current_param] += f" {line}"
        elif line:
            description_lines.append(line)

    metadata["description"] = " ".join(description_lines)
    if return_description:
        metadata["description"] += f"\nReturns: {return_description}"

    if invisible_params:
        metadata["invisible_args"] = {}

    # Construct argument metadata from param descriptions and type hints
    for param_name, param_description in param_descriptions.items():
        param_type = func.__annotations__.get(param_name, "unknown").__name__
        arg_description = {
            "title": param_name,
            "type": param_type,
            "description": param_description.strip(),
        }
        if param_name in invisible_params:
            metadata["invisible_args"][param_name] = arg_description
        else:
            metadata["args"][param_name] = arg_description

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.name = metadata["name"]
    wrapper.description = metadata["description"]
    wrapper.args = metadata["args"]
    wrapper.invisible_args = metadata["invisible_args"]

    return wrapper


@register_tool_executor("default")
class ToolExecutor:

    def __init__(self, tools: Optional[dict[str, Callable]] = None):
        self.tools = tools or {}

    def register_tools(self, tool_list: list[Callable] | Callable) -> None:
        if isinstance(tool_list, list):
            new_tools = {tool.__name__: tool for tool in tool_list}
            for name in new_tools:
                if name in self.tools:
                    logger.warning(f"Tool '{name}' already exists and will be overwritten.")
            self.tools = {**self.tools, **new_tools}
        else:
            tool_name = tool_list.__name__
            if tool_name in self.tools:
                logger.warning(f"Tool '{tool_name}' already exists and will be overwritten.")
            self.tools = {**self.tools, tool_name: tool_list}

    def execute_tool(self, tool_call: ToolCall, metadata: dict) -> ToolCallResult:
        if tool := self.tools.get(tool_call.function_name):
            invisible_args = {}
            if tool.invisible_args:
                invisible_args = {key: metadata[key] for key in tool.invisible_args if key in metadata}
            return ToolCallResult(
                tool_call_id=tool_call.tool_call_id,
                function_name=tool_call.function_name,
                result=tool(**{**tool_call.arguments, **invisible_args}),
            )
        else:
            raise ValueError(f"Tool '{tool_call.function_name}' not found.")


def create_func_with_closure(func: Callable, config: ToolConfig, dependency_config: list[ToolDependency]) -> Callable:
    func_args = {}

    if config.dependencies:
        dependency_config_map = {cfg.name: cfg for cfg in dependency_config}
        func_args = {dep: dependency_config_map[dep] for dep in config.dependencies if dep in dependency_config_map}

        missing_deps = [dep for dep in config.dependencies if dep not in dependency_config_map]
        if missing_deps:
            raise ValueError(f"Missing configurations for dependencies: {', '.join(missing_deps)}")

    try:
        return func(config.configs, **func_args)

    except AttributeError as e:
        raise ImportError(f"Could not load '{config.function_name}'" + f"from '{config.module_path.name}': {e}")


def load_tools(tool_configs: list[ToolConfig], tool_dependencies: list[ToolDependency]) -> list[Callable]:
    modules_to_functions = defaultdict(list)

    for tool in tool_configs:
        modules_to_functions[tool.module_path].append(tool.function_name)

    tool_idx = {tool_cfg.function_name: tool_cfg for tool_cfg in tool_configs}

    tool_functions = []

    for module_path, function_names in modules_to_functions.items():
        module_name = Path(module_path).stem
        module_path = Path(module_path).resolve()
        module = load_module_from_path(module_name, module_path)

        for function_name in function_names:
            if tool_idx[function_name].dependencies or tool_idx[function_name].configs:
                func = getattr(module, "create_" + function_name)
                func = create_func_with_closure(
                    func=func, config=tool_idx[function_name], dependency_config=tool_dependencies
                )
            else:
                try:
                    func = getattr(module, function_name)

                except AttributeError as e:
                    try:
                        func = getattr(module, "create_" + function_name)
                    except AttributeError as fallback_e:
                        raise ImportError(
                            f"Could neither load '{function_name}' nor create_+{function_name}"
                            + f"from '{module_path.name}': {fallback_e}"
                        )

            tool_functions.append(func)

    return tool_functions
