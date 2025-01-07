import inspect
import logging
import re
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Any

from diskurs.config import ToolConfig, ToolDependencyConfig
from diskurs.entities import ToolCallResult, ToolCall
from diskurs.protocols import ToolExecutor as ToolExecutorProtocol, ToolDependency
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
    metadata = {
        "name": func.__name__,
        "description": "",
        "args": {},
        "metadata": {},
        "invisible_args": {},
    }

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
class ToolExecutor(ToolExecutorProtocol):

    def __init__(
        self, tools: Optional[dict[str, Callable]] = None, dependencies: Optional[dict[str, ToolDependency]] = None
    ):
        self.tools = tools or {}
        self.dependencies = dependencies or {}

    def register_tools(self, tools: list[Callable] | Callable) -> None:
        if not isinstance(tools, list):
            tools = [tools]
        for tool in tools:
            tool_name = tool.__name__
            if tool_name in self.tools:
                logger.warning(f"Tool '{tool_name}' already exists and will be overwritten.")
            self.tools[tool_name] = tool

    def register_dependencies(self, dependencies: list[ToolDependency] | ToolDependency) -> None:
        if not isinstance(dependencies, list):
            dependencies = [dependencies]
        for dependency in dependencies:
            if dependency.name in self.dependencies:
                logger.warning(f"Dependency '{dependency.name}' already exists and will be overwritten.")
            self.dependencies[dependency.name] = dependency

    async def execute_tool(self, tool_call: ToolCall, metadata: dict) -> ToolCallResult:
        if tool := self.tools.get(tool_call.function_name):
            invisible_args = {}
            if tool.invisible_args:
                invisible_args = {key: metadata[key] for key in tool.invisible_args if key in metadata}
            return ToolCallResult(
                tool_call_id=tool_call.tool_call_id,
                function_name=tool_call.function_name,
                result=await tool(**{**tool_call.arguments, **invisible_args}),
            )
        else:
            raise ValueError(f"Tool '{tool_call.function_name}' not found.")

    async def call_tool(self, function_name: str, arguments: dict[str, Any]) -> Any:
        """
        Can be used to call a tool directly by providing the function name and arguments.
        This can be handy, when one wants to manually call a tool, without calling an LLM.
        :param function_name: The name of the function to call.

        :param arguments: The arguments to pass to the function.
        :return: The result of the function call.
        """
        tool_call = ToolCall(tool_call_id="0", function_name=function_name, arguments=arguments)
        tool_response = await self.execute_tool(tool_call, {})
        return tool_response.result


def create_func_with_closure(func: Callable, config: ToolConfig, dependencies: list[ToolDependency]) -> Callable:
    func_args = {}

    if config.dependencies:
        dependency_map = {dep.name: dep for dep in dependencies}
        func_args = {
            dep_name: dependency_map[dep_name] for dep_name in config.dependencies if dep_name in dependency_map
        }

        missing_deps = [dep for dep in config.dependencies if dep not in dependency_map]
        if missing_deps:
            raise ValueError(f"Missing configurations for dependencies: {', '.join(missing_deps)}")

    try:
        return func(config.configs, **func_args)

    except AttributeError as e:
        raise ImportError(f"Could not load '{config.function_name}'" + f"from '{config.module_path.name}': {e}")


def load_tools(
    tool_dependencies: list[ToolDependency], tool_configs: list[ToolConfig], custom_modules, base_path
) -> list[Callable]:
    modules_to_functions = defaultdict(list)

    for tool in tool_configs:
        try:
            module_path = base_path / next(
                (module["location"] for module in custom_modules if module["name"] == tool.module_name)
            )
        except StopIteration as e:
            raise FileNotFoundError(f"Could not find module '{tool.module_name}' in custom modules: {e}")

        modules_to_functions[module_path].append(tool.function_name)

    tool_idx = {tool_cfg.function_name: tool_cfg for tool_cfg in tool_configs}

    tool_functions = []

    for module_path, function_names in modules_to_functions.items():
        module_name = Path(module_path).stem
        module_path = Path(module_path).resolve()
        module = load_module_from_path(module_path)

        for function_name in function_names:
            if tool_idx[function_name].dependencies or tool_idx[function_name].configs:
                func = getattr(module, "create_" + function_name)
                func = create_func_with_closure(
                    func=func,
                    config=tool_idx[function_name],
                    dependencies=tool_dependencies,
                )
            else:
                try:
                    func = getattr(module, function_name)

                except AttributeError as e:
                    try:
                        func = getattr(module, "create_" + function_name)
                    except AttributeError as fallback_e:
                        raise ImportError(
                            f"Could neither load '{function_name}' nor create_+{function_name}: {e}"
                            + f"from '{module_path.name}': {fallback_e}"
                        )

            tool_functions.append(func)

    return tool_functions


def load_dependencies(
    dependency_configs: list[ToolDependencyConfig], custom_modules: list[dict], base_path: Path
) -> list[ToolDependency]:
    dependencies = []

    modules_to_classes = defaultdict(list)
    for dependency in dependency_configs:
        try:
            module_path = base_path / next(
                (module["location"] for module in custom_modules if module["name"] == dependency.module_name)
            )
        except StopIteration as e:
            raise FileNotFoundError(f"Could not find module '{dependency.module_name}' in custom modules: {e}")

        modules_to_classes[module_path].append(dependency)

    for module_path, dependencies_list in modules_to_classes.items():
        module_name = Path(module_path).stem
        module_path = Path(module_path).resolve()
        module = load_module_from_path(module_path)

        for dependency in dependencies_list:
            class_ = getattr(module, dependency.class_name)

            instance = class_.create(name=dependency.name, **dependency.parameters)

            dependencies.append(instance)

    return dependencies
