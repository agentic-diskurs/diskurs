import inspect
import re
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional
import logging

from entities import ToolCallResult, ToolCall

# Set up logging (this should ideally be done at the module or class level)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def map_python_type_to_json(python_type: str) -> str:
    type_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }
    return type_mapping.get(python_type, "string")


@dataclass
class ToolDescription:
    name: str
    description: str
    arguments: dict[str, dict[str, str]]

    @classmethod
    def from_function(cls, function: Callable):
        """
        Create a ToolDescription instance from a decorated function.
        Assumes the function has been annotated with the @tool decorator.
        """
        return cls(
            name=function.name,
            description=function.description,
            arguments=function.args,
        )


def tool(func):
    docstring = inspect.getdoc(func) or ""

    # Initialize metadata
    metadata = {
        "name": func.__name__,
        "description": "",
        "args": {},
    }

    param_descriptions = {}
    current_param = None

    # Process docstring using regular expressions for :param and :return:
    param_regex = re.compile(r":param\s+(\w+):\s*(.+)")
    return_regex = re.compile(r":return:\s*(.+)")
    description_lines = []
    return_description = None

    for line in map(str.strip, docstring.splitlines()):
        if param_match := param_regex.match(line):
            current_param, param_description = param_match.groups()
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

    # Construct argument metadata from param descriptions and type hints
    for param_name, param_description in param_descriptions.items():
        param_type = func.__annotations__.get(param_name, "unknown").__name__
        metadata["args"][param_name] = {
            "title": param_name.capitalize(),
            "type": param_type,
            "description": param_description.strip(),
        }

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.name = metadata["name"]
    wrapper.description = metadata["description"]
    wrapper.args = metadata["args"]

    return wrapper


class ToolExecutor:

    def __init__(self, tools: Optional[dict[str, Callable]] = None):
        self.tools = tools or {}

    def register_tools(self, tool_list: list[Callable] | Callable) -> None:
        if isinstance(tool_list, list):
            new_tools = {tool.__name__: tool for tool in tool_list}
            for name in new_tools:
                if name in self.tools:
                    logger.warning(
                        f"Tool '{name}' already exists and will be overwritten."
                    )
            self.tools = {**self.tools, **new_tools}
        else:
            tool_name = tool_list.__name__
            if tool_name in self.tools:
                logger.warning(
                    f"Tool '{tool_name}' already exists and will be overwritten."
                )
            self.tools = {**self.tools, tool_name: tool_list}

    def execute_tool(self, tool_call: ToolCall) -> ToolCallResult:
        if tool := self.tools.get(tool_call.function_name):
            return ToolCallResult(
                tool_call_id=tool_call.tool_call_id,
                function_name=tool_call.function_name,
                result=tool(**tool_call.arguments),
            )
        else:
            raise ValueError(f"Tool '{tool_call.function_name}' not found.")
