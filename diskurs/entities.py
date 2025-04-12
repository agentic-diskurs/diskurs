from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Annotated, Any, Callable, Generic, Optional, TypeVar, Union, get_args, get_origin


class AccessMode(Enum):
    """
    Enumeration representing different access modes or states for fields.
    """

    INPUT = "input"  # Field accepts user input
    OUTPUT = "output"  # Field displays output only
    LOCKED = "locked"  # Field is locked/unchangeable

    def __str__(self):
        """Return the value of the enum member."""
        return self.value


# Type variable for generic field types
T = TypeVar("T")


@dataclass
class JsonSerializable:
    def to_dict(self):
        def serialize(obj):
            if is_dataclass(obj):
                return {f.name: serialize(getattr(obj, f.name)) for f in fields(obj)}
            elif isinstance(obj, Enum):
                return obj.value  # Convert Enum to its value
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: serialize(value) for key, value in obj.items()}
            else:
                return obj

        return serialize(self)

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return None

        def deserialize(cls_or_type, data):
            if data is None:
                return None
            if is_dataclass(cls_or_type):
                kwargs = {}
                for field in fields(cls_or_type):
                    field_name = field.name
                    field_type = field.type
                    value = data.get(field_name)
                    if value is not None:
                        kwargs[field_name] = deserialize_field(field_type, value)
                return cls_or_type(**kwargs)
            elif isinstance(cls_or_type, type) and issubclass(cls_or_type, Enum):
                return cls_or_type(data)
            else:
                return data

        def deserialize_field(field_type, value):

            origin = get_origin(field_type)
            args = get_args(field_type)

            if origin is Annotated:
                field_type = args[0]  # The actual type is the first argument
                origin = get_origin(field_type)
                args = get_args(field_type)

            if origin is list:
                item_type = args[0]
                return [deserialize_field(item_type, item) for item in value]
            elif origin is dict:
                key_type, val_type = args
                return {deserialize_field(key_type, k): deserialize_field(val_type, v) for k, v in value.items()}
            elif origin is Optional or origin is Union:
                # Handle Optional fields
                for arg in args:
                    try:
                        return deserialize_field(arg, value)
                    except (ValueError, TypeError):
                        continue
                return value
            elif is_dataclass(field_type):
                return deserialize(field_type, value)
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                return field_type(value)  # Convert value back to Enum
            else:
                return value

        return deserialize(cls, data)


class Role(Enum):
    """
    Enumeration of roles in the conversation.
    """

    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"
    USER = "user"

    def __str__(self):
        """Return the value of the enum member."""
        return self.value


class MessageType(Enum):
    """
    Enum to represent the type of message
    """

    CONDUCTOR = "conductor"
    CONVERSATION = "conversation"

    def __str__(self):
        """
        Override the default string representation to return the enum's value.
        """
        return self.value


GenericConductorLongtermMemory = TypeVar("GenericConductorLongtermMemory", bound="ConductorLongtermMemory")

# TODO: Implement to string method for entities i.e. __format__


@dataclass
class ToolCall(JsonSerializable):
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallResult:
    tool_call_id: str
    function_name: str
    result: Any


@dataclass
class ChatMessage(JsonSerializable):
    role: Role
    content: Optional[str] = ""
    name: Optional[str] = ""
    tool_call_id: Optional[str] = ""
    tool_calls: Optional[list[ToolCall]] = None
    type: MessageType = MessageType.CONVERSATION

    def __post_init__(self):
        if isinstance(self.role, str):
            self.role = Role(self.role)

        if isinstance(self.type, str):
            self.type = MessageType(self.type)

        if self.tool_calls and len(self.tool_calls) > 0 and isinstance(self.tool_calls[0], dict):
            self.tool_calls = [ToolCall.from_dict(tc) for tc in self.tool_calls]


# Field metadata classes for access mode information
class InputFieldMetadata:
    """Metadata for input fields."""

    def is_input(self) -> bool:
        return True

    def is_output(self) -> bool:
        return False

    def is_locked(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "InputFieldMetadata()"


class OutputFieldMetadata:
    """Metadata for output fields."""

    def is_input(self) -> bool:
        return False

    def is_output(self) -> bool:
        return True

    def is_locked(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "OutputFieldMetadata()"


class LockedFieldMetadata:
    """Metadata for locked fields."""

    def is_input(self) -> bool:
        return False

    def is_output(self) -> bool:
        return False

    def is_locked(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "LockedFieldMetadata()"


# Special generic classes that work both as runtime values and type annotations
class InputField(Generic[T]):
    """
    Field that accepts user input.

    This can be used both as a type annotation and with an assigned value:

    Example:
        >>> @dataclass
        >>> class MyPrompt(PromptArgument):
        ...     # Field with input mode (type annotation only)
        ...     input_field: InputField[bool] = False
    """

    # This allows Pylance to treat the class as a proper Generic
    __slots__ = ()

    # This tells Python what the actual runtime type is
    @classmethod
    def __class_getitem__(cls, item: type) -> Any:
        """Support for InputField[Type] syntax."""
        # Create the annotated type with our access mode metadata
        return Annotated[item, InputFieldMetadata()]


class OutputField(Generic[T]):
    """
    Field for output-only values.

    This can be used both as a type annotation and with an assigned value:

    Example:
        >>> @dataclass
        >>> class MyPrompt(PromptArgument):
        ...     # Field with output mode (type annotation only)
        ...     output_field: OutputField[str] = "default value"
    """

    __slots__ = ()

    @classmethod
    def __class_getitem__(cls, item: type) -> Any:
        """Support for OutputField[Type] syntax."""
        return Annotated[item, OutputFieldMetadata()]


class LockedField(Generic[T]):
    """
    Field for locked values that cannot be changed.

    This can be used both as a type annotation and with an assigned value:

    Example:
        >>> @dataclass
        >>> class MyPrompt(PromptArgument):
        ...     # Field with locked mode (type annotation only)
        ...     locked_field: LockedField[int] = 42
    """

    __slots__ = ()

    @classmethod
    def __class_getitem__(cls, item: type) -> Any:
        """Support for LockedField[Type] syntax."""
        return Annotated[item, LockedFieldMetadata()]


@dataclass
class PromptArgument(JsonSerializable):
    pass


@dataclass
class LongtermMemory(JsonSerializable):
    user_query: str


@dataclass
class DiskursInput(JsonSerializable):
    metadata: Optional[dict[str, Any]] = field(default_factory=dict)
    user_query: Optional[str] = ""
    conversation_id: Optional[str] = ""


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


@dataclass
class ResultHolder(JsonSerializable):
    result: Optional[dict[str, Any]] = None


@dataclass
class RoutingRule(JsonSerializable):
    """A rule used for deterministic routing decisions"""

    name: str
    description: str
    condition: Callable[[Conversation], bool]
    target_agent: str
