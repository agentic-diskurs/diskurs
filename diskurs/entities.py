from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, replace, fields, is_dataclass
from enum import Enum
from typing import Annotated, TypeVar, TypeAlias, get_type_hints
from typing import Any, Callable, Optional, Union, get_args, get_origin, TYPE_CHECKING


class AccessMode(Enum):
    """
    Enumeration representing different access modes or states for fields.
    """

    INPUT = "input"
    OUTPUT = "output"
    LOCKED = "locked"
    PER_TURN = "per_turn"

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


T = TypeVar("T")


# This is our metadata container.
class PromptField:
    def __init__(self, access_mode: str) -> None:
        self.access_mode = access_mode

    def __repr__(self) -> str:
        return f"PromptField(access_mode={self.access_mode!r})"

    def is_input(self) -> bool:
        """Check if this field has INPUT access mode."""
        return self.access_mode == AccessMode.INPUT.value

    def is_output(self) -> bool:
        """Check if this field has OUTPUT access mode."""
        return self.access_mode == AccessMode.OUTPUT.value

    def is_locked(self) -> bool:
        """Check if this field has LOCKED access mode."""
        return self.access_mode == AccessMode.LOCKED.value

    def is_per_turn(self) -> bool:
        """Check if this field should be reset between turns."""
        return self.access_mode == AccessMode.PER_TURN.value


# For type checking purposes, define these types as just the generic type
if TYPE_CHECKING:

    V = TypeVar("V")
    # Treat InputField[T] as just T under type checking
    InputField: TypeAlias = V
    OutputField: TypeAlias = V
    LockedField: TypeAlias = V
    PerTurnField: TypeAlias = V

else:
    # Runtime implementation
    class InputField:
        """Input field type for dataclasses."""

        def __new__(cls, value=None):
            return value

        @classmethod
        def __class_getitem__(cls, t):
            return Annotated[t, PromptField("input")]

    class OutputField:
        """Output field type for dataclasses."""

        def __new__(cls, value=None):
            return value

        @classmethod
        def __class_getitem__(cls, t):
            return Annotated[t, PromptField("output")]

    class LockedField:
        """Locked field type for dataclasses."""

        def __new__(cls, value=None):
            return value

        @classmethod
        def __class_getitem__(cls, t):
            return Annotated[t, PromptField("locked")]

    class PerTurnField:
        """Field that resets between conversation turns."""

        def __new__(cls, value=None):
            return value

        @classmethod
        def __class_getitem__(cls, t):
            return Annotated[t, PromptField("per_turn")]


@dataclass
class PromptArgument(JsonSerializable):
    def init(self, source: Union["PromptArgument", LongtermMemory]) -> "PromptArgument":
        """
        Initialize a new instance by reading the fields of another PromptArgument or LongtermMemory.
        Only fields of type InputField will be copied from the source.

        :param source: The source object to copy fields from
        :return: A new instance with InputField values copied from source
        """
        if not source:
            return self

        common_fields = {f.name for f in fields(self)}.intersection({f.name for f in fields(source)})
        update_values = {}

        # Get type hints with metadata for the current class
        hints = get_type_hints(self.__class__, include_extras=True)

        for field_name in common_fields:
            # Check if the field has InputField metadata
            field_type = hints.get(field_name)

            if field_type and hasattr(field_type, "__metadata__"):
                for metadata in field_type.__metadata__:
                    if isinstance(metadata, PromptField) and metadata.is_input():
                        # Only copy InputField fields
                        update_values[field_name] = getattr(source, field_name)
                        break

        # Create a new instance with updated values
        return self.__class__(**{**asdict(self), **update_values})

    def update(self, other: PromptArgument) -> PromptArgument:
        """
        Return a new instance with all non-locked/non-input fields
        taken from `other`.
        """
        hints = get_type_hints(self.__class__, include_extras=True)
        updates = {
            f.name: val
            for f in fields(self)
            if not any(
                isinstance(m, PromptField) and (m.is_locked() or m.is_input())
                for m in getattr(hints.get(f.name), "__metadata__", ())
            )
            and (val := getattr(other, f.name)) not in (None, "")
        }
        return replace(self, **updates)

    def get_output_fields(self) -> dict[str, Any]:
        """
        Returns a dictionary containing only the fields that should be included in JSON output.
        This includes fields with OutputField annotation and fields with no specific annotation.

        :return: Dictionary of field names to values that should be included in output
        """
        hints = get_type_hints(self.__class__, include_extras=True)

        # Filter fields for output
        output_fields = {}
        for field_info in fields(self):
            key = field_info.name
            value = getattr(self, key)
            hint = hints.get(key)
            include_field = True

            if hint and hasattr(hint, "__metadata__"):
                include_field = False
                for metadata in hint.__metadata__:
                    if isinstance(metadata, PromptField) and metadata.is_output():
                        include_field = True
                        break

            if include_field:
                output_fields[key] = value

        return output_fields


@dataclass
class LongtermMemory(JsonSerializable):
    user_query: str = ""

    def update(self, prompt_argument: PromptArgument | LongtermMemory) -> "LongtermMemory":
        """
        Update fields in the longterm memory using values from a PromptArgument.
        Only fields of type OutputField in the PromptArgument will be copied to
        the LongtermMemory if they have matching names.

        :param prompt_argument: The PromptArgument to copy fields from
        :return: A new LongtermMemory instance with updated fields
        """
        # If no prompt_argument provided, return self unchanged
        if prompt_argument is None:
            return self
        # Identify common fields between memory and source
        common_fields = {f.name for f in fields(self)}.intersection({f.name for f in fields(prompt_argument)})
        update_values = {}
        if isinstance(prompt_argument, PromptArgument):
            # Copy only OutputField annotated fields from PromptArgument
            hints = get_type_hints(prompt_argument.__class__, include_extras=True)
            for field_name in common_fields:
                hint = hints.get(field_name)
                if hint and hasattr(hint, "__metadata__"):
                    for metadata in hint.__metadata__:
                        if isinstance(metadata, PromptField) and metadata.is_output():
                            val = getattr(prompt_argument, field_name)
                            if val not in (None, ""):
                                update_values[field_name] = val
                            break
        else:
            # If it's another LongtermMemory, copy all non-empty fields
            for field_name in common_fields:
                val = getattr(prompt_argument, field_name)
                if val is not None and val != "":
                    update_values[field_name] = val
        return replace(self, **update_values)

    def reset_per_turn_fields(self) -> "LongtermMemory":
        """
        Reset all fields marked with PerTurnField decorator to their default values.

        :return: A new LongtermMemory instance with per-turn fields reset
        """
        hints = get_type_hints(self.__class__, include_extras=True)
        reset_values = {}

        # Find all fields that are marked as per-turn and get their default values
        for f in fields(self.__class__):
            field_type = hints.get(f.name)
            if field_type and hasattr(field_type, "__metadata__"):
                for metadata in field_type.__metadata__:
                    if isinstance(metadata, PromptField) and metadata.is_per_turn():
                        # Get default value from the field's default factory or default value
                        if f.default_factory is not MISSING:
                            reset_values[f.name] = f.default_factory()
                        elif f.default is not MISSING:
                            reset_values[f.name] = f.default
                        else:
                            # For fields with no default, use empty values based on type
                            origin_type = get_origin(get_args(field_type)[0]) if get_args(field_type) else field_type
                            if origin_type is list:
                                reset_values[f.name] = []
                            elif origin_type is dict:
                                reset_values[f.name] = {}
                            elif origin_type is str:
                                reset_values[f.name] = ""
                            else:
                                reset_values[f.name] = None
                        break

        return replace(self, **reset_values)

    @classmethod
    def from_dict(cls, data, reset_per_turn=True):
        """
        Create a new instance from a dictionary, applying special handling for per-turn fields.

        :param data: Dictionary containing serialized field values
        :param reset_per_turn: Whether to reset per-turn fields during deserialization
        :return: A new instance with fields initialized from the dictionary
        """
        if data is None:
            return None

        instance = super().from_dict(data)

        if reset_per_turn and isinstance(instance, LongtermMemory):
            instance = instance.reset_per_turn_fields()

        return instance


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
    condition: Callable[["Conversation"], bool]
    target_agent: str
