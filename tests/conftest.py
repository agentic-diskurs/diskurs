import enum
import inspect
from dataclasses import dataclass
from typing import Type, Optional
from unittest.mock import Mock, AsyncMock

import pytest

from diskurs import (
    LongtermMemory,
    PromptArgument,
    ConductorAgent,
    ImmutableConversation,
    MultistepPrompt,
)
from diskurs.entities import ChatMessage, Role, MessageType
from diskurs.heuristic_agent import HeuristicAgent
from diskurs.protocols import ConductorPrompt


@dataclass
class MyLongtermMemory(LongtermMemory):
    field1: str = ""
    field2: str = ""
    field3: str = ""


class MyLongtermMemory2(LongtermMemory):
    field4: str = ""
    field5: str = ""
    field6: str = ""


@dataclass
class MyUserPromptArgument(PromptArgument):
    field1: str = ""
    field2: str = ""
    field3: str = ""


class MySystemPromptArgument(PromptArgument):
    field1: str = ""
    field2: str = ""


@pytest.fixture
def prompt_arguments():
    return MyUserPromptArgument, MySystemPromptArgument


@pytest.fixture
def longterm_memories():
    return MyLongtermMemory, MyLongtermMemory2


def create_conductor_mock(name, system_prompt_argument, prompt_argument, longterm_memory):
    agent_mock = Mock(spec=ConductorAgent)
    prompt = Mock(spec=ConductorPrompt)
    prompt.system_prompt_argument = system_prompt_argument
    prompt.prompt_argument = prompt_argument
    prompt.longterm_memory = longterm_memory
    agent_mock.prompt = prompt
    agent_mock.name = name

    return agent_mock


@pytest.fixture
def conductor_mock():
    return create_conductor_mock(
        name="my_conductor",
        system_prompt_argument=MySystemPromptArgument(),
        prompt_argument=MyUserPromptArgument(),
        longterm_memory=MyLongtermMemory,
    )


@pytest.fixture()
def conductor_mock2():
    return create_conductor_mock(
        name="my_conductor_2",
        system_prompt_argument=MySystemPromptArgument(),
        prompt_argument=MyUserPromptArgument(),
        longterm_memory=MyLongtermMemory2,
    )


@pytest.fixture
def conversation():
    conversation = ImmutableConversation(
        prompt_argument=MyUserPromptArgument(
            field1="user prompt field 1",
            field2="user prompt field 2",
            field3="user prompt field 3",
        ),
        chat=[
            ChatMessage(role=Role.USER, content="Hello, world!", name="Alice"),
            ChatMessage(
                role=Role.USER, content="{'next_agent': 'my_agent'}", name="Alice", type=MessageType.CONDUCTOR
            ),
        ],
        longterm_memory={
            "my_conductor": MyLongtermMemory(
                field1="longterm_val1",
                field2="longterm_val2",
                field3="longterm_val3",
                user_query="How's the weather?",
            ),
            "my_conductor_2": MyLongtermMemory2(
                user_query="How's the aquarium?",
            ),
        },
        active_agent="my_conductor",
        conversation_id="my_conversation_id",
    )
    return conversation


def are_classes_structurally_similar(class_a: Type, class_b: Type) -> bool:
    """
    Check if two classes are structurally similar, i.e. they have the same class name, parent classes, and fields.
    This can be useful when checking if two classes are the same, but one has been dynamically imported
    :param class_a: The first class to compare
    :param class_b: The second class to compare
    :return: True if the classes are structurally similar, False otherwise
    """
    # Check class names
    if class_a.__name__ != class_b.__name__:
        return False

    # Check parent classes - look for similar class names rather than identity
    parents_a_names = {parent.__name__ for parent in inspect.getmro(class_a)}
    parents_b_names = {parent.__name__ for parent in inspect.getmro(class_b)}
    if not parents_a_names.intersection(parents_b_names):
        return False

    # Get type hints and attributes of each class
    type_hints_a = inspect.get_annotations(class_a)
    type_hints_b = inspect.get_annotations(class_b)

    # Check field names
    if type_hints_a.keys() != type_hints_b.keys():
        return False

    # Compare type hints by name rather than identity
    for field, type_hint_a in type_hints_a.items():
        type_hint_b = type_hints_b.get(field)

        # Get the string representation of both types for comparison
        type_a_str = str(type_hint_a)
        type_b_str = str(type_hint_b)

        # Strip module names for comparison to handle dynamically imported types
        # Example: convert 'module1.List[module1.Step]' to 'List[Step]'
        type_a_stripped = type_a_str.split(".")[-1]
        type_b_stripped = type_b_str.split(".")[-1]

        # Handle generic types like List[X] by comparing the base names
        if "[" in type_a_stripped and "[" in type_b_stripped:
            type_a_base = type_a_stripped.split("[")[0]
            type_b_base = type_b_stripped.split("[")[0]

            # If the generic container types match (like 'List'), check contents
            if type_a_base == type_b_base:
                # Extract what's inside the brackets (e.g., 'Step' from 'List[Step]')
                type_a_inner = type_a_stripped[type_a_stripped.find("[") + 1 : type_a_stripped.rfind("]")]
                type_b_inner = type_b_stripped[type_b_stripped.find("[") + 1 : type_b_stripped.rfind("]")]

                # Compare the inner types by name
                if type_a_inner.split(".")[-1] != type_b_inner.split(".")[-1]:
                    return False
            else:
                return False
        # For non-generic types, compare the stripped type names
        elif type_a_stripped != type_b_stripped:
            return False

    return True


def create_prompt(prompt_argument):
    prompt = AsyncMock(spec=MultistepPrompt)
    prompt.create_system_prompt_argument.return_value = AsyncMock()
    prompt.create_prompt_argument.return_value = prompt_argument
    prompt.render_user_template.return_value = ChatMessage(
        role=Role.USER,
        name="my_multistep",
        content="rendered template",
        type=MessageType.CONVERSATION,
    )
    prompt.is_final.return_value = True
    prompt.prompt_argument = prompt_argument

    return prompt


@pytest.fixture
def mock_prompt():
    return create_prompt(MyUserPromptArgument())


@dataclass
class MyExtendedLongtermMemory(LongtermMemory):
    field2: str = ""
    field3: str = ""


@dataclass
class MySourceUserPromptArgument(PromptArgument):
    field3: str = ""
    field4: str = ""


@dataclass
class MyExtendedUserPromptArgument(PromptArgument):
    field1: str = "extended user prompt field 1"
    field2: str = "extended user prompt field 2"
    field3: str = "extended user prompt field 3"
    field4: str = "extended user prompt field 4"


@pytest.fixture
def mock_extended_prompt():
    return create_prompt(MyExtendedUserPromptArgument())


@pytest.fixture
def extended_conversation():
    conversation = ImmutableConversation(
        prompt_argument=MySourceUserPromptArgument(
            field3="user prompt field 3",
            field4="user prompt field 4",
        ),
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
        longterm_memory={
            "my_conductor": MyExtendedLongtermMemory(
                field2="longterm val 2",
                field3="longterm val 3",
                user_query="longterm user query",
            ),
        },
        active_agent="my_conductor",
        conversation_id="my_conversation_id",
    )
    return conversation


@pytest.fixture
def finalizer_conversation():
    conversation = ImmutableConversation(
        prompt_argument=MyUserPromptArgument(field1="user prompt field 1", field2="user prompt field 2"),
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
        longterm_memory={
            "my_conductor": MyLongtermMemory(user_query="longterm user query"),
        },
        active_agent="my_conductor",
        conversation_id="my_conversation_id",
    )
    return conversation


@pytest.fixture
def heuristic_agent_mock():
    mock = Mock(spec=HeuristicAgent)
    mock.name = "heuristic_agent"
    prompt_mock = Mock()
    # Ensure prompt doesn't have system_prompt_argument
    if hasattr(prompt_mock, "system_prompt_argument"):
        delattr(prompt_mock, "system_prompt_argument")
    mock.prompt = prompt_mock
    return mock


@pytest.fixture
def conversation_dict():
    return {
        "system_prompt": None,
        "user_prompt": None,
        "system_prompt_argument": None,
        "prompt_argument": None,
        "chat": [],
        "longterm_memory": {},
        "metadata": {},
        "active_agent": "my_conductor",
        "conversation_id": "",
    }


class ChatType(enum.Enum):
    """Test enum to verify serialization behavior"""

    DIRECT = "direct"
    GROUP = "group"
    CHANNEL = "channel"


class Priority(enum.Enum):
    """Test numeric enum"""

    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class EnumPromptArgument(PromptArgument):
    """Test prompt argument with various enum fields"""

    chat_type: ChatType = ChatType.DIRECT
    priority: Priority = Priority.MEDIUM
    message_type: Optional[MessageType] = None


@dataclass
class EnumLongtermMemory(LongtermMemory):
    """Test longterm memory with enum fields"""

    user_query: str = ""
    preferred_chat_type: ChatType = ChatType.DIRECT
