from dataclasses import dataclass

from diskurs import Conversation, PromptArgument


@dataclass
class LongtermMemory:
    field1: str = ""
    field2: str = ""
    field3: str = ""


@dataclass
class UserPromptArgument(PromptArgument):
    field1: str = ""
    field2: str = ""
    field3: str = ""


def test_basic_update():
    conversation = Conversation()
    longterm_memory = LongtermMemory(field1="longterm_val1", field2="longterm_val2", field3="longterm_val3")
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(user_prompt_argument=UserPromptArgument())

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.user_prompt_argument.field1 == "longterm_val1"
    assert updated_conversation.user_prompt_argument.field2 == "longterm_val2"
    assert updated_conversation.user_prompt_argument.field3 == "longterm_val3"


def test_partial_update():
    conversation = Conversation()
    longterm_memory = LongtermMemory(field1="longterm_val1", field3="longterm_val3")
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(
        user_prompt_argument=UserPromptArgument(field1="initial_prompt_argument1", field2="initial_prompt_argument2")
    )

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.user_prompt_argument.field1 == "longterm_val1"
    assert updated_conversation.user_prompt_argument.field2 == "initial_prompt_argument2"
    assert updated_conversation.user_prompt_argument.field3 == "longterm_val3"


def test_empty_longterm_memory():
    conversation = Conversation()
    longterm_memory = LongtermMemory()
    conversation = conversation.update_agent_longterm_memory(
        agent_name="my_conductor", longterm_memory=longterm_memory
    )

    conversation = conversation.update(
        user_prompt_argument=UserPromptArgument(field1="initial_prompt_argument1", field2="initial_prompt_argument2")
    )

    updated_conversation = conversation.update_prompt_argument_with_longterm_memory("my_conductor")

    assert updated_conversation.user_prompt_argument == conversation.user_prompt_argument  # No changes
