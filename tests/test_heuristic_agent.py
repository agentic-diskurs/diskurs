import pytest
from unittest.mock import MagicMock

from diskurs.entities import MessageType
from diskurs.heuristic_agent import HeuristicAgent
from diskurs import Conversation, PromptArgument


def test_create_method():
    name = "test_agent"
    prompt = MagicMock()
    topics = ["topic1", "topic2"]
    dispatcher = MagicMock()
    tool_executor = MagicMock()
    render_prompt = True

    agent = HeuristicAgent.create(
        name=name,
        prompt=prompt,
        topics=topics,
        dispatcher=dispatcher,
        tool_executor=tool_executor,
        render_prompt=render_prompt,
    )

    assert agent.name == name
    assert agent.prompt == prompt
    assert agent.topics == topics
    assert agent.dispatcher == dispatcher
    assert agent.tool_executor == tool_executor
    assert agent.render_prompt == render_prompt


def test_get_conductor_name():
    name = "test_agent"
    prompt = MagicMock()
    topics = ["conductor1", "topic2"]

    agent = HeuristicAgent(name=name, prompt=prompt, topics=topics)

    assert agent.get_conductor_name() == "conductor1"


def test_register_dispatcher():
    name = "test_agent"
    prompt = MagicMock()
    dispatcher = MagicMock()

    agent = HeuristicAgent(name=name, prompt=prompt)
    agent.register_dispatcher(dispatcher)

    assert agent.dispatcher == dispatcher


def test_prepare_conversation():
    name = "test_agent"
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    user_prompt_argument = MagicMock(spec=PromptArgument)
    updated_conversation = MagicMock()

    conversation.update.return_value = updated_conversation

    agent = HeuristicAgent(name=name, prompt=prompt)
    result = agent.prepare_conversation(conversation, user_prompt_argument)

    conversation.update.assert_called_once_with(user_prompt_argument=user_prompt_argument, active_agent=name)
    assert result == updated_conversation


def test_invoke():
    name = "test_agent"
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    updated_conversation = MagicMock(spec=Conversation)
    updated_conversation_with_memory = MagicMock(spec=Conversation)
    heuristic_sequence_conversation = MagicMock(spec=Conversation)
    final_conversation = MagicMock(spec=Conversation)
    tool_executor = MagicMock()
    prompt.create_user_prompt_argument.return_value = MagicMock()

    agent = HeuristicAgent(
        name=name,
        prompt=prompt,
        tool_executor=tool_executor,
        topics=["conductor_name"],  # Provide a non-empty topics list
        init_prompt_arguments_with_longterm_memory=True,
        render_prompt=True,
    )

    agent.prepare_conversation = MagicMock(return_value=updated_conversation)
    updated_conversation.update_prompt_argument_with_longterm_memory.return_value = updated_conversation_with_memory
    prompt.heuristic_sequence.return_value = heuristic_sequence_conversation
    prompt.render_user_template.return_value = "rendered_template"
    heuristic_sequence_conversation.append.return_value = final_conversation

    result = agent.invoke(conversation)

    agent.prepare_conversation.assert_called_once_with(
        conversation=conversation, user_prompt_argument=prompt.create_user_prompt_argument.return_value
    )
    updated_conversation.update_prompt_argument_with_longterm_memory.assert_called_once_with(
        conductor_name=agent.get_conductor_name()
    )
    prompt.heuristic_sequence.assert_called_once_with(
        updated_conversation_with_memory, call_tool=tool_executor.call_tool
    )
    prompt.render_user_template.assert_called_once_with(
        agent.name,
        prompt_args=heuristic_sequence_conversation.user_prompt_argument,
        message_type=MessageType.CONVERSATION,
    )
    heuristic_sequence_conversation.append.assert_called_once_with(name=agent.name, message="rendered_template")
    assert result == final_conversation


def test_process_conversation():
    name = "test_agent"
    prompt = MagicMock()
    conversation = MagicMock(spec=Conversation)
    dispatcher = MagicMock()
    updated_conversation = MagicMock(spec=Conversation)

    agent = HeuristicAgent(name=name, prompt=prompt, dispatcher=dispatcher)
    agent.invoke = MagicMock(return_value=updated_conversation)
    agent.get_conductor_name = MagicMock(return_value="conductor_name")

    agent.process_conversation(conversation)

    agent.invoke.assert_called_once_with(conversation)
    agent.get_conductor_name.assert_called_once()
    dispatcher.publish.assert_called_once_with(topic="conductor_name", conversation=updated_conversation)
