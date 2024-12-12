from diskurs import ImmutableConversation
from diskurs.agent import is_previous_agent_conductor, get_last_conductor_name
from diskurs.entities import ChatMessage, Role, MessageType


def test_is_previous_agent_conductor_empty_conversation():
    conversation = ImmutableConversation()  # Assuming Conversation can be instantiated empty
    assert is_previous_agent_conductor(conversation) == False


def test_is_previous_agent_conductor_with_conductor():
    conversation = ImmutableConversation()
    conductor_message = ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, content="conductor message")
    conversation = conversation.append(conductor_message)
    assert is_previous_agent_conductor(conversation) == True


def test_is_previous_agent_conductor_with_non_conductor():
    conversation = ImmutableConversation()
    regular_message = ChatMessage(role=Role.ASSISTANT, type=MessageType.CONVERSATION, content="regular message")
    conversation = conversation.append(regular_message)
    assert is_previous_agent_conductor(conversation) == False


def test_get_last_conductor_name_empty_chat():
    chat: list[ChatMessage] = []
    assert get_last_conductor_name(chat) is None


def test_get_last_conductor_name_no_conductor():
    chat = [
        ChatMessage(role=Role.ASSISTANT, type=MessageType.CONVERSATION, name="agent1"),
        ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"),
    ]
    assert get_last_conductor_name(chat) is None


def test_get_last_conductor_name_single_conductor():
    chat = [
        ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor1"),
        ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"),
    ]
    assert get_last_conductor_name(chat) == "conductor1"


def test_get_last_conductor_name_multiple_conductors():
    chat = [
        ChatMessage(role=Role.ASSISTANT, type=MessageType.CONVERSATION, name="agent1"),
        ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor2"),
        ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"),
        ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor1"),
    ]
    assert get_last_conductor_name(chat) == "conductor1"
