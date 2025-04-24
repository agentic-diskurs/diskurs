from diskurs import ImmutableConversation
from diskurs.entities import ChatMessage, Role, MessageType

# Remove the import statement that's causing the error
# These are methods on ImmutableConversation, not standalone functions


def test_is_previous_agent_conductor_empty_conversation():
    conversation = ImmutableConversation()  # Assuming Conversation can be instantiated empty
    assert conversation.is_previous_agent_conductor() == False


def test_is_previous_agent_conductor_with_conductor():
    conversation = ImmutableConversation()
    conductor_message = ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, content="conductor message")
    conversation = conversation.append(conductor_message)
    assert conversation.is_previous_agent_conductor() == True


def test_is_previous_agent_conductor_with_non_conductor():
    conversation = ImmutableConversation()
    regular_message = ChatMessage(role=Role.ASSISTANT, type=MessageType.CONVERSATION, content="regular message")
    conversation = conversation.append(regular_message)
    assert conversation.is_previous_agent_conductor() == False


def test_get_last_conductor_name_empty_chat():
    # This test needs to be rewritten to use the method on ImmutableConversation
    conversation = ImmutableConversation()
    assert conversation.get_last_conductor_name() is None


def test_get_last_conductor_name_no_conductor():
    # Create a conversation with non-conductor messages
    conversation = ImmutableConversation()
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONVERSATION, name="agent1"))
    conversation = conversation.append(ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"))
    assert conversation.get_last_conductor_name() is None


def test_get_last_conductor_name_single_conductor():
    # Create a conversation with a conductor message
    conversation = ImmutableConversation()
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor1"))
    conversation = conversation.append(ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"))
    assert conversation.get_last_conductor_name() == "conductor1"


def test_get_last_conductor_name_multiple_conductors():
    # Create a conversation with multiple conductor messages
    conversation = ImmutableConversation()
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONVERSATION, name="agent1"))
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor2"))
    conversation = conversation.append(ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"))
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor1"))
    assert conversation.get_last_conductor_name() == "conductor1"


def test_has_conductor_been_called_empty_chat():
    conversation = ImmutableConversation()
    assert conversation.has_conductor_been_called() is False


def test_has_conductor_been_called_no_conductor():
    # Create a conversation with non-conductor messages
    conversation = ImmutableConversation()
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONVERSATION, name="agent1"))
    conversation = conversation.append(ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"))
    assert conversation.has_conductor_been_called() is False


def test_has_conductor_been_called_single_conductor():
    # Create a conversation with a conductor message
    conversation = ImmutableConversation()
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor1"))
    conversation = conversation.append(ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"))
    assert conversation.has_conductor_been_called() is True


def test_has_conductor_been_called_multiple_conductors():
    # Create a conversation with multiple conductor messages
    conversation = ImmutableConversation()
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONVERSATION, name="agent1"))
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor2"))
    conversation = conversation.append(ChatMessage(role=Role.USER, type=MessageType.CONVERSATION, name="user1"))
    conversation = conversation.append(ChatMessage(role=Role.ASSISTANT, type=MessageType.CONDUCTOR, name="conductor1"))
    assert conversation.has_conductor_been_called() is True
