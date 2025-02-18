from diskurs import ImmutableConversation


def test_conversation_to_dict(conversation):

    conversation_dict = conversation.to_dict()

    assert isinstance(conversation_dict, dict)
    assert isinstance(conversation_dict["chat"], list)
    assert conversation_dict["chat"][0]["role"] == "user"
    assert conversation_dict["chat"][0]["content"] == "Hello, world!"
    assert conversation_dict["longterm_memory"]["my_conductor"]["field1"] == "longterm_val1"
    assert conversation_dict["active_agent"] == "my_conductor"


def test_conversation_from_dict(conversation, conductor_mock, conductor_mock2):

    conversation_dict = conversation.to_dict()
    new_conversation = ImmutableConversation.from_dict(
        data=conversation_dict, agents=[conductor_mock, conductor_mock2]
    )

    assert new_conversation.chat[0].role == conversation.chat[0].role
    assert new_conversation.chat[0].content == conversation.chat[0].content
    assert (
        new_conversation._longterm_memory["my_conductor"].field1
        == conversation._longterm_memory["my_conductor"].field1
    )
    assert (
        new_conversation._longterm_memory["my_conductor"].field2
        == conversation._longterm_memory["my_conductor"].field2
    )
    assert (
        new_conversation._longterm_memory["my_conductor"].field3
        == conversation._longterm_memory["my_conductor"].field3
    )
    assert (
        new_conversation._longterm_memory["my_conductor"].user_query
        == conversation._longterm_memory["my_conductor"].user_query
    )
    assert new_conversation.user_prompt_argument.field1 == conversation.user_prompt_argument.field1
    assert new_conversation.user_prompt_argument.field2 == conversation.user_prompt_argument.field2
    assert new_conversation.user_prompt_argument.field3 == conversation.user_prompt_argument.field3
    assert new_conversation.user_prompt == conversation.user_prompt
    assert new_conversation.system_prompt == conversation.system_prompt
    assert new_conversation.active_agent == conversation.active_agent


def test_conversation_from_dict_with_heuristic_agent(heuristic_agent_mock, conversation_dict):
    """Test that from_dict works correctly with an agent that has no system_prompt_argument."""
    # Modify conversation_dict to not include system_prompt_argument
    conversation_dict["active_agent"] = "heuristic_agent"
    conversation_dict["system_prompt_argument"] = None

    new_conversation = ImmutableConversation.from_dict(
        data=conversation_dict,
        agents=[heuristic_agent_mock]
    )

    assert new_conversation.system_prompt_argument is None
    assert new_conversation.active_agent == "heuristic_agent"

def test_conversation_from_dict_missing_system_prompt_argument(conductor_mock):
    """Test that from_dict handles missing system_prompt_argument in data dict."""
    conversation_dict = {
        "system_prompt": None,
        "user_prompt": None,
        # system_prompt_argument intentionally omitted
        "user_prompt_argument": None,
        "chat": [],
        "longterm_memory": {},
        "metadata": {},
        "active_agent": "my_conductor",
        "conversation_id": ""
    }

    new_conversation = ImmutableConversation.from_dict(
        data=conversation_dict,
        agents=[conductor_mock]
    )

    assert new_conversation.system_prompt_argument is None

def test_conversation_from_dict_none_values(conductor_mock):
    """Test that from_dict properly handles None values for all optional fields."""
    conversation_dict = {
        "system_prompt": None,
        "user_prompt": None,
        "system_prompt_argument": None,
        "user_prompt_argument": None,
        "chat": [],
        "longterm_memory": {},
        "metadata": {},
        "active_agent": "my_conductor"
    }

    new_conversation = ImmutableConversation.from_dict(
        data=conversation_dict,
        agents=[conductor_mock]
    )

    assert new_conversation.system_prompt is None
    assert new_conversation.user_prompt is None
    assert new_conversation.system_prompt_argument is None
    assert new_conversation.user_prompt_argument is None
    assert new_conversation.chat == []
    assert new_conversation.metadata == {}
