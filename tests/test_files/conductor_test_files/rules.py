from diskurs.protocols import Conversation

def test_rule_condition(conversation: Conversation) -> bool:
    """Simple test rule that always returns True"""
    return True

def test_rule_condition_false(conversation: Conversation) -> bool:
    """Simple test rule that always returns False"""
    return False

def check_metadata_field(conversation: Conversation) -> bool:
    """Rule that checks if a specific field exists in metadata"""
    return bool(conversation.metadata.get("test_field"))