import json

from diskurs.protocols import ConversationStore, Conversation


class LocalConversationStore(ConversationStore):
    def __init__(self, conversation_store_path: str):
        self.conversation_store_path = conversation_store_path
        self.conversations = {}

    def load_conversations(self):
        with open(self.conversation_store_path, "r") as conversation_store_file:
            self.conversations = json.load(conversation_store_file)

    def save_conversations(self):
        with open(self.conversation_store_path, "w") as conversation_store_file:
            json.dump(self.conversations, conversation_store_file)

    def get_conversation(self, conversation_id: str) -> Conversation:
        return self.conversations.get(conversation_id)

    def add_conversation(self, conversation: Conversation):
        self.conversations[conversation.id] = conversation

    def remove_conversation(self, conversation_id: str):
        self.conversations.pop(conversation_id)
