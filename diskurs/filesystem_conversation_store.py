import json
import os
from pathlib import Path

from diskurs.protocols import ConversationStore, Conversation


class FilesystemConversationStore(ConversationStore):
    def __init__(self, directory: Path, agents: list, conversation_class: Conversation):
        self.directory: Path = directory
        self.agents = agents
        self.conversation_class = conversation_class
        self.directory.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, conversation_id: str) -> Path:
        return self.directory / f"{conversation_id}.json"

    def persist(self, conversation: Conversation) -> None:
        assert conversation.conversation_id, "Conversation ID must be set before persisting"
        file_path = self._get_file_path(conversation.conversation_id)
        with open(file_path, "w") as f:
            json.dump(conversation.to_dict(), f)

    def fetch(self, conversation_id: str) -> Conversation:
        file_path = self._get_file_path(conversation_id)
        with open(file_path, "r") as f:
            data = json.load(f)
        return self.conversation_class.from_dict(data=data, agents=self.agents)

    def delete(self, conversation_id: str) -> None:
        file_path = self._get_file_path(conversation_id)

        if file_path.exists():
            os.remove(file_path)

    def exists(self, conversation_id: str) -> bool:
        file_path = self._get_file_path(conversation_id)
        return os.path.exists(file_path)
