import json
import os
from pathlib import Path
from typing import Self

from diskurs.logger_setup import get_logger
from diskurs.protocols import ConversationStore, Conversation
from diskurs.registry import register_conversation_store


@register_conversation_store("filesystem")
class FilesystemConversationStore(ConversationStore):
    def __init__(self, directory: Path, agents: list, conversation_class: Conversation):
        self.directory: Path = directory
        self.agents = agents
        self.conversation_class = conversation_class
        self.logger = get_logger(f"diskurs.{__name__}")

        self.logger.info(f"Initializing conversation store with directory {directory}")

    def _get_file_path(self, conversation_id: str) -> Path:
        return self.directory / f"{conversation_id}.json"

    @classmethod
    def create(cls, **kwargs) -> Self:
        if directory := kwargs.get("directory"):
            directory.mkdir(parents=True, exist_ok=True)

        return cls(**kwargs)

    def persist(self, conversation: Conversation) -> None:
        assert conversation.conversation_id, "Conversation ID must be set before persisting"

        self.logger.info(f"Persisting conversation {conversation.conversation_id}")

        file_path = self._get_file_path(conversation.conversation_id)
        with open(file_path, "w") as f:
            json.dump(conversation.to_dict(), f)

    def fetch(self, conversation_id: str) -> Conversation:
        self.logger.info(f"Fetching conversation {conversation_id}")

        file_path = self._get_file_path(conversation_id)
        with open(file_path, "r") as f:
            data = json.load(f)
        return self.conversation_class.from_dict(data=data, agents=self.agents)

    def delete(self, conversation_id: str) -> None:
        self.logger.info(f"Deleting conversation {conversation_id}")

        file_path = self._get_file_path(conversation_id)

        if file_path.exists():
            os.remove(file_path)

    def exists(self, conversation_id: str) -> bool:
        file_path = self._get_file_path(conversation_id)
        return os.path.exists(file_path)
