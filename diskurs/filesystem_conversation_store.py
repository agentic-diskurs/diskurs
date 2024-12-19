import json
import os
from pathlib import Path
from typing import Self
import asyncio

import aiofiles

from diskurs.logger_setup import get_logger
from diskurs.protocols import ConversationStore, Conversation
from diskurs.registry import register_conversation_store


@register_conversation_store("filesystem")
class AsyncFilesystemConversationStore(ConversationStore):
    def __init__(self, directory: Path, agents: list, conversation_class: Conversation):
        self.directory: Path = directory
        self.agents = agents
        self.conversation_class = conversation_class
        self.logger = get_logger(f"diskurs.{__name__}")

        self.logger.info(f"Initializing async conversation store with directory {directory}")

    def _get_file_path(self, conversation_id: str) -> Path:
        return self.directory / f"{conversation_id}.json"

    @classmethod
    def create(cls, **kwargs) -> Self:
        # Directory creation is fast and synchronous, but if you want, you could run it in a thread pool.
        directory = kwargs.get("directory")
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        return cls(**kwargs)

    async def persist(self, conversation: Conversation) -> None:
        assert conversation.conversation_id, "Conversation ID must be set before persisting"

        self.logger.info(f"Persisting conversation {conversation.conversation_id}")

        file_path = self._get_file_path(conversation.conversation_id)
        data = json.dumps(conversation.to_dict())

        async with aiofiles.open(file_path, "w") as f:
            await f.write(data)

    async def fetch(self, conversation_id: str) -> Conversation:
        self.logger.info(f"Fetching conversation {conversation_id}")

        file_path = self._get_file_path(conversation_id)
        async with aiofiles.open(file_path, "r") as f:
            data_str = await f.read()

        data = json.loads(data_str)
        return self.conversation_class.from_dict(data=data, agents=self.agents)

    async def delete(self, conversation_id: str) -> None:
        self.logger.info(f"Deleting conversation {conversation_id}")

        file_path = self._get_file_path(conversation_id)
        if file_path.exists():
            # Removing a file could also be done in a thread pool if desired:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, os.remove, file_path)

    async def exists(self, conversation_id: str) -> bool:
        file_path = self._get_file_path(conversation_id)
        # Checking existence is a quick operation, but to maintain consistency,
        # we can just do it synchronously here.
        return file_path.exists()
