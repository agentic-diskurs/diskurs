from concurrent.futures import Future

from diskurs.entities import Conversation, DiskursInput
from diskurs.protocols import ConversationParticipant, ConversationDispatcher
from diskurs.registry import register_dispatcher


@register_dispatcher("synchronous")
class SynchronousConversationDispatcher(ConversationDispatcher):
    def __init__(self):
        self._topics: dict[str, list[ConversationParticipant]] = {}
        self.future = Future()

    def subscribe(self, topic: str, subscriber: ConversationParticipant) -> None:
        """Subscribe an agent to a specific topic."""
        if topic not in self._topics:
            self._topics[topic] = []
        if subscriber not in self._topics[topic]:
            self._topics[topic].append(subscriber)

    def unsubscribe(self, topic: str, subscriber: ConversationParticipant) -> None:
        """Unsubscribe an agent from a specific topic."""
        if topic in self._topics:
            self._topics[topic].remove(subscriber)

    def publish(self, topic: str, conversation: Conversation, finish_diskurs: bool = False) -> None:
        """Publish a conversation to all agents subscribed to the topic."""
        if finish_diskurs:
            if not self.future.done():
                self.future.set_result(self.final_conversation)
        if topic in self._topics:
            for agent in self._topics[topic]:
                agent.process_conversation(conversation)

    def finalize(self, response: dict) -> None:
        """Finalize a diskurs by setting the future."""
        if not self.future.done():
            self.future.set_result(response)

    def run(self, participant: ConversationParticipant, diskurs_input: DiskursInput) -> dict:
        """Finish the conversation."""
        participant.start_conversation(diskurs_input)

        final_result = self.future.result()
        return final_result
