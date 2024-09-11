from entities import Conversation
from interfaces import ConversationParticipant, ConversationDispatcher


class SynchronousConversationDispatcher(ConversationDispatcher):
    def __init__(self):
        self._topics = {}

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

    def publish(self, topic: str, conversation: Conversation) -> None:
        """Publish a conversation to all agents subscribed to the topic."""
        if topic in self._topics:
            for agent in self._topics[topic]:
                agent.process_conversation(conversation)
