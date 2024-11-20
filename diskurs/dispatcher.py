from diskurs.logger_setup import get_logger
from diskurs.protocols import (
    ConversationParticipant,
    ConversationDispatcher,
    Conversation,
)
from diskurs.registry import register_dispatcher


@register_dispatcher("synchronous")
class SynchronousConversationDispatcher(ConversationDispatcher):
    def __init__(self):
        self._topics: dict[str, list[ConversationParticipant]] = {}
        self.logger = get_logger(f"diskurs.{__name__}")

        self.logger.info(f"Initializing synchronous conversation dispatcher")

    def subscribe(self, topic: str, subscriber: ConversationParticipant) -> None:
        self.logger.debug(f"Subscribing {subscriber.name} to topic {topic}")

        if topic not in self._topics:
            self._topics[topic] = []
        if subscriber not in self._topics[topic]:
            self._topics[topic].append(subscriber)

    def unsubscribe(self, topic: str, subscriber: ConversationParticipant) -> None:
        self.logger.debug(f"Unsubscribing {subscriber.name} from topic {topic}")
        if topic in self._topics:
            self._topics[topic].remove(subscriber)

    def publish(self, topic: str, conversation: Conversation) -> None:
        self.logger.debug(f"Publishing conversation to topic {topic}")

        if topic in self._topics:
            for agent in self._topics[topic]:
                agent.process_conversation(conversation)
        else:
            self.logger.error(f"No subscribers for topic {topic}")
            raise ValueError(f"No subscribers for topic {topic}")

    def run(self, participant: ConversationParticipant, conversation: Conversation) -> dict:
        participant.process_conversation(conversation=conversation)

        return conversation.final_result
