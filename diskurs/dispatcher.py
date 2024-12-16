from diskurs.logger_setup import get_logger
from diskurs.protocols import (
    ConversationParticipant,
    ConversationDispatcher,
    Conversation,
    ConversationFinalizer,
    ConversationResponder,
)
from diskurs.registry import register_dispatcher


@register_dispatcher("asynchronous")
class AsynchronousConversationDispatcher(ConversationDispatcher):
    def __init__(self):
        self._topics: dict[str, list[ConversationParticipant | ConversationFinalizer | ConversationResponder]] = {}
        self.logger = get_logger(f"diskurs.{__name__}")

        self.logger.info(f"Initializing asynchronous conversation dispatcher")

    def subscribe(
        self, topic: str, subscriber: ConversationParticipant | ConversationFinalizer | ConversationResponder
    ) -> None:
        self.logger.debug(f"Subscribing {subscriber.name} to topic {topic}")

        if topic not in self._topics:
            self._topics[topic] = []
        if subscriber not in self._topics[topic]:
            self._topics[topic].append(subscriber)

    def unsubscribe(
        self, topic: str, subscriber: ConversationParticipant | ConversationFinalizer | ConversationResponder
    ) -> None:
        self.logger.debug(f"Unsubscribing {subscriber.name} from topic {topic}")
        if topic in self._topics:
            self._topics[topic].remove(subscriber)

    async def publish(self, topic: str, conversation: Conversation) -> None:
        self.logger.debug(f"Publishing conversation to topic {topic}")

        if topic in self._topics:
            for agent in self._topics[topic]:
                await agent.process_conversation(conversation)
        else:
            self.logger.error(f"No subscribers for topic {topic}")
            raise ValueError(f"No subscribers for topic {topic}")

    async def publish_final(self, topic: str, conversation: Conversation) -> None:
        self.logger.debug(f"Publishing to generate final answer to topic {topic}")

        if topic in self._topics:
            # TODO: we assume that the topic is the name of the finalizer, this implementation cannot
            #  handle cases with multiple finalizers neither would I know how to implement that
            await self._topics[topic][0].finalize_conversation(conversation)

    async def request_response(self, topic: str, conversation: Conversation) -> Conversation:
        updated_conversation = await self._topics[topic][0].respond(conversation=conversation)
        return updated_conversation

    async def run(self, participant: ConversationParticipant, conversation: Conversation) -> dict:
        await participant.process_conversation(conversation=conversation)

        return conversation.final_result
