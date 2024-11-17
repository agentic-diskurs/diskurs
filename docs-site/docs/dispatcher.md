# Module: Dispatcher

### *class* diskurs.dispatcher.SynchronousConversationDispatcher

Bases: [`ConversationDispatcher`](protocols.md#diskurs.protocols.ConversationDispatcher)

#### subscribe(topic, subscriber)

Subscribe a participant to a specific topic.

This method registers a ConversationParticipant to receive conversations
related to the specified topic. When a conversation is published to the topic,
all subscribed participants will be notified and can process the conversation.

* **Parameters:**
  * **topic** (`str`) – The topic to which the participant will be subscribed.
  * **participant** – The participant to be subscribed to the topic.
  * **subscriber** ([*ConversationParticipant*](protocols.md#diskurs.protocols.ConversationParticipant))
* **Return type:**
  `None`

#### unsubscribe(topic, subscriber)

Unsubscribe a participant from a specific topic.

This method removes a ConversationParticipant from the list of subscribers for the given topic.
Once unsubscribed, the participant will no longer receive conversations related to that topic.

* **Parameters:**
  * **topic** (`str`) – The topic from which the participant will be unsubscribed.
  * **participant** – The participant to be unsubscribed from the topic.
  * **subscriber** ([*ConversationParticipant*](protocols.md#diskurs.protocols.ConversationParticipant))
* **Return type:**
  `None`

#### publish(topic, conversation, finish_diskurs=False)

Dispatch a conversation to all participants subscribed to the topic.

This method sends the given conversation to all participants who are subscribed
to the specified topic. Each participant will receive the conversation and can
process it accordingly.

* **Parameters:**
  * **topic** (`str`) – The topic to which the conversation will be published.
  * **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The conversation to be dispatched.
  * **finish_diskurs** (*bool*)
* **Return type:**
  `None`

#### finalize(response)

This method is responsible for ending the conversation by setting the future object.

It is called when the conversation is finalized, and sets the dictionary response as the result
which is eventually returned by the future object.

* **Parameters:**
  **response** (`dict`) – A dictionary containing the final response data for the conversation.
* **Return type:**
  `None`

#### run(participant, conversation)

Entry point for starting a conversation with a participant.

This method starts a conversation by dispatching it to the participant, passed into it.
It also handles the finalization of the conversation, as soon as the future object is set.
returns a dictionary containing the final response data.

* **Parameters:**
  * **participant** ([`ConversationParticipant`](protocols.md#diskurs.protocols.ConversationParticipant)) – The ConversationParticipant that is involved in the conversation.
  * **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The Conversation object representing the current state of the conversation.
* **Return type:**
  `dict`
* **Returns:**
  A dictionary containing the final response data.
