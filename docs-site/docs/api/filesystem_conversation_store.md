# Module: Filesystem Conversation Store

### *class* diskurs.filesystem_conversation_store.FilesystemConversationStore(directory, agents, conversation_class)

Bases: [`ConversationStore`](protocols.md#diskurs.protocols.ConversationStore)

* **Parameters:**
  * **directory** (*Path*)
  * **agents** (*list*)
  * **conversation_class** ([*Conversation*](protocols.md#diskurs.protocols.Conversation))

#### *classmethod* create(\*\*kwargs)

* **Return type:**
  `Self`

#### persist(conversation)

Persists the given conversation.

This method is responsible for saving the state of the provided conversation
to a persistent storage. Implementations of this method should ensure that
the conversation data is reliably stored and can be retrieved later.

* **Parameters:**
  **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The Conversation object representing the current state of the conversation.
* **Return type:**
  `None`

#### fetch(conversation_id)

Fetches a conversation by its unique identifier.

This method retrieves the conversation associated with the given conversation ID from the persistent storage.
It ensures that the conversation data is accurately fetched and returned as a Conversation object.

* **Parameters:**
  **conversation_id** (`str`) – The unique identifier of the conversation to be fetched.
* **Return type:**
  [`Conversation`](protocols.md#diskurs.protocols.Conversation)
* **Returns:**
  The Conversation object representing the fetched conversation.

#### delete(conversation_id)

Deletes a conversation by its unique identifier.

This method removes the conversation associated with the given conversation ID from the persistent storage.
It ensures that the conversation data is permanently deleted and can no longer be retrieved.

* **Parameters:**
  **conversation_id** (`str`) – The unique identifier of the conversation to be deleted.
* **Return type:**
  `None`
* **Returns:**
  None

#### exists(conversation_id)

Checks if a conversation with the given unique identifier exists in the persistent storage.

This method is responsible for verifying the existence of a conversation by its unique ID.
It returns a boolean value indicating whether the conversation is present in the storage.

* **Parameters:**
  **conversation_id** (`str`) – The unique identifier of the conversation to check.
* **Return type:**
  `bool`
* **Returns:**
  True if the conversation exists, False otherwise.
