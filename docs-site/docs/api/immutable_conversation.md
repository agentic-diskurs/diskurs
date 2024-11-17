# Module: Immutable Conversation

### *class* diskurs.immutable_conversation.ImmutableConversation(system_prompt=None, user_prompt=None, system_prompt_argument=None, user_prompt_argument=None, chat=None, longterm_memory=None, metadata=None, active_agent='', conversation_id='')

Bases: [`Conversation`](protocols.md#diskurs.protocols.Conversation)

* **Parameters:**
  * **system_prompt** (*ChatMessage* *|* *None*)
  * **user_prompt** (*ChatMessage* *|* *None*)
  * **system_prompt_argument** (*GenericSystemPromptArg* *|* *None*)
  * **user_prompt_argument** (*GenericUserPromptArg* *|* *None*)
  * **longterm_memory** (*dict* *[**str* *,* *LongtermMemory* *]*  *|* *None*)
  * **metadata** (*dict* *[**str* *,* *str* *]*  *|* *None*)
  * **active_agent** (*str*)

#### *property* conversation_id *: str*

Retrieves the unique identifier for the conversation.

This method returns the conversation ID, which is a unique string
used to identify the conversation instance. The conversation ID
is typically used for tracking and managing conversations within
the system.

* **Returns:**
  The unique identifier for the conversation.

#### *property* active_agent

Retrieves the name of the active agent.

The active agent is the agent currently responsible for processing the conversation.
This method returns the name of the active agent as a string.

* **Returns:**
  The name of the active agent.
* **Return type:**
  str

#### *property* chat *: list[ChatMessage]*

Provides a deep copy of the chat messages to ensure immutability.

This method returns a list of ChatMessage objects representing the conversation’s chat history.
The returned list is a deep copy, ensuring that the original chat messages remain unchanged.

* **Returns:**
  A deep copy of the list of chat messages.
* **Return type:**
  List[ChatMessage]

#### *property* system_prompt *: ChatMessage*

Retrieves the system prompt of the conversation.

The system prompt is a message that sets the initial context or instructions for the conversation.
It is typically used to guide the conversation’s flow and provide necessary background information.
Each time an agent takes a turn in the conversation, it updates the system prompt accordingly.

* **Returns:**
  The system prompt message if available, otherwise None.

#### *property* user_prompt *: ChatMessage*

Retrieves the user prompt of the conversation.

The user prompt is a message that represents the user’s input or query in the conversation.
It is typically used to direct the agent’s next response.
Each time an agent takes a turn in the conversation, it updates the user prompt accordingly.

* **Returns:**
  The user prompt message if available, otherwise None.

#### *property* system_prompt_argument *: GenericSystemPromptArg*

Retrieves the system prompt arguments.

The system prompt argument is used to render the system prompt message.
Each time an agent takes a turn in the conversation, it updates the system prompt argument.

* **Returns:**
  The system prompt arguments if available, otherwise None.

#### *property* user_prompt_argument *: GenericUserPromptArg*

Retrieves the user prompt arguments.

The user prompt argument is used to render the user prompt message.
Each time an agent takes a turn in the conversation, it updates the user prompt argument.

* **Returns:**
  The user prompt arguments if available, otherwise None.

#### *property* metadata *: dict[str, str]*

Provides a deep copy of the metadata dictionary to ensure immutability.

The metadata dictionary contains additional information about the conversation,
and can be freely defined and updated by the implementing system.

* **Returns:**
  A deep copy of the metadata dictionary.

#### *property* last_message *: ChatMessage*

Retrieves the last message in the conversation.

This property returns the most recent ChatMessage object from the conversation’s chat history.
It is used to access the latest message exchanged in the conversation, which can be useful for
determining the current state or context of the conversation.

* **Returns:**
  The last message in the chat.
* **Return type:**
  ChatMessage

#### get_agent_longterm_memory(agent_name)

Provides a deep copy of the long-term memory for the specified agent.

This method retrieves the long-term memory associated with a given agent name.
Allowing the caller to safely modify the returned memory without affecting the original.

* **Parameters:**
  **agent_name** (`str`) – The name of the agent whose long-term memory is to be retrieved.
* **Return type:**
  `LongtermMemory`
* **Returns:**
  A deep copy of the agent’s long-term memory, or None if no memory is found.

#### update_agent_longterm_memory(agent_name, longterm_memory)

Updates the long-term memory for a specific agent.

This method updates the long-term memory associated with the given agent name.
It returns a new instance of the Conversation with the updated long-term memory,
ensuring immutability of the conversation state.

* **Parameters:**
  * **agent_name** (`str`) – The name of the agent whose long-term memory is to be updated.
  * **longterm_memory** (`LongtermMemory`) – The new long-term memory for the agent.
* **Return type:**
  [`ImmutableConversation`](#diskurs.immutable_conversation.ImmutableConversation)
* **Returns:**
  A new instance of the Conversation with updated long-term memory.

#### update_prompt_argument_with_longterm_memory(conductor_name)

Updates the prompt arguments with the long-term memory of the conductor agent.

This method retrieves the long-term memory associated with the specified conductor agent
and updates the prompt arguments of the conversation accordingly. It ensures that the
conversation’s state is updated with the relevant long-term memory data, preserving
immutability by returning a new instance of the Conversation.

* **Parameters:**
  **conductor_name** (`str`) – The name of the conductor agent whose long-term memory is to be used.
* **Return type:**
  [`ImmutableConversation`](#diskurs.immutable_conversation.ImmutableConversation)
* **Returns:**
  A new instance of the Conversation with updated prompt arguments.

#### update(chat=None, system_prompt_argument=None, user_prompt_argument=None, system_prompt=None, user_prompt=None, longterm_memory=None, metadata=None, active_agent=None)

Returns a new instance of Conversation with updated fields, preserving immutability.

* **Parameters:**
  * **chat** (`Optional`[`list`[`ChatMessage`]]) – Optional list of ChatMessage objects representing the conversation’s chat history.
  * **system_prompt_argument** (`Optional`[`TypeVar`(`GenericSystemPromptArg`, bound= PromptArgument)]) – Optional system prompt argument to update.
  * **user_prompt_argument** (`Optional`[`TypeVar`(`GenericUserPromptArg`, bound= PromptArgument)]) – Optional user prompt argument to update.
  * **system_prompt** (`Optional`[`ChatMessage`]) – Optional system prompt message to update.
  * **user_prompt** (`Union`[`ChatMessage`, `list`[`ChatMessage`], `None`]) – Optional user prompt message(s) to update.
  * **longterm_memory** (`Optional`[`dict`[`str`, `Any`]]) – Optional dictionary of long-term memory to update.
  * **metadata** (`Optional`[`dict`[`str`, `str`]]) – Optional dictionary of metadata to update.
  * **active_agent** (`Optional`[`str`]) – Optional name of the active agent to update.
* **Return type:**
  [`ImmutableConversation`](#diskurs.immutable_conversation.ImmutableConversation)
* **Returns:**
  A new instance of the Conversation class with updated fields.

#### append(message, role='', name='')

Appends a new chat message and returns a new instance of Conversation.

* **Parameters:**
  * **message** (`ChatMessage` | `list`[`ChatMessage`]) – The ChatMessage object to be added to the conversation, alternatively a string can be provided.
  * **role** (`Optional`[`Role`]) – Only needed if message is str, the role (system, user, assistant)
  * **name** (`Optional`[`str`]) – Only needed if message is str, name of the agent
* **Return type:**
  [`ImmutableConversation`](#diskurs.immutable_conversation.ImmutableConversation)
* **Returns:**
  A new instance of Conversation with the appended message.

#### render_chat(message_type=MessageType.CONVERSATION)

Returns the complete chat with the system prompt prepended and the user prompt appended.

This method generates a list of ChatMessage objects representing the full chat history.
It includes the system prompt at the beginning and the user prompt at the end of the chat.

* **Parameters:**
  **message_type** (`MessageType`) – The type of message to be rendered. Defaults to MessageType.CONVERSATION.
* **Return type:**
  `list`[`ChatMessage`]
* **Returns:**
  A list representing the full chat.

#### is_empty()

Checks if the chat is empty.

* **Return type:**
  `bool`
* **Returns:**
  True if the chat is empty, False otherwise.

#### has_pending_tool_call()

Checks if there is a pending tool call in the conversation.

* **Returns:**
  True if there is a pending tool call, False otherwise.

#### has_pending_tool_response()

Checks if there is a pending tool response in the conversation,
this would be the case if the last message was a tool call from
the LLM.

* **Return type:**
  `bool`
* **Returns:**
  True if there is a pending tool response, False otherwise.

#### *classmethod* from_dict(data, agents)

Creates a Conversation instance from a dictionary.

* **Parameters:**
  * **data** (`dict`[`str`, `Any`]) – The data dictionary.
  * **agents** (`list`) – A list of agent instances.
* **Returns:**
  A new instance of Conversation.

#### to_dict()

Converts the Conversation instance to a dictionary.

* **Return type:**
  `dict`[`str`, `Any`]
* **Returns:**
  A dictionary representation of the Conversation.
