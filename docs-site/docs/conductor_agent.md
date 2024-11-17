# Module: Conductor Agent

### *class* diskurs.conductor_agent.ConductorAgent(name, prompt, llm_client, topics, agent_descriptions, finalizer_name, dispatcher=None, max_trials=5, max_dispatches=50)

Bases: `BaseAgent`[[`ConductorPrompt`](protocols.md#diskurs.protocols.ConductorPrompt)], [`ConductorAgent`](protocols.md#diskurs.protocols.ConductorAgent)

* **Parameters:**
  * **name** (*str*)
  * **prompt** ([*ConductorPrompt*](protocols.md#diskurs.protocols.ConductorPrompt))
  * **llm_client** ([*LLMClient*](protocols.md#diskurs.protocols.LLMClient))
  * **topics** (*list* *[**str* *]*)
  * **agent_descriptions** (*dict* *[**str* *,* *str* *]*)
  * **finalizer_name** (*str*)
  * **dispatcher** ([*ConversationDispatcher*](protocols.md#diskurs.protocols.ConversationDispatcher) *|* *None*)
  * **max_trials** (*int*)
  * **max_dispatches** (*int*)

#### *classmethod* create(name, \*\*kwargs)

* **Return type:**
  `Self`
* **Parameters:**
  **name** (*str*)

#### *static* update_longterm_memory(source, target, overwrite)

* **Return type:**
  `LongtermMemory`
* **Parameters:**
  * **source** (*LongtermMemory* *|* *PromptArgument*)
  * **target** (*LongtermMemory*)
  * **overwrite** (*bool*)

#### *static* is_previous_agent_conductor(conversation)

#### create_or_update_longterm_memory(conversation, overwrite=False)

Creates or updates the long-term memory for the conductor agent.

This method is responsible for either creating a new long-term memory instance
or updating an existing one based on the provided conversation. It ensures that
the long-term memory is synchronized with the current state of the conversation.

* **Parameters:**
  * **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The current state of the conversation, represented as a Conversation object.
  * **overwrite** (`bool`) – A boolean flag indicating whether to overwrite existing memory fields. Defaults to False.
* **Return type:**
  [`Conversation`](protocols.md#diskurs.protocols.Conversation)
* **Returns:**
  An updated Conversation object with the new or updated long-term memory.

#### invoke(conversation)

Run the agent on a conversation.

This method processes the given conversation by invoking the agent’s logic.
It takes a Conversation object representing the conversation
and returns an updated Conversation object after processing.

* **Parameters:**
  **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The current state of the conversation, represented as a Conversation object
* **Return type:**
  [`Conversation`](protocols.md#diskurs.protocols.Conversation)
* **Returns:**
  An updated Conversation object with the processed state.

#### finalize(conversation)

* **Return type:**
  `dict`[`str`, `Any`]
* **Parameters:**
  **conversation** ([*Conversation*](protocols.md#diskurs.protocols.Conversation))

#### fail(conversation)

* **Return type:**
  `dict`[`str`, `Any`]
* **Parameters:**
  **conversation** ([*Conversation*](protocols.md#diskurs.protocols.Conversation))

#### process_conversation(conversation)

Receives a conversation from the dispatcher, i.e. message bus, processes it and finally publishes
a deep copy of the resulting conversation back to the dispatcher.

* **Parameters:**
  **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The conversation object to process.
* **Return type:**
  `None`

#### generate_validated_response(conversation, message_type=MessageType.CONVERSATION)

Generates a validated response for the given conversation.

This method attempts to generate a valid response for the conversation by
interacting with the LLM client and validating the response. It performs
multiple trials if necessary, and handles tool calls and corrective messages.

* **Parameters:**
  * **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The conversation object to generate a response for.
  * **message_type** (`MessageType`) – The type of message to render the user prompt as, defaults to MessageType.CONVERSATION.
* **Return type:**
  [`Conversation`](protocols.md#diskurs.protocols.Conversation)
* **Returns:**
  The updated conversation object with the validated response.

#### prepare_conversation(conversation, system_prompt_argument, user_prompt_argument, message_type=MessageType.CONVERSATION)

Ensures the conversation is in a valid state by creating a new set of prompts
and prompt_variables for system and user, as well creating a fresh copy of the conversation.

* **Parameters:**
  * **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – A conversation object, possible passed from another agent
    or a string to start a new conversation.
  * **system_prompt_argument** (`PromptArgument`) – The system prompt argument to use for the system prompt.
  * **user_prompt_argument** (`PromptArgument`) – The user prompt argument to use for the user prompt.
  * **message_type** (`MessageType`) – The type of message to render the user prompt as.
* **Return type:**
  [`Conversation`](protocols.md#diskurs.protocols.Conversation)
* **Returns:**
  A deep copy of the conversation, in a valid state for this agent

#### register_dispatcher(dispatcher)

Registers a dispatcher with the conversation participant.

This method is responsible for associating a ConversationDispatcher with the
conversation participant. The dispatcher will handle the distribution and management
of conversations involving this participant.

* **Parameters:**
  **dispatcher** ([`ConversationDispatcher`](protocols.md#diskurs.protocols.ConversationDispatcher)) – The ConversationDispatcher instance to be registered.
* **Return type:**
  `None`

#### return_fail_validation_message(response)

#### *property* topics *: list[str]*

#### name *: `str`*

#### prompt *: [`ConductorPrompt`](protocols.md#diskurs.protocols.ConductorPrompt)*
