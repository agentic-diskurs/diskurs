# Module: Heuristic Agent

### *class* diskurs.heuristic_agent.HeuristicAgent(name, prompt, topics=None, dispatcher=None, tool_executor=None, init_prompt_arguments_with_longterm_memory=True, render_prompt=True)

Bases: [`Agent`](protocols.md#diskurs.protocols.Agent), [`ConversationParticipant`](protocols.md#diskurs.protocols.ConversationParticipant)

* **Parameters:**
  * **name** (*str*)
  * **prompt** ([*HeuristicPrompt*](protocols.md#diskurs.protocols.HeuristicPrompt))
  * **topics** (*list* *[**str* *]*)
  * **dispatcher** ([*ConversationDispatcher*](protocols.md#diskurs.protocols.ConversationDispatcher) *|* *None*)
  * **tool_executor** ([*ToolExecutor*](protocols.md#diskurs.protocols.ToolExecutor) *|* *None*)
  * **init_prompt_arguments_with_longterm_memory** (*bool*)
  * **render_prompt** (*bool*)

#### name *: `str`*

#### topics *: `list`[`str`]*

#### *classmethod* create(name, prompt, \*\*kwargs)

* **Return type:**
  `Self`
* **Parameters:**
  * **name** (*str*)
  * **prompt** ([*HeuristicPrompt*](protocols.md#diskurs.protocols.HeuristicPrompt))

#### get_conductor_name()

* **Return type:**
  `str`

#### register_dispatcher(dispatcher)

Registers a dispatcher with the conversation participant.

This method is responsible for associating a ConversationDispatcher with the
conversation participant. The dispatcher will handle the distribution and management
of conversations involving this participant.

* **Parameters:**
  **dispatcher** ([`ConversationDispatcher`](protocols.md#diskurs.protocols.ConversationDispatcher)) – The ConversationDispatcher instance to be registered.
* **Return type:**
  `None`

#### prepare_conversation(conversation, user_prompt_argument)

* **Return type:**
  [`Conversation`](protocols.md#diskurs.protocols.Conversation)
* **Parameters:**
  * **conversation** ([*Conversation*](protocols.md#diskurs.protocols.Conversation))
  * **user_prompt_argument** (*PromptArgument*)

#### invoke(conversation)

Run the agent on a conversation.

This method processes the given conversation by invoking the agent’s logic.
It takes a Conversation object representing the conversation
and returns an updated Conversation object after processing.

* **Parameters:**
  **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation) | `str`) – The current state of the conversation, represented as a Conversation object
* **Return type:**
  [`Conversation`](protocols.md#diskurs.protocols.Conversation)
* **Returns:**
  An updated Conversation object with the processed state.

#### process_conversation(conversation)

Actively participate in a conversation.

This method is responsible for processing the given conversation. The implementation should  handle
the conversation appropriately, updating its state and generating responses as needed.
This method is called by the conversation dispatcher.

* **Parameters:**
  **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The conversation to be processed.
* **Return type:**
  `None`
