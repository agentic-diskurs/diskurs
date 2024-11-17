# Module: Multistep Agent

### *class* diskurs.multistep_agent.MultiStepAgent(name, prompt, llm_client, topics=None, dispatcher=None, tool_executor=None, tools=None, max_reasoning_steps=5, max_trials=5, init_prompt_arguments_with_longterm_memory=True)

Bases: `BaseAgent`[[`MultistepPrompt`](protocols.md#diskurs.protocols.MultistepPrompt)]

* **Parameters:**
  * **name** (*str*)
  * **prompt** ([*MultistepPrompt*](protocols.md#diskurs.protocols.MultistepPrompt))
  * **llm_client** ([*LLMClient*](protocols.md#diskurs.protocols.LLMClient))
  * **topics** (*list* *[**str* *]*)
  * **dispatcher** ([*ConversationDispatcher*](protocols.md#diskurs.protocols.ConversationDispatcher) *|* *None*)
  * **tool_executor** ([*ToolExecutor*](protocols.md#diskurs.protocols.ToolExecutor) *|* *None*)
  * **tools** (*list* *[**ToolDescription* *]*  *|* *None*)
  * **max_reasoning_steps** (*int*)
  * **max_trials** (*int*)
  * **init_prompt_arguments_with_longterm_memory** (*bool*)

#### *classmethod* create(name, prompt, llm_client, \*\*kwargs)

* **Return type:**
  `Self`
* **Parameters:**
  * **name** (*str*)
  * **prompt** ([*MultistepPrompt*](protocols.md#diskurs.protocols.MultistepPrompt))
  * **llm_client** ([*LLMClient*](protocols.md#diskurs.protocols.LLMClient))

#### get_conductor_name()

* **Return type:**
  `str`

#### register_tools(tools)

Registers one or more tools with the executor.

This method allows the registration of a single tool or a list of tools
that can be executed by the executor. Each tool is a callable that can
be invoked with specific arguments.

* **Parameters:**
  **tools** (`Union`[`list`[`Callable`], `Callable`]) – A single callable or a list of callables representing the tools to be registered.
* **Return type:**
  `None`

#### compute_tool_response(response)

Executes the tool calls in the response and returns the tool responses.

* **Parameters:**
  **response** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The conversation object containing the tool calls to execute.
* **Return type:**
  `list`[`ChatMessage`]
* **Returns:**
  One or more ChatMessage objects containing the tool responses.

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

#### process_conversation(conversation)

Receives a conversation from the dispatcher, i.e. message bus, processes it and finally publishes
a deep copy of the resulting conversation back to the dispatcher.

* **Parameters:**
  **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The conversation object to process.
* **Return type:**
  `None`

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
