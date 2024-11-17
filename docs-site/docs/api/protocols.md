# Protocols

### *class* diskurs.protocols.LongtermMemoryHandler(\*args, \*\*kwargs)

Bases: `Protocol`

#### can_finalize(longterm_memory)

* **Return type:**
  `bool`
* **Parameters:**
  **longterm_memory** (*Any*)

### *class* diskurs.protocols.PromptValidator(\*args, \*\*kwargs)

Bases: `Protocol`

Protocol for validating prompt responses.

This protocol defines methods for validating responses from a language model (LLM).
It includes methods for validating responses as dataclasses and JSON objects.

#### *classmethod* validate_dataclass(parsed_response, user_prompt_argument, strict=False)

Validates a parsed response dictionary against a dataclass type.

* **Parameters:**
  * **parsed_response** (`dict`[`str`, `Any`]) – The dictionary containing the parsed response from the LLM.
  * **user_prompt_argument** (`Type`[`dataclass`]) – The dataclass type to validate against.
  * **strict** (`bool`) – If True, enforce strict validation rules.
* **Return type:**
  `dataclass`
* **Returns:**
  An instance of the dataclass populated with the validated data.

#### *classmethod* validate_json(llm_response)

Validates a JSON response from a language model (LLM).

This method takes a JSON string response from an LLM and validates it,
ensuring it conforms to the expected structure and content.

* **Parameters:**
  **llm_response** (`str`) – The JSON string response from the LLM.
* **Return type:**
  `dict`
* **Returns:**
  A dictionary representation of the validated JSON response.

### *class* diskurs.protocols.Prompt(\*args, \*\*kwargs)

Bases: `Protocol`

Protocol for prompt implementations.

This protocol defines the structure and methods required for creating and handling prompts
in a conversation. Implementations of this protocol are responsible for generating prompt
arguments, rendering templates, and parsing responses from a language model (LLM).

#### system_prompt_argument *: `Type`[`TypeVar`(`SystemPromptArg`, bound= `PromptArgument`)]*

#### user_prompt_argument *: `Type`[`TypeVar`(`UserPromptArg`, bound= `PromptArgument`)]*

#### create_system_prompt_argument(\*\*prompt_args)

Creates an instance of the system prompt argument dataclass.

This method is responsible for generating the system prompt argument
based on the provided keyword arguments. The system prompt argument
is used to configure the initial state and context for the conversation.

* **Parameters:**
  **prompt_args** (`Any`) – Keyword arguments used to initialize the system prompt argument.
* **Return type:**
  `TypeVar`(`SystemPromptArg`, bound= `PromptArgument`)
* **Returns:**
  An instance of the system prompt argument dataclass.

#### create_user_prompt_argument(\*\*prompt_args)

Creates an instance of the user prompt argument dataclass.

This method is responsible for generating the user prompt argument
based on the provided keyword arguments. The user prompt argument
is used to configure the user’s input and context for the conversation.

* **Parameters:**
  **prompt_args** (`Any`) – Keyword arguments used to initialize the user prompt argument.
* **Return type:**
  `TypeVar`(`UserPromptArg`, bound= `PromptArgument`)
* **Returns:**
  An instance of the user prompt argument dataclass.

#### render_system_template(name, prompt_args, return_json=True)

Renders the system template with the provided prompt arguments.

This method is responsible for rendering the system template using the given prompt arguments.
It can optionally return the rendered template as a JSON object.

* **Parameters:**
  * **name** (`str`) – The name of the template to be rendered.
  * **prompt_args** (`PromptArgument`) – The prompt arguments to be used for rendering the template.
  * **return_json** (`bool`) – If True, the rendered template will be returned as a JSON object. Defaults to True.
* **Return type:**
  `ChatMessage`
* **Returns:**
  A ChatMessage object containing the rendered template.

#### render_user_template(name, prompt_args, message_type=MessageType.CONVERSATION)

Renders the user template with the provided prompt arguments.

This method is responsible for rendering the user template using the given prompt arguments.
It generates a ChatMessage object that represents the rendered template.

* **Parameters:**
  * **name** (`str`) – The name of the template to be rendered.
  * **prompt_args** (`PromptArgument`) – The prompt arguments to be used for rendering the template.
  * **message_type** (`MessageType`) – The type of the message to be rendered. Defaults to MessageType.CONVERSATION.
* **Return type:**
  `ChatMessage`
* **Returns:**
  A ChatMessage object containing the rendered template.

#### parse_user_prompt(llm_response, old_user_prompt_argument, message_type=MessageType.ROUTING)

Parses the LLM response into a prompt argument or ChatMessage.

This method takes the response from a language model (LLM) and parses it into either a
PromptArgument or a ChatMessage object. It uses the old user prompt argument and
the message type to guide the parsing process.

* **Parameters:**
  * **llm_response** (`str`) – The response string from the language model.
  * **old_user_prompt_argument** (`PromptArgument`) – The previous user prompt argument to be used as a reference.
  * **message_type** (`MessageType`) – The type of message to be parsed. Defaults to MessageType.ROUTING.
* **Return type:**
  `Union`[`PromptArgument`, `ChatMessage`]
* **Returns:**
  A PromptArgument or ChatMessage object based on the parsed response.

### *class* diskurs.protocols.MultistepPrompt(\*args, \*\*kwargs)

Bases: [`Prompt`](#diskurs.protocols.Prompt)

#### is_final(user_prompt_argument)

Determines if the user prompt argument indicates the final state.

This method checks the provided user prompt argument to determine if it represents
the final state in the conversation. It is used to decide whether the conversation
can be concluded based on the user’s input.

* **Parameters:**
  **user_prompt_argument** (`PromptArgument`) – The user prompt argument to be evaluated.
* **Return type:**
  `bool`
* **Returns:**
  True if the user prompt argument indicates the final state, False otherwise.

#### is_valid(user_prompt_argument)

Validates the user prompt argument.

This method checks if the provided user prompt argument meets the required
criteria for validity. It ensures that the user prompt argument is correctly
structured and contains the necessary information for further processing.

* **Parameters:**
  **user_prompt_argument** (`PromptArgument`) – The user prompt argument to be validated.
* **Return type:**
  `bool`
* **Returns:**
  True if the user prompt argument is valid, False otherwise.

#### create_system_prompt_argument(\*\*prompt_args)

Creates an instance of the system prompt argument dataclass.

This method is responsible for generating the system prompt argument
based on the provided keyword arguments. The system prompt argument
is used to configure the initial state and context for the conversation.

* **Parameters:**
  **prompt_args** (`Any`) – Keyword arguments used to initialize the system prompt argument.
* **Return type:**
  `TypeVar`(`SystemPromptArg`, bound= `PromptArgument`)
* **Returns:**
  An instance of the system prompt argument dataclass.

#### create_user_prompt_argument(\*\*prompt_args)

Creates an instance of the user prompt argument dataclass.

This method is responsible for generating the user prompt argument
based on the provided keyword arguments. The user prompt argument
is used to configure the user’s input and context for the conversation.

* **Parameters:**
  **prompt_args** (`Any`) – Keyword arguments used to initialize the user prompt argument.
* **Return type:**
  `TypeVar`(`UserPromptArg`, bound= `PromptArgument`)
* **Returns:**
  An instance of the user prompt argument dataclass.

#### parse_user_prompt(llm_response, old_user_prompt_argument, message_type=MessageType.ROUTING)

Parses the LLM response into a prompt argument or ChatMessage.

This method takes the response from a language model (LLM) and parses it into either a
PromptArgument or a ChatMessage object. It uses the old user prompt argument and
the message type to guide the parsing process.

* **Parameters:**
  * **llm_response** (`str`) – The response string from the language model.
  * **old_user_prompt_argument** (`PromptArgument`) – The previous user prompt argument to be used as a reference.
  * **message_type** (`MessageType`) – The type of message to be parsed. Defaults to MessageType.ROUTING.
* **Return type:**
  `Union`[`PromptArgument`, `ChatMessage`]
* **Returns:**
  A PromptArgument or ChatMessage object based on the parsed response.

#### render_system_template(name, prompt_args, return_json=True)

Renders the system template with the provided prompt arguments.

This method is responsible for rendering the system template using the given prompt arguments.
It can optionally return the rendered template as a JSON object.

* **Parameters:**
  * **name** (`str`) – The name of the template to be rendered.
  * **prompt_args** (`PromptArgument`) – The prompt arguments to be used for rendering the template.
  * **return_json** (`bool`) – If True, the rendered template will be returned as a JSON object. Defaults to True.
* **Return type:**
  `ChatMessage`
* **Returns:**
  A ChatMessage object containing the rendered template.

#### render_user_template(name, prompt_args, message_type=MessageType.CONVERSATION)

Renders the user template with the provided prompt arguments.

This method is responsible for rendering the user template using the given prompt arguments.
It generates a ChatMessage object that represents the rendered template.

* **Parameters:**
  * **name** (`str`) – The name of the template to be rendered.
  * **prompt_args** (`PromptArgument`) – The prompt arguments to be used for rendering the template.
  * **message_type** (`MessageType`) – The type of the message to be rendered. Defaults to MessageType.CONVERSATION.
* **Return type:**
  `ChatMessage`
* **Returns:**
  A ChatMessage object containing the rendered template.

#### system_prompt_argument *: `Type`[`TypeVar`(`SystemPromptArg`, bound= `PromptArgument`)]*

#### user_prompt_argument *: `Type`[`TypeVar`(`UserPromptArg`, bound= `PromptArgument`)]*

### *class* diskurs.protocols.ConductorPrompt(\*args, \*\*kwargs)

Bases: [`Prompt`](#diskurs.protocols.Prompt)

Protocol for conductor prompts.

This protocol defines the methods required for handling conductor prompts in a conversation.
It includes methods for initializing, finalizing, and validating long-term memory, as well as
determining the final state of the conversation.

#### longterm_memory *: `Type`[`LongtermMemory`]*

#### can_finalize(longterm_memory)

Determines if the conductor can finalize based on the long-term memory.

This method evaluates the provided long-term memory to decide whether the
conversation can be concluded. It checks if the necessary conditions are met
for finalizing the conversation.

* **Parameters:**
  **longterm_memory** (`LongtermMemory`) – The long-term memory to be evaluated.
* **Return type:**
  `bool`
* **Returns:**
  True if the conversation can be finalized, False otherwise.

#### finalize(longterm_memory)

Finalizes the conversation based on the long-term memory.

This method processes the provided long-term memory to generate a final response
for the conversation. It is responsible for concluding the conversation by
utilizing the accumulated long-term memory data.

* **Parameters:**
  **longterm_memory** (`LongtermMemory`) – The long-term memory to be used for finalizing the conversation.
* **Return type:**
  `dict`[`str`, `Any`]
* **Returns:**
  A dictionary containing the final response data.

#### fail(longterm_memory)

Handles the case failure i.e. when the long-term memory does not meet the criteria defined
for finalization.

This method processes the provided long-term memory to generate a failure response
for the conversation. It is responsible for concluding the conversation in a failure
state by utilizing the accumulated long-term memory data.

* **Parameters:**
  **longterm_memory** (`LongtermMemory`) – The long-term memory to be used for generating the failure response.
* **Return type:**
  `dict`[`str`, `Any`]
* **Returns:**
  A dictionary containing the failure response data.

#### init_longterm_memory(\*\*kwargs)

Initializes the long-term memory.

This method is responsible for creating and initializing the long-term memory
for the conductor agent. It uses the provided keyword arguments to set up the
initial state of the long-term memory.

* **Parameters:**
  **kwargs** (`Any`) – Keyword arguments used to initialize the long-term memory.
* **Return type:**
  `LongtermMemory`
* **Returns:**
  An instance of the LongtermMemory class.

#### is_final(user_prompt_argument)

Determines if the user prompt argument indicates the final state.

This method checks the provided user prompt argument to determine if it represents
the final state in the conversation. It is used to decide whether the conversation
can be concluded based on the user’s input.

* **Parameters:**
  **user_prompt_argument** (`PromptArgument`) – The user prompt argument to be evaluated.
* **Return type:**
  `bool`
* **Returns:**
  True if the user prompt argument indicates the final state, False otherwise.

#### is_valid(user_prompt_argument)

Validates the user prompt argument.

This method checks if the provided user prompt argument meets the required
criteria for validity. It ensures that the user prompt argument is correctly
structured and contains the necessary information for further processing.

* **Parameters:**
  **user_prompt_argument** (`PromptArgument`) – The user prompt argument to be validated.
* **Return type:**
  `bool`
* **Returns:**
  True if the user prompt argument is valid, False otherwise.

#### create_system_prompt_argument(\*\*prompt_args)

Creates an instance of the system prompt argument dataclass.

This method is responsible for generating the system prompt argument
based on the provided keyword arguments. The system prompt argument
is used to configure the initial state and context for the conversation.

* **Parameters:**
  **prompt_args** (`Any`) – Keyword arguments used to initialize the system prompt argument.
* **Return type:**
  `TypeVar`(`SystemPromptArg`, bound= `PromptArgument`)
* **Returns:**
  An instance of the system prompt argument dataclass.

#### create_user_prompt_argument(\*\*prompt_args)

Creates an instance of the user prompt argument dataclass.

This method is responsible for generating the user prompt argument
based on the provided keyword arguments. The user prompt argument
is used to configure the user’s input and context for the conversation.

* **Parameters:**
  **prompt_args** (`Any`) – Keyword arguments used to initialize the user prompt argument.
* **Return type:**
  `TypeVar`(`UserPromptArg`, bound= `PromptArgument`)
* **Returns:**
  An instance of the user prompt argument dataclass.

#### parse_user_prompt(llm_response, old_user_prompt_argument, message_type=MessageType.ROUTING)

Parses the LLM response into a prompt argument or ChatMessage.

This method takes the response from a language model (LLM) and parses it into either a
PromptArgument or a ChatMessage object. It uses the old user prompt argument and
the message type to guide the parsing process.

* **Parameters:**
  * **llm_response** (`str`) – The response string from the language model.
  * **old_user_prompt_argument** (`PromptArgument`) – The previous user prompt argument to be used as a reference.
  * **message_type** (`MessageType`) – The type of message to be parsed. Defaults to MessageType.ROUTING.
* **Return type:**
  `Union`[`PromptArgument`, `ChatMessage`]
* **Returns:**
  A PromptArgument or ChatMessage object based on the parsed response.

#### render_system_template(name, prompt_args, return_json=True)

Renders the system template with the provided prompt arguments.

This method is responsible for rendering the system template using the given prompt arguments.
It can optionally return the rendered template as a JSON object.

* **Parameters:**
  * **name** (`str`) – The name of the template to be rendered.
  * **prompt_args** (`PromptArgument`) – The prompt arguments to be used for rendering the template.
  * **return_json** (`bool`) – If True, the rendered template will be returned as a JSON object. Defaults to True.
* **Return type:**
  `ChatMessage`
* **Returns:**
  A ChatMessage object containing the rendered template.

#### render_user_template(name, prompt_args, message_type=MessageType.CONVERSATION)

Renders the user template with the provided prompt arguments.

This method is responsible for rendering the user template using the given prompt arguments.
It generates a ChatMessage object that represents the rendered template.

* **Parameters:**
  * **name** (`str`) – The name of the template to be rendered.
  * **prompt_args** (`PromptArgument`) – The prompt arguments to be used for rendering the template.
  * **message_type** (`MessageType`) – The type of the message to be rendered. Defaults to MessageType.CONVERSATION.
* **Return type:**
  `ChatMessage`
* **Returns:**
  A ChatMessage object containing the rendered template.

#### system_prompt_argument *: `Type`[`TypeVar`(`SystemPromptArg`, bound= `PromptArgument`)]*

#### user_prompt_argument *: `Type`[`TypeVar`(`UserPromptArg`, bound= `PromptArgument`)]*

### *class* diskurs.protocols.CallTool(\*args, \*\*kwargs)

Bases: `Protocol`

### *class* diskurs.protocols.HeuristicSequence(\*args, \*\*kwargs)

Bases: `Protocol`

### *class* diskurs.protocols.HeuristicPrompt(\*args, \*\*kwargs)

Bases: `Protocol`

Protocol for heuristic prompts.

This protocol defines the structure and methods required for creating and handling heuristic prompts
in a conversation. Implementations of this protocol are responsible for generating user prompt arguments,
rendering user templates, and executing heuristic sequences to process conversations.

#### user_prompt_argument *: `Type`[`PromptArgument`]*

#### heuristic_sequence(conversation, call_tool)

Executes a heuristic sequence on the given conversation.

This method processes the conversation using a series of heuristic steps,
optionally utilizing a tool for specific operations. The heuristic sequence
is designed to guide the conversation towards a desired outcome based on
predefined rules and logic.

* **Parameters:**
  * **conversation** ([`Conversation`](#diskurs.protocols.Conversation)) – The conversation object to be processed.
  * **call_tool** ([`CallTool`](#diskurs.protocols.CallTool)) – An optional callable tool that can be used during the heuristic sequence.
* **Return type:**
  [`Conversation`](#diskurs.protocols.Conversation)
* **Returns:**
  The updated conversation object after processing the heuristic sequence.

#### create_user_prompt_argument(\*\*prompt_args)

Creates an instance of the user prompt argument dataclass.

This method is responsible for generating the user prompt argument
based on the provided keyword arguments. The user prompt argument
is used to configure the user’s input and context for the conversation.

* **Parameters:**
  **prompt_args** – Keyword arguments used to initialize the user prompt argument.
* **Return type:**
  `PromptArgument`
* **Returns:**
  An instance of the user prompt argument dataclass.

#### render_user_template(name, prompt_args, message_type=MessageType.CONVERSATION)

Renders the user template with the provided prompt arguments.

This method is responsible for rendering the user template using the given prompt arguments.
It generates a ChatMessage object that represents the rendered template.

* **Parameters:**
  * **name** (`str`) – The name of the template to be rendered.
  * **prompt_args** (`PromptArgument`) – The prompt arguments to be used for rendering the template.
  * **message_type** (`MessageType`) – The type of the message to be rendered. Defaults to MessageType.CONVERSATION.
* **Return type:**
  `ChatMessage`
* **Returns:**
  A ChatMessage object containing the rendered template.

### *class* diskurs.protocols.Conversation(\*args, \*\*kwargs)

Bases: `Protocol`[`SystemPromptArg`, `UserPromptArg`]

Protocol for conversation management.

This protocol defines the structure and methods required for handling conversations
between agents and users. It includes properties and methods for accessing and updating
chat messages, prompt arguments, long-term memory, and metadata. Implementations of this
protocol are responsible for managing the state and flow of conversations, ensuring
immutability, and providing mechanisms for rendering and parsing prompts.

#### *property* chat *: List[ChatMessage]*

Provides a deep copy of the chat messages to ensure immutability.

This method returns a list of ChatMessage objects representing the conversation’s chat history.
The returned list is a deep copy, ensuring that the original chat messages remain unchanged.

* **Returns:**
  A deep copy of the list of chat messages.
* **Return type:**
  List[ChatMessage]

#### *property* system_prompt *: ChatMessage | None*

Retrieves the system prompt of the conversation.

The system prompt is a message that sets the initial context or instructions for the conversation.
It is typically used to guide the conversation’s flow and provide necessary background information.
Each time an agent takes a turn in the conversation, it updates the system prompt accordingly.

* **Returns:**
  The system prompt message if available, otherwise None.

#### *property* user_prompt *: ChatMessage | None*

Retrieves the user prompt of the conversation.

The user prompt is a message that represents the user’s input or query in the conversation.
It is typically used to direct the agent’s next response.
Each time an agent takes a turn in the conversation, it updates the user prompt accordingly.

* **Returns:**
  The user prompt message if available, otherwise None.

#### *property* system_prompt_argument *: SystemPromptArg | None*

Retrieves the system prompt arguments.

The system prompt argument is used to render the system prompt message.
Each time an agent takes a turn in the conversation, it updates the system prompt argument.

* **Returns:**
  The system prompt arguments if available, otherwise None.

#### *property* user_prompt_argument *: UserPromptArg | None*

Retrieves the user prompt arguments.

The user prompt argument is used to render the user prompt message.
Each time an agent takes a turn in the conversation, it updates the user prompt argument.

* **Returns:**
  The user prompt arguments if available, otherwise None.

#### *property* metadata *: Dict[str, str]*

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

#### *property* active_agent *: str*

Retrieves the name of the active agent.

The active agent is the agent currently responsible for processing the conversation.
This method returns the name of the active agent as a string.

* **Returns:**
  The name of the active agent.
* **Return type:**
  str

#### *property* conversation_id *: str*

Retrieves the unique identifier for the conversation.

This method returns the conversation ID, which is a unique string
used to identify the conversation instance. The conversation ID
is typically used for tracking and managing conversations within
the system.

* **Returns:**
  The unique identifier for the conversation.

#### get_agent_longterm_memory(agent_name)

Provides a deep copy of the long-term memory for the specified agent.

This method retrieves the long-term memory associated with a given agent name.
Allowing the caller to safely modify the returned memory without affecting the original.

* **Parameters:**
  **agent_name** (`str`) – The name of the agent whose long-term memory is to be retrieved.
* **Return type:**
  `Optional`[`LongtermMemory`]
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
  [`Conversation`](#diskurs.protocols.Conversation)
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
  [`Conversation`](#diskurs.protocols.Conversation)
* **Returns:**
  A new instance of the Conversation with updated prompt arguments.

#### update(chat=None, system_prompt_argument=None, user_prompt_argument=None, system_prompt=None, user_prompt=None, longterm_memory=None, metadata=None, active_agent=None)

Returns a new instance of Conversation with updated fields, preserving immutability.

* **Parameters:**
  * **chat** (`Optional`[`List`[`ChatMessage`]]) – Optional list of ChatMessage objects representing the conversation’s chat history.
  * **system_prompt_argument** (`Optional`[`TypeVar`(`SystemPromptArg`, bound= `PromptArgument`)]) – Optional system prompt argument to update.
  * **user_prompt_argument** (`Optional`[`TypeVar`(`UserPromptArg`, bound= `PromptArgument`)]) – Optional user prompt argument to update.
  * **system_prompt** (`Optional`[`ChatMessage`]) – Optional system prompt message to update.
  * **user_prompt** (`Union`[`ChatMessage`, `List`[`ChatMessage`], `None`]) – Optional user prompt message(s) to update.
  * **longterm_memory** (`Optional`[`Dict`[`str`, `LongtermMemory`]]) – Optional dictionary of long-term memory to update.
  * **metadata** (`Optional`[`Dict`[`str`, `str`]]) – Optional dictionary of metadata to update.
  * **active_agent** (`Optional`[`str`]) – Optional name of the active agent to update.
* **Return type:**
  [`Conversation`](#diskurs.protocols.Conversation)
* **Returns:**
  A new instance of the Conversation class with updated fields.

#### append(message, role='', name='')

Appends a new chat message and returns a new instance of Conversation.

* **Parameters:**
  * **message** (`Union`[`ChatMessage`, `List`[`ChatMessage`], `str`]) – The message to be added.
  * **role** (`Optional`[`str`]) – The role of the message sender.
  * **name** (`Optional`[`str`]) – The name of the message sender.
* **Return type:**
  [`Conversation`](#diskurs.protocols.Conversation)
* **Returns:**
  A new instance of Conversation with the appended message.

#### render_chat(message_type=MessageType.CONVERSATION)

Returns the complete chat with the system prompt prepended and the user prompt appended.

This method generates a list of ChatMessage objects representing the full chat history.
It includes the system prompt at the beginning and the user prompt at the end of the chat.

* **Parameters:**
  **message_type** (`MessageType`) – The type of message to be rendered. Defaults to MessageType.CONVERSATION.
* **Return type:**
  `List`[`ChatMessage`]
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

* **Return type:**
  `bool`
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
  * **agents** (`list`[`Any`]) – A list of agent instances.
* **Return type:**
  [`Conversation`](#diskurs.protocols.Conversation)
* **Returns:**
  A new instance of Conversation.

#### to_dict()

Converts the Conversation instance to a dictionary.

* **Return type:**
  `Dict`[`str`, `Any`]
* **Returns:**
  A dictionary representation of the Conversation.

### *class* diskurs.protocols.LLMClient(\*args, \*\*kwargs)

Bases: `Protocol`

Protocol for interacting with a Language Model (LLM) client.
This protocol defines the methods required for creating and generating responses
from a language model within a conversation. Implementations of this protocol
are responsible for initializing the LLM client and generating responses based
on the current state of the conversation and optional tools.

#### *classmethod* create(\*\*kwargs)

* **Return type:**
  `Self`

#### generate(conversation, tools=None)

Generates a response from the language model (LLM) based on the current state of the conversation.

This method takes the current conversation and optionally a list of tools, and generates a response
from the language model. The response is integrated into the conversation, updating its state.

* **Parameters:**
  * **conversation** ([`Conversation`](#diskurs.protocols.Conversation)) – The current state of the conversation, represented as a Conversation object.
  * **tools** (`Optional`[`list`[`ToolDescription`]]) – An optional list of ToolDescription objects that can be used by the LLM to generate the response.
* **Return type:**
  [`Conversation`](#diskurs.protocols.Conversation)
* **Returns:**
  An updated Conversation object with the generated response.

### *class* diskurs.protocols.ConversationParticipant(\*args, \*\*kwargs)

Bases: `Protocol`

Protocol for conversation participants.

This protocol defines the methods and properties required for an entity to participate in a conversation.
Implementations of this protocol are responsible for processing conversations, registering dispatchers,
and handling topics of interest. It ensures that any participant can actively engage in and manage
conversations within the system.

#### topics *: `list`[`str`]*

#### process_conversation(conversation)

Actively participate in a conversation.

This method is responsible for processing the given conversation. The implementation should  handle
the conversation appropriately, updating its state and generating responses as needed.
This method is called by the conversation dispatcher.

* **Parameters:**
  **conversation** ([`Conversation`](#diskurs.protocols.Conversation)) – The conversation to be processed.
* **Return type:**
  `None`

#### register_dispatcher(dispatcher)

Registers a dispatcher with the conversation participant.

This method is responsible for associating a ConversationDispatcher with the
conversation participant. The dispatcher will handle the distribution and management
of conversations involving this participant.

* **Parameters:**
  **dispatcher** ([`ConversationDispatcher`](#diskurs.protocols.ConversationDispatcher)) – The ConversationDispatcher instance to be registered.
* **Return type:**
  `None`

### *class* diskurs.protocols.ConversationDispatcher(\*args, \*\*kwargs)

Bases: `Protocol`

Protocol for  dispatching conversations.

This protocol defines the methods required for subscribing participants to topics,
publishing conversations to subscribed participants, and finalizing conversations.
Implementations of this protocol are responsible for coordinating the flow of conversations
between agents and participants, ensuring that conversations are properly managed and
dispatched to the appropriate entities based on their topics of interest.

#### subscribe(topic, participant)

Subscribe a participant to a specific topic.

This method registers a ConversationParticipant to receive conversations
related to the specified topic. When a conversation is published to the topic,
all subscribed participants will be notified and can process the conversation.

* **Parameters:**
  * **topic** (`str`) – The topic to which the participant will be subscribed.
  * **participant** ([`ConversationParticipant`](#diskurs.protocols.ConversationParticipant)) – The participant to be subscribed to the topic.
* **Return type:**
  `None`

#### unsubscribe(topic, participant)

Unsubscribe a participant from a specific topic.

This method removes a ConversationParticipant from the list of subscribers for the given topic.
Once unsubscribed, the participant will no longer receive conversations related to that topic.

* **Parameters:**
  * **topic** (`str`) – The topic from which the participant will be unsubscribed.
  * **participant** ([`ConversationParticipant`](#diskurs.protocols.ConversationParticipant)) – The participant to be unsubscribed from the topic.
* **Return type:**
  `None`

#### publish(topic, conversation)

Dispatch a conversation to all participants subscribed to the topic.

This method sends the given conversation to all participants who are subscribed
to the specified topic. Each participant will receive the conversation and can
process it accordingly.

* **Parameters:**
  * **topic** (`str`) – The topic to which the conversation will be published.
  * **conversation** ([`Conversation`](#diskurs.protocols.Conversation)) – The conversation to be dispatched.
* **Return type:**
  `None`

#### run(participant, conversation)

Entry point for starting a conversation with a participant.

This method starts a conversation by dispatching it to the participant, passed into it.
It also handles the finalization of the conversation, as soon as the future object is set.
returns a dictionary containing the final response data.

* **Parameters:**
  * **participant** ([`ConversationParticipant`](#diskurs.protocols.ConversationParticipant)) – The ConversationParticipant that is involved in the conversation.
  * **conversation** ([`Conversation`](#diskurs.protocols.Conversation)) – The Conversation object representing the current state of the conversation.
* **Return type:**
  `dict`
* **Returns:**
  A dictionary containing the final response data.

#### finalize(response)

This method is responsible for ending the conversation by setting the future object.

It is called when the conversation is finalized, and sets the dictionary response as the result
which is eventually returned by the future object.

* **Parameters:**
  **response** (`dict`) – A dictionary containing the final response data for the conversation.
* **Return type:**
  `None`

### *class* diskurs.protocols.ConversationStore(\*args, \*\*kwargs)

Bases: `Protocol`

Protocol for managing the persistence of conversations.

This protocol defines the methods required for creating, persisting, fetching, deleting, and checking
the existence of conversations.
Implementations of this protocol are responsible for handling the storage and retrieval of conversation data,
ensuring that conversations can be reliably saved and accessed as needed.

#### *classmethod* create(\*\*kwargs)

* **Return type:**
  `Self`

#### persist(conversation)

Persists the given conversation.

This method is responsible for saving the state of the provided conversation
to a persistent storage. Implementations of this method should ensure that
the conversation data is reliably stored and can be retrieved later.

* **Parameters:**
  **conversation** ([`Conversation`](#diskurs.protocols.Conversation)) – The Conversation object representing the current state of the conversation.
* **Return type:**
  `None`

#### fetch(conversation_id)

Fetches a conversation by its unique identifier.

This method retrieves the conversation associated with the given conversation ID from the persistent storage.
It ensures that the conversation data is accurately fetched and returned as a Conversation object.

* **Parameters:**
  **conversation_id** (`str`) – The unique identifier of the conversation to be fetched.
* **Return type:**
  [`Conversation`](#diskurs.protocols.Conversation)
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

### *class* diskurs.protocols.Agent(\*args, \*\*kwargs)

Bases: `Protocol`

The Agent protocol defines the core interface for agents within the Diskurs framework.

Agents are central to the Diskurs system, responsible for processing conversations and
generating responses based on the current state of the conversation. Implementations of
this protocol must provide methods for creating agents and invoking their logic on
conversations. Agents interact with various components such as prompts, long-term memory,
and language model clients to facilitate meaningful and context-aware dialogues.

Key Responsibilities:
- Creating and initializing agents with specific configurations.
- Processing conversations by invoking the agent’s logic.
- Interacting with prompts and long-term memory to maintain conversation context.
- Generating responses using language model clients.

This protocol ensures that any implementing class can seamlessly integrate into the
Diskurs framework, enabling flexible and extensible conversation management.

#### name *: `str`*

#### *classmethod* create(name, prompt, llm_client, \*\*kwargs)

* **Parameters:**
  * **name** (*str*)
  * **prompt** ([*Prompt*](#diskurs.protocols.Prompt))
  * **llm_client** ([*LLMClient*](#diskurs.protocols.LLMClient))

#### invoke(conversation)

Run the agent on a conversation.

This method processes the given conversation by invoking the agent’s logic.
It takes a Conversation object representing the conversation
and returns an updated Conversation object after processing.

* **Parameters:**
  **conversation** ([`Conversation`](#diskurs.protocols.Conversation)) – The current state of the conversation, represented as a Conversation object
* **Return type:**
  [`Conversation`](#diskurs.protocols.Conversation)
* **Returns:**
  An updated Conversation object with the processed state.

### *class* diskurs.protocols.ConductorAgent(\*args, \*\*kwargs)

Bases: `Protocol`

The ConductorAgent protocol defines the interface for agents that manage and coordinate conversations.
Conductor agents are responsible for creating or updating long-term memory, invoking conversation logic,
and ensuring the conversation progresses smoothly. They are responsible for dispatching conversation to the
most appropriate agents to maintain context and generate responses. This protocol ensures that any
implementing class can effectively manage and direct conversations.

#### name *: `str`*

#### prompt *: [`ConductorPrompt`](#diskurs.protocols.ConductorPrompt)*

#### create_or_update_longterm_memory(conversation, overwrite=False)

Creates or updates the long-term memory for the conductor agent.

This method is responsible for either creating a new long-term memory instance
or updating an existing one based on the provided conversation. It ensures that
the long-term memory is synchronized with the current state of the conversation.

* **Parameters:**
  * **conversation** ([`Conversation`](#diskurs.protocols.Conversation)) – The current state of the conversation, represented as a Conversation object.
  * **overwrite** (`bool`) – A boolean flag indicating whether to overwrite existing memory fields. Defaults to False.
* **Return type:**
  [`Conversation`](#diskurs.protocols.Conversation)
* **Returns:**
  An updated Conversation object with the new or updated long-term memory.

### *class* diskurs.protocols.ToolExecutor(\*args, \*\*kwargs)

Bases: `Protocol`

The ToolExecutor protocol defines the interface for executing tools within the Diskurs framework.
Implementations of this protocol are responsible for registering tools, executing them based on tool calls,
and providing a mechanism to directly call specific tool functions. This protocol ensures that any implementing
class can manage and execute tools effectively, facilitating the integration of various tools into the conversation
processing workflow.

#### tools *: `Dict`[`str`, `Callable`]*

#### register_tools(tool_list)

Registers one or more tools with the executor.

This method allows the registration of a single tool or a list of tools
that can be executed by the executor. Each tool is a callable that can
be invoked with specific arguments.

* **Parameters:**
  **tool_list** (`Union`[`List`[`Callable`], `Callable`]) – A single callable or a list of callables representing the tools to be registered.
* **Return type:**
  `None`

#### execute_tool(tool_call, metadata)

Executes a registered tool based on the provided tool call and metadata.

This method is responsible for invoking a specific tool using the details
provided in the tool_call object. It utilizes the metadata to provide
additional context or parameters required for the tool execution. The result
of the tool execution is returned as a ToolCallResult object.

* **Parameters:**
  * **tool_call** (`ToolCall`) – The ToolCall object containing the details of the tool to be executed.
  * **metadata** (`Dict`[`str`, `Any`]) – A dictionary containing additional context or parameters for the tool execution.
* **Return type:**
  `ToolCallResult`
* **Returns:**
  A ToolCallResult object containing the result of the tool execution.

#### call_tool(function_name, arguments)

Calls a registered tool with the specified function name and arguments.

This method is responsible for invoking a tool that has been registered with the executor.
It uses the provided function name and arguments to execute the tool and returns the result.
It is meant to be used by a developer to directly call a specific tool function withing a
heuristic sequence.

* **Parameters:**
  * **function_name** (`str`) – The name of the tool function to be called.
  * **arguments** (`Dict`[`str`, `Any`]) – A dictionary containing the arguments to be passed to the tool function.
* **Return type:**
  `Any`
* **Returns:**
  The result of the tool function execution.