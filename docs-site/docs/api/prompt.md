# Module: Prompt

### *exception* diskurs.prompt.PromptValidationError(message)

Bases: `Exception`

* **Parameters:**
  **message** (*str*)

#### add_note()

Exception.add_note(note) –
add a note to the exception

#### args

#### with_traceback()

Exception.with_traceback(tb) –
set self._\_traceback_\_ to tb and return self.

### *class* diskurs.prompt.DefaultConductorSystemPromptArgument(agent_descriptions)

Bases: `PromptArgument`

* **Parameters:**
  **agent_descriptions** (*dict* *[**str* *,* *str* *]*)

#### agent_descriptions *: `dict`[`str`, `str`]*

#### *classmethod* from_dict(data)

#### to_dict()

### *class* diskurs.prompt.DefaultConductorUserPromptArgument(next_agent=None)

Bases: `PromptArgument`

* **Parameters:**
  **next_agent** (*str* *|* *None*)

#### next_agent *: `Optional`[`str`]* *= None*

#### *classmethod* from_dict(data)

#### to_dict()

### diskurs.prompt.load_symbol(symbol_name, loaded_module)

### diskurs.prompt.load_template(location)

Loads a Jinja2 template from the provided file path.

* **Parameters:**
  **location** (`Path`) – Path to the template file.
* **Return type:**
  `Template`
* **Returns:**
  Jinja2 Template object.
* **Raises:**
  **FileNotFoundError** – If the template file does not exist.

### *class* diskurs.prompt.PromptParserMixin

Bases: `object`

Mixin to handle parsing and validation of LLM responses into dataclasses.

#### *classmethod* validate_dataclass(parsed_response, user_prompt_argument, strict=False)

Validate that the JSON fields match the target dataclass.
In strict mode, all fields must be present. If not strict, all required fields (without default values)
must be present at minimum.

* **Parameters:**
  * **parsed_response** (`dict`[`str`, `Any`]) – Dictionary representing the LLM’s response.
  * **user_prompt_argument** (`Type`[`dataclass`]) – The dataclass type to validate against.
  * **strict** (`bool`) – If True, all fields must be present. If False, required fields must be present.
* **Return type:**
  `dataclass`
* **Returns:**
  An instance of the dataclass if validation succeeds.
* **Raises:**
  LLMResponseParseError if validation fails.

#### *classmethod* validate_json(llm_response)

Parse and validate the LLM response as JSON, handling nested JSON strings.

* **Parameters:**
  **llm_response** (`str`) – The raw text response from the LLM.
* **Return type:**
  `dict`
* **Returns:**
  Parsed dictionary if valid JSON.
* **Raises:**
  [**PromptValidationError**](#diskurs.prompt.PromptValidationError) – If the response is not valid JSON.

#### parse_user_prompt(llm_response, old_user_prompt_argument, message_type=MessageType.ROUTING)

Parse the text returned from the LLM into a structured prompt argument.
First validate the text, then parse it into the prompt argument.
If the text is not valid, raise a PromptValidationError, and generate a user prompt with the error message,
for the LLM to correct its output.

* **Parameters:**
  * **llm_response** (`str`) – Response from the LLM.
  * **old_user_prompt_argument** (`PromptArgument`) – The previous user prompt argument.
  * **message_type** (`MessageType`) – Type of message to be created.
* **Return type:**
  `PromptArgument` | `ChatMessage`
* **Returns:**
  Validated prompt argument or a ChatMessage with an error message.
* **Raises:**
  [**PromptValidationError**](#diskurs.prompt.PromptValidationError) – If the text is not valid.

### *class* diskurs.prompt.PromptRendererMixin(system_prompt_argument_class, user_prompt_argument_class, system_template, user_template, json_formatting_template=None, is_valid=None, is_final=None)

Bases: `object`

Mixin to handle the rendering of system and user templates, as well as validation
This class can be reused across prompts.

* **Parameters:**
  * **system_prompt_argument_class** (*Type* *[**SystemPromptArg* *]*)
  * **user_prompt_argument_class** (*Type* *[**UserPromptArg* *]*)
  * **system_template** (*Template*)
  * **user_template** (*Template*)
  * **json_formatting_template** (*Template* *|* *None*)
  * **is_valid** (*Callable* *[* *[**UserPromptArg* *]* *,* *bool* *]*  *|* *None*)
  * **is_final** (*Callable* *[* *[**UserPromptArg* *]* *,* *bool* *]*  *|* *None*)

#### is_final(user_prompt_argument)

* **Return type:**
  `bool`
* **Parameters:**
  **user_prompt_argument** (*PromptArgument*)

#### is_valid(user_prompt_argument)

* **Return type:**
  `bool`
* **Parameters:**
  **user_prompt_argument** (*PromptArgument*)

#### create_system_prompt_argument(\*\*prompt_args)

* **Return type:**
  `TypeVar`(`SystemPromptArg`)
* **Parameters:**
  **prompt_args** (*dict*)

#### create_user_prompt_argument(\*\*prompt_args)

* **Return type:**
  `TypeVar`(`UserPromptArg`)
* **Parameters:**
  **prompt_args** (*dict*)

#### render_json_formatting_prompt(prompt_args)

* **Return type:**
  `str`
* **Parameters:**
  **prompt_args** (*dict*)

#### render_system_template(name, prompt_args, return_json=True)

* **Return type:**
  `ChatMessage`
* **Parameters:**
  * **name** (*str*)
  * **prompt_args** (*PromptArgument*)
  * **return_json** (*bool*)

#### render_user_template(name, prompt_args, message_type=MessageType.CONVERSATION)

* **Return type:**
  `ChatMessage`
* **Parameters:**
  * **name** (*str*)
  * **prompt_args** (*PromptArgument*)
  * **message_type** (*MessageType*)

### *class* diskurs.prompt.PromptLoaderMixin

Bases: `object`

#### *classmethod* prepare_create(agent_description_filename, code_filename, kwargs, location, system_prompt_argument_class, system_template_filename, user_prompt_argument_class, user_template_filename)

#### *classmethod* load_prompt_functions(system_prompt_argument_class, user_prompt_argument_class, loaded_module, kwargs)

* **Return type:**
  `dict`[`str`, `Callable`]

#### *classmethod* load_user_assets(agent_description_filename, code_filename, location, system_template_filename, user_template_filename)

### *class* diskurs.prompt.MultistepPrompt(agent_description, system_template, user_template, system_prompt_argument_class, user_prompt_argument_class, json_formatting_template=None, is_valid=None, is_final=None)

Bases: [`PromptRendererMixin`](#diskurs.prompt.PromptRendererMixin), [`PromptParserMixin`](#diskurs.prompt.PromptParserMixin), [`PromptLoaderMixin`](#diskurs.prompt.PromptLoaderMixin), [`MultistepPrompt`](protocols.md#diskurs.protocols.MultistepPrompt)

* **Parameters:**
  * **agent_description** (*str*)
  * **system_template** (*Template*)
  * **user_template** (*Template*)
  * **system_prompt_argument_class** (*Type* *[**SystemPromptArg* *]*)
  * **user_prompt_argument_class** (*Type* *[**UserPromptArg* *]*)
  * **json_formatting_template** (*Template* *|* *None*)
  * **is_valid** (*Callable* *[* *[**Any* *]* *,* *bool* *]*  *|* *None*)
  * **is_final** (*Callable* *[* *[**Any* *]* *,* *bool* *]*  *|* *None*)

#### *classmethod* create(location, system_prompt_argument_class, user_prompt_argument_class, agent_description_filename='agent_description.txt', code_filename='prompt.py', user_template_filename='user_template.jinja2', system_template_filename='system_template.jinja2', \*\*kwargs)

Factory method to create a Prompt object. Loads templates and code dynamically
based on the provided directory and filenames.

* **Parameters:**
  * **location** (`Path`) – Base path where prompt.py and templates are located.
  * **system_prompt_argument_class** (`str`) – Name of the class that specifies the placeholders of the system prompt template
  * **user_prompt_argument_class** (`str`) – Name of the class that specifies the placeholders of the user prompt template
  * **agent_description_filename** (`str`) – location of the text file containing the agent’s description
  * **code_filename** (`str`) – Name of the file containing PromptArguments and validation logic.
  * **user_template_filename** (`str`) – Filename of the user template (Jinja2 format).
  * **system_template_filename** (`str`) – Filename of the system template (Jinja2 format).
* **Return type:**
  `Self`
* **Returns:**
  An instance of the Prompt class.

#### *classmethod* load_prompt_functions(system_prompt_argument_class, user_prompt_argument_class, loaded_module, kwargs)

* **Return type:**
  `dict`[`str`, `Callable`]

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

#### create_system_prompt_argument(\*\*prompt_args)

Creates an instance of the system prompt argument dataclass.

This method is responsible for generating the system prompt argument
based on the provided keyword arguments. The system prompt argument
is used to configure the initial state and context for the conversation.

* **Parameters:**
  **prompt_args** (`dict`) – Keyword arguments used to initialize the system prompt argument.
* **Return type:**
  `TypeVar`(`SystemPromptArg`)
* **Returns:**
  An instance of the system prompt argument dataclass.

#### create_user_prompt_argument(\*\*prompt_args)

Creates an instance of the user prompt argument dataclass.

This method is responsible for generating the user prompt argument
based on the provided keyword arguments. The user prompt argument
is used to configure the user’s input and context for the conversation.

* **Parameters:**
  **prompt_args** (`dict`) – Keyword arguments used to initialize the user prompt argument.
* **Return type:**
  `TypeVar`(`UserPromptArg`)
* **Returns:**
  An instance of the user prompt argument dataclass.

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

#### *classmethod* load_user_assets(agent_description_filename, code_filename, location, system_template_filename, user_template_filename)

#### parse_user_prompt(llm_response, old_user_prompt_argument, message_type=MessageType.ROUTING)

Parse the text returned from the LLM into a structured prompt argument.
First validate the text, then parse it into the prompt argument.
If the text is not valid, raise a PromptValidationError, and generate a user prompt with the error message,
for the LLM to correct its output.

* **Parameters:**
  * **llm_response** (`str`) – Response from the LLM.
  * **old_user_prompt_argument** (`PromptArgument`) – The previous user prompt argument.
  * **message_type** (`MessageType`) – Type of message to be created.
* **Return type:**
  `PromptArgument` | `ChatMessage`
* **Returns:**
  Validated prompt argument or a ChatMessage with an error message.
* **Raises:**
  [**PromptValidationError**](#diskurs.prompt.PromptValidationError) – If the text is not valid.

#### *classmethod* prepare_create(agent_description_filename, code_filename, kwargs, location, system_prompt_argument_class, system_template_filename, user_prompt_argument_class, user_template_filename)

#### render_json_formatting_prompt(prompt_args)

* **Return type:**
  `str`
* **Parameters:**
  **prompt_args** (*dict*)

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

#### *classmethod* validate_dataclass(parsed_response, user_prompt_argument, strict=False)

Validate that the JSON fields match the target dataclass.
In strict mode, all fields must be present. If not strict, all required fields (without default values)
must be present at minimum.

* **Parameters:**
  * **parsed_response** (`dict`[`str`, `Any`]) – Dictionary representing the LLM’s response.
  * **user_prompt_argument** (`Type`[`dataclass`]) – The dataclass type to validate against.
  * **strict** (`bool`) – If True, all fields must be present. If False, required fields must be present.
* **Return type:**
  `dataclass`
* **Returns:**
  An instance of the dataclass if validation succeeds.
* **Raises:**
  LLMResponseParseError if validation fails.

#### *classmethod* validate_json(llm_response)

Parse and validate the LLM response as JSON, handling nested JSON strings.

* **Parameters:**
  **llm_response** (`str`) – The raw text response from the LLM.
* **Return type:**
  `dict`
* **Returns:**
  Parsed dictionary if valid JSON.
* **Raises:**
  [**PromptValidationError**](#diskurs.prompt.PromptValidationError) – If the response is not valid JSON.

#### system_prompt_argument *: `Type`[`TypeVar`(`SystemPromptArg`)]*

#### user_prompt_argument *: `Type`[`TypeVar`(`UserPromptArg`)]*

### *class* diskurs.prompt.ConductorPrompt(agent_description, system_template, user_template, system_prompt_argument_class, user_prompt_argument_class, json_formatting_template=None, longterm_memory_class=None, can_finalize=None, finalize=None, fail=None, topics=None)

Bases: [`PromptRendererMixin`](#diskurs.prompt.PromptRendererMixin), [`PromptParserMixin`](#diskurs.prompt.PromptParserMixin), [`PromptLoaderMixin`](#diskurs.prompt.PromptLoaderMixin), [`ConductorPrompt`](protocols.md#diskurs.protocols.ConductorPrompt)

* **Parameters:**
  * **agent_description** (*str*)
  * **system_template** (*Template*)
  * **user_template** (*Template*)
  * **system_prompt_argument_class** (*Type* *[**SystemPromptArg* *]*)
  * **user_prompt_argument_class** (*Type* *[**UserPromptArg* *]*)
  * **json_formatting_template** (*Template* *|* *None*)
  * **longterm_memory_class** (*Type* *[**GenericConductorLongtermMemory* *]*)
  * **can_finalize** (*Callable* *[* *[**GenericConductorLongtermMemory* *]* *,* *bool* *]*)
  * **finalize** (*Callable* *[* *[**GenericConductorLongtermMemory* *]* *,* *dict* *[**str* *,* *Any* *]* *]*)
  * **fail** (*Callable* *[* *[**GenericConductorLongtermMemory* *]* *,* *dict* *[**str* *,* *Any* *]* *]*)
  * **topics** (*list* *[**str* *]*)

#### longterm_memory *: Type[LongtermMemory]*

#### *static* create_default_is_valid(topics)

* **Return type:**
  `Callable`[[`TypeVar`(`UserPromptArg`)], `bool`]
* **Parameters:**
  **topics** (*list* *[**str* *]*)

#### *static* create_default_is_final(topics)

* **Return type:**
  `Callable`[[`TypeVar`(`UserPromptArg`)], `bool`]
* **Parameters:**
  **topics** (*list* *[**str* *]*)

#### *classmethod* create(location, system_prompt_argument_class, user_prompt_argument_class=None, agent_description_filename='agent_description.txt', code_filename='prompt.py', user_template_filename=None, system_template_filename=None, \*\*kwargs)

* **Return type:**
  [`ConductorPrompt`](#diskurs.prompt.ConductorPrompt)
* **Parameters:**
  * **location** (*Path*)
  * **system_prompt_argument_class** (*str*)
  * **user_prompt_argument_class** (*str* *|* *None*)
  * **agent_description_filename** (*str* *|* *None*)
  * **code_filename** (*str*)
  * **user_template_filename** (*str* *|* *None*)
  * **system_template_filename** (*str* *|* *None*)

#### *classmethod* load_prompt_functions(system_prompt_argument_class, user_prompt_argument_class, loaded_module, kwargs)

* **Return type:**
  `dict`[`str`, `Callable`]

#### can_finalize(longterm_memory)

Determines if the conductor can finalize based on the long-term memory.

This method evaluates the provided long-term memory to decide whether the
conversation can be concluded. It checks if the necessary conditions are met
for finalizing the conversation.

* **Parameters:**
  **longterm_memory** (`TypeVar`(`GenericConductorLongtermMemory`, bound= ConductorLongtermMemory)) – The long-term memory to be evaluated.
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
  **longterm_memory** (`TypeVar`(`GenericConductorLongtermMemory`, bound= ConductorLongtermMemory)) – The long-term memory to be used for finalizing the conversation.
* **Return type:**
  `TypeVar`(`GenericConductorLongtermMemory`, bound= ConductorLongtermMemory)
* **Returns:**
  A dictionary containing the final response data.

#### fail(longterm_memory)

Handles the case failure i.e. when the long-term memory does not meet the criteria defined
for finalization.

This method processes the provided long-term memory to generate a failure response
for the conversation. It is responsible for concluding the conversation in a failure
state by utilizing the accumulated long-term memory data.

* **Parameters:**
  **longterm_memory** (`TypeVar`(`GenericConductorLongtermMemory`, bound= ConductorLongtermMemory)) – The long-term memory to be used for generating the failure response.
* **Return type:**
  `TypeVar`(`GenericConductorLongtermMemory`, bound= ConductorLongtermMemory)
* **Returns:**
  A dictionary containing the failure response data.

#### init_longterm_memory(\*\*kwargs)

Initializes the long-term memory.

This method is responsible for creating and initializing the long-term memory
for the conductor agent. It uses the provided keyword arguments to set up the
initial state of the long-term memory.

* **Parameters:**
  **kwargs** – Keyword arguments used to initialize the long-term memory.
* **Return type:**
  `TypeVar`(`GenericConductorLongtermMemory`, bound= ConductorLongtermMemory)
* **Returns:**
  An instance of the LongtermMemory class.

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

#### create_system_prompt_argument(\*\*prompt_args)

Creates an instance of the system prompt argument dataclass.

This method is responsible for generating the system prompt argument
based on the provided keyword arguments. The system prompt argument
is used to configure the initial state and context for the conversation.

* **Parameters:**
  **prompt_args** (`dict`) – Keyword arguments used to initialize the system prompt argument.
* **Return type:**
  `TypeVar`(`SystemPromptArg`)
* **Returns:**
  An instance of the system prompt argument dataclass.

#### create_user_prompt_argument(\*\*prompt_args)

Creates an instance of the user prompt argument dataclass.

This method is responsible for generating the user prompt argument
based on the provided keyword arguments. The user prompt argument
is used to configure the user’s input and context for the conversation.

* **Parameters:**
  **prompt_args** (`dict`) – Keyword arguments used to initialize the user prompt argument.
* **Return type:**
  `TypeVar`(`UserPromptArg`)
* **Returns:**
  An instance of the user prompt argument dataclass.

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

#### *classmethod* load_user_assets(agent_description_filename, code_filename, location, system_template_filename, user_template_filename)

#### parse_user_prompt(llm_response, old_user_prompt_argument, message_type=MessageType.ROUTING)

Parse the text returned from the LLM into a structured prompt argument.
First validate the text, then parse it into the prompt argument.
If the text is not valid, raise a PromptValidationError, and generate a user prompt with the error message,
for the LLM to correct its output.

* **Parameters:**
  * **llm_response** (`str`) – Response from the LLM.
  * **old_user_prompt_argument** (`PromptArgument`) – The previous user prompt argument.
  * **message_type** (`MessageType`) – Type of message to be created.
* **Return type:**
  `PromptArgument` | `ChatMessage`
* **Returns:**
  Validated prompt argument or a ChatMessage with an error message.
* **Raises:**
  [**PromptValidationError**](#diskurs.prompt.PromptValidationError) – If the text is not valid.

#### *classmethod* prepare_create(agent_description_filename, code_filename, kwargs, location, system_prompt_argument_class, system_template_filename, user_prompt_argument_class, user_template_filename)

#### render_json_formatting_prompt(prompt_args)

* **Return type:**
  `str`
* **Parameters:**
  **prompt_args** (*dict*)

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

#### *classmethod* validate_dataclass(parsed_response, user_prompt_argument, strict=False)

Validate that the JSON fields match the target dataclass.
In strict mode, all fields must be present. If not strict, all required fields (without default values)
must be present at minimum.

* **Parameters:**
  * **parsed_response** (`dict`[`str`, `Any`]) – Dictionary representing the LLM’s response.
  * **user_prompt_argument** (`Type`[`dataclass`]) – The dataclass type to validate against.
  * **strict** (`bool`) – If True, all fields must be present. If False, required fields must be present.
* **Return type:**
  `dataclass`
* **Returns:**
  An instance of the dataclass if validation succeeds.
* **Raises:**
  LLMResponseParseError if validation fails.

#### *classmethod* validate_json(llm_response)

Parse and validate the LLM response as JSON, handling nested JSON strings.

* **Parameters:**
  **llm_response** (`str`) – The raw text response from the LLM.
* **Return type:**
  `dict`
* **Returns:**
  Parsed dictionary if valid JSON.
* **Raises:**
  [**PromptValidationError**](#diskurs.prompt.PromptValidationError) – If the response is not valid JSON.

#### system_prompt_argument *: Type[SystemPromptArg]*

#### user_prompt_argument *: Type[UserPromptArg]*

### *class* diskurs.prompt.HeuristicPrompt(user_prompt_argument_class, heuristic_sequence, user_template=None, agent_description='')

Bases: [`HeuristicPrompt`](protocols.md#diskurs.protocols.HeuristicPrompt)

* **Parameters:**
  * **user_prompt_argument_class** (*Type* *[**PromptArgument* *]*)
  * **heuristic_sequence** ([*HeuristicSequence*](protocols.md#diskurs.protocols.HeuristicSequence))
  * **user_template** (*Template* *|* *None*)
  * **agent_description** (*str* *|* *None*)

#### user_prompt_argument *: `Type`[`PromptArgument`]*

#### *classmethod* create(location, user_prompt_argument_class, agent_description_filename='agent_description.txt', user_template_filename='user_template.jinja2', code_filename='prompt.py', heuristic_sequence_name='heuristic_sequence', \*\*kwargs)

* **Return type:**
  `Self`
* **Parameters:**
  * **location** (*Path*)
  * **user_prompt_argument_class** (*str*)
  * **agent_description_filename** (*str*)
  * **user_template_filename** (*str*)
  * **code_filename** (*str*)
  * **heuristic_sequence_name** (*str*)

#### heuristic_sequence(conversation, call_tool=None)

Executes a heuristic sequence on the given conversation.

This method processes the conversation using a series of heuristic steps,
optionally utilizing a tool for specific operations. The heuristic sequence
is designed to guide the conversation towards a desired outcome based on
predefined rules and logic.

* **Parameters:**
  * **conversation** ([`Conversation`](protocols.md#diskurs.protocols.Conversation)) – The conversation object to be processed.
  * **call_tool** (`Optional`[[`CallTool`](protocols.md#diskurs.protocols.CallTool)]) – An optional callable tool that can be used during the heuristic sequence.
* **Return type:**
  [`Conversation`](protocols.md#diskurs.protocols.Conversation)
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
