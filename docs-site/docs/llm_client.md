# Module: LLM Client

### *class* diskurs.llm_client.BaseOaiApiLLMClient(client, model, tokenizer, max_tokens, max_repeat=3)

Bases: [`LLMClient`](protocols.md#diskurs.protocols.LLMClient)

* **Parameters:**
  * **client** (*OpenAI*)
  * **model** (*str*)
  * **tokenizer** (*Callable* *[* *[**str* *]* *,* *int* *]*)
  * **max_tokens** (*int*)
  * **max_repeat** (*int*)

#### *abstract classmethod* create(\*\*kwargs)

* **Return type:**
  `Self`

#### send_request(body)

* **Return type:**
  `ChatCompletion`
* **Parameters:**
  **body** (*dict* *[**str* *,* *Any* *]*)

#### *static* format_tool_description_for_llm(tool)

Formats a ToolDescription object into a dictionary that can be sent to the LLM model.
:type tool: `ToolDescription`
:param tool: Tool description to be formatted
:rtype: `dict`[`str`, `Any`]
:return: JSON-serializable dictionary containing the tool data

* **Parameters:**
  **tool** (*ToolDescription*)
* **Return type:**
  dict[str, *Any*]

#### *static* format_message_for_llm(message)

Formats a ChatMessage object into a dictionary that can be sent to the LLM model.
Used by the format_conversation_for_llm method to prepare individual messages for the LLM.

* **Parameters:**
  **message** (`ChatMessage`) – Message to be formatted
* **Return type:**
  `dict`[`str`, `str`]
* **Returns:**
  JSON-serializable dictionary containing the message data

#### format_conversation_for_llm(conversation, tools=None)

Formats the conversation object into a dictionary that can be sent to the LLM model.
This comprises the user prompt, chat history, and tool descriptions.
:type conversation: [`ImmutableConversation`](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation)
:param conversation: Contains all interactions so far
:type tools: `Optional`[`list`[`ToolDescription`]]
:param tools: The descriptions of all tools that the agent can use
:rtype: `dict`[`str`, `Any`]
:return: A JSON-serializable dictionary containing the conversation data ready for the LLM

* **Parameters:**
  * **conversation** ([*ImmutableConversation*](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation))
  * **tools** (*list* *[**ToolDescription* *]*  *|* *None*)
* **Return type:**
  dict[str, *Any*]

#### *classmethod* is_tool_call(completion)

* **Return type:**
  `bool`
* **Parameters:**
  **completion** (*ChatCompletion*)

#### *classmethod* llm_response_to_chat_message(completion, agent_name, message_type)

Converts the message returned by the LLM to a typed ChatMessage.
:type completion: `ChatCompletion`
:param completion: The response from the LLM model
:type agent_name: `str`
:param agent_name: The name of the agent whose question the completion is a response to
:type message_type: `MessageType`
:param message_type: The type of message to be created
:rtype: `ChatMessage`
:return: A ChatMessage object containing the structured response

* **Parameters:**
  * **completion** (*ChatCompletion*)
  * **agent_name** (*str*)
  * **message_type** (*MessageType*)
* **Return type:**
  *ChatMessage*

#### *classmethod* concatenate_user_prompt_with_llm_response(conversation, completion)

Creates a list of ChatMessages that combines the user prompt with the LLM response.
Ensures a flat list, even if there are multiple messages in the user prompt (as is the case when
multiple tools are executed in a single pass).

* **Parameters:**
  * **conversation** ([`ImmutableConversation`](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation)) – the conversation containing the user prompt
  * **completion** (`ChatCompletion`) – the response from the LLM model
* **Return type:**
  `list`[`ChatMessage`]
* **Returns:**
  Flat list of ChatMessages containing the user prompt and LLM response

#### count_tokens_in_conversation(messages)

Count the number of tokens used by a list of messages i.e. chat history.
The implementation is based on OpenAI’s token counting guidelines.

* **Return type:**
  `int`
* **Parameters:**
  **messages** (*list* *[**dict* *]*)

#### count_tokens_recursively(value)

#### count_tokens(text)

Counts the number of tokens in a text string.
:type text: `str`
:param text: The text string to tokenize.
:rtype: `int`
:return: The number of tokens in the text string.

* **Parameters:**
  **text** (*str*)
* **Return type:**
  int

#### count_tokens_of_tool_descriptions(tool_descriptions)

Return the number of tokens used by the tool i.e. function description.
Unfortunately, there’s no documented way of counting those tokens, therefore we resort to best effort approach,
hoping this implementation is a true upper bound.
The implementation is taken from:
[https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11](https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11)

* **Parameters:**
  **tool_descriptions** (`list`[`dict`[`str`, `Any`]]) – The description of all the tools
* **Return type:**
  `int`
* **Returns:**
  The number of tokens used by the tools

#### truncate_chat_history(messages, n_tokens_tool_descriptions)

Truncate the chat history to fit within the maximum token limit. The token limit is calculated as follows:
We retain the first two messages i.e. system prompt and initial user prompt and the last message.
We then truncate from left, removing messages from the chat history until the total token count is within the
limit. We also account for the token count of the tool descriptions.

* **Parameters:**
  * **messages** – The list of messages in the conversation
  * **n_tokens_tool_descriptions** – The number of tokens used by the tool descriptions
* **Return type:**
  `list`[`dict`]
* **Returns:**
  The truncated chat history

#### generate(conversation, tools=None)

Generates a response from the LLM model for the given conversation.
Handles conversion from Conversation to LLM request format, sending the request to the LLM model,
and converting the response back to a Conversation object.

* **Parameters:**
  * **conversation** ([`ImmutableConversation`](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation)) – The conversation object containing the user prompt and chat history.
  * **tools** (`Optional`[`ToolDescription`]) – Description of all the tools that the agent can use
* **Return type:**
  [`ImmutableConversation`](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation)
* **Returns:**
  Updated conversation object with the LLM response appended to the chat history.

### *class* diskurs.llm_client.OpenAILLMClient(client, model, tokenizer, max_tokens, max_repeat=3)

Bases: [`BaseOaiApiLLMClient`](#diskurs.llm_client.BaseOaiApiLLMClient)

* **Parameters:**
  * **client** (*OpenAI*)
  * **model** (*str*)
  * **tokenizer** (*Callable* *[* *[**str* *]* *,* *int* *]*)
  * **max_tokens** (*int*)
  * **max_repeat** (*int*)

#### *classmethod* create(\*\*kwargs)

* **Return type:**
  `Self`

#### *classmethod* concatenate_user_prompt_with_llm_response(conversation, completion)

Creates a list of ChatMessages that combines the user prompt with the LLM response.
Ensures a flat list, even if there are multiple messages in the user prompt (as is the case when
multiple tools are executed in a single pass).

* **Parameters:**
  * **conversation** ([`ImmutableConversation`](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation)) – the conversation containing the user prompt
  * **completion** (`ChatCompletion`) – the response from the LLM model
* **Return type:**
  `list`[`ChatMessage`]
* **Returns:**
  Flat list of ChatMessages containing the user prompt and LLM response

#### count_tokens(text)

Counts the number of tokens in a text string.
:type text: `str`
:param text: The text string to tokenize.
:rtype: `int`
:return: The number of tokens in the text string.

* **Parameters:**
  **text** (*str*)
* **Return type:**
  int

#### count_tokens_in_conversation(messages)

Count the number of tokens used by a list of messages i.e. chat history.
The implementation is based on OpenAI’s token counting guidelines.

* **Return type:**
  `int`
* **Parameters:**
  **messages** (*list* *[**dict* *]*)

#### count_tokens_of_tool_descriptions(tool_descriptions)

Return the number of tokens used by the tool i.e. function description.
Unfortunately, there’s no documented way of counting those tokens, therefore we resort to best effort approach,
hoping this implementation is a true upper bound.
The implementation is taken from:
[https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11](https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11)

* **Parameters:**
  **tool_descriptions** (`list`[`dict`[`str`, `Any`]]) – The description of all the tools
* **Return type:**
  `int`
* **Returns:**
  The number of tokens used by the tools

#### count_tokens_recursively(value)

#### format_conversation_for_llm(conversation, tools=None)

Formats the conversation object into a dictionary that can be sent to the LLM model.
This comprises the user prompt, chat history, and tool descriptions.
:type conversation: [`ImmutableConversation`](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation)
:param conversation: Contains all interactions so far
:type tools: `Optional`[`list`[`ToolDescription`]]
:param tools: The descriptions of all tools that the agent can use
:rtype: `dict`[`str`, `Any`]
:return: A JSON-serializable dictionary containing the conversation data ready for the LLM

* **Parameters:**
  * **conversation** ([*ImmutableConversation*](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation))
  * **tools** (*list* *[**ToolDescription* *]*  *|* *None*)
* **Return type:**
  dict[str, *Any*]

#### *static* format_message_for_llm(message)

Formats a ChatMessage object into a dictionary that can be sent to the LLM model.
Used by the format_conversation_for_llm method to prepare individual messages for the LLM.

* **Parameters:**
  **message** (`ChatMessage`) – Message to be formatted
* **Return type:**
  `dict`[`str`, `str`]
* **Returns:**
  JSON-serializable dictionary containing the message data

#### *static* format_tool_description_for_llm(tool)

Formats a ToolDescription object into a dictionary that can be sent to the LLM model.
:type tool: `ToolDescription`
:param tool: Tool description to be formatted
:rtype: `dict`[`str`, `Any`]
:return: JSON-serializable dictionary containing the tool data

* **Parameters:**
  **tool** (*ToolDescription*)
* **Return type:**
  dict[str, *Any*]

#### generate(conversation, tools=None)

Generates a response from the LLM model for the given conversation.
Handles conversion from Conversation to LLM request format, sending the request to the LLM model,
and converting the response back to a Conversation object.

* **Parameters:**
  * **conversation** ([`ImmutableConversation`](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation)) – The conversation object containing the user prompt and chat history.
  * **tools** (`Optional`[`ToolDescription`]) – Description of all the tools that the agent can use
* **Return type:**
  [`ImmutableConversation`](immutable_conversation.md#diskurs.immutable_conversation.ImmutableConversation)
* **Returns:**
  Updated conversation object with the LLM response appended to the chat history.

#### *classmethod* is_tool_call(completion)

* **Return type:**
  `bool`
* **Parameters:**
  **completion** (*ChatCompletion*)

#### *classmethod* llm_response_to_chat_message(completion, agent_name, message_type)

Converts the message returned by the LLM to a typed ChatMessage.
:type completion: `ChatCompletion`
:param completion: The response from the LLM model
:type agent_name: `str`
:param agent_name: The name of the agent whose question the completion is a response to
:type message_type: `MessageType`
:param message_type: The type of message to be created
:rtype: `ChatMessage`
:return: A ChatMessage object containing the structured response

* **Parameters:**
  * **completion** (*ChatCompletion*)
  * **agent_name** (*str*)
  * **message_type** (*MessageType*)
* **Return type:**
  *ChatMessage*

#### send_request(body)

* **Return type:**
  `ChatCompletion`
* **Parameters:**
  **body** (*dict* *[**str* *,* *Any* *]*)

#### truncate_chat_history(messages, n_tokens_tool_descriptions)

Truncate the chat history to fit within the maximum token limit. The token limit is calculated as follows:
We retain the first two messages i.e. system prompt and initial user prompt and the last message.
We then truncate from left, removing messages from the chat history until the total token count is within the
limit. We also account for the token count of the tool descriptions.

* **Parameters:**
  * **messages** – The list of messages in the conversation
  * **n_tokens_tool_descriptions** – The number of tokens used by the tool descriptions
* **Return type:**
  `list`[`dict`]
* **Returns:**
  The truncated chat history
