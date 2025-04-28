from typing import Any, Callable, Dict, List, Optional, Protocol, Self, Type, Union, runtime_checkable

from jinja2 import Template

from diskurs.entities import (
    ChatMessage,
    LongtermMemory,
    MessageType,
    PromptArgument,
    Role,
    ToolCall,
    ToolCallResult,
    ToolDescription,
)


class LongtermMemoryHandler(Protocol):
    def can_finalize(self, longterm_memory: Any) -> bool: ...


class PromptValidator(Protocol):
    """
    Protocol for validating prompt responses.

    Defines methods for validating responses from a language model (LLM),
    including conversion to dataclass instances and JSON parsing.
    """

    @classmethod
    def validate_dataclass(
        cls, parsed_response: Dict[str, Any], prompt_argument: Type[PromptArgument]
    ) -> PromptArgument:
        """
        Validates a parsed response dictionary against a dataclass type.

        :param parsed_response: The dictionary containing the parsed response from the LLM.
        :param prompt_argument: The dataclass type to validate against.
        :param strict: If True, enforce strict validation rules.

        :return: An instance of the dataclass populated with the validated data.
        """
        ...

    @classmethod
    def validate_json(cls, llm_response: str) -> Union[Dict[str, Any], List[Any]]:
        """
        Validates a JSON response from a language model (LLM).

        This method takes a JSON string response from an LLM and validates it,
        ensuring it conforms to the expected structure and content.

        :param llm_response: The JSON string response from the LLM.
        :return: A dictionary representation of the validated JSON response.
        """
        ...


class Prompt(Protocol):
    """
    Protocol for prompt implementations.

    This protocol defines the interface for creating and handling prompts in a conversation.
    Prompt implementations are responsible for:
    - Creating and validating prompt arguments
    - Rendering system and user templates
    - Parsing responses from language models
    - Determining when a conversation has reached its final state

    Prompts serve as the bridge between raw LLM responses and structured data that agents
    can use to reason about conversations and make decisions.
    """

    prompt_argument: Type[PromptArgument]
    agent_description: str
    system_template: Template
    user_template: Template
    json_formatting_template: Optional[Template]

    def create_prompt_argument(self, **prompt_args: Any) -> PromptArgument:
        """
        Creates an instance of the prompt argument dataclass.

        This method instantiates a prompt argument dataclass with the provided keyword arguments,
        which will be used to render templates and structure conversation data.

        :param prompt_args: Keyword arguments used to initialize the prompt argument.
        :return: An instance of the prompt argument dataclass.
        """
        ...

    def render_user_template(
        self,
        name: str,
        prompt_args: PromptArgument,
        message_type: MessageType = MessageType.CONVERSATION,
    ) -> ChatMessage:
        """
        Renders the user template with the provided prompt arguments.

        This method generates a ChatMessage using prompt arguments to populate the user template.
        The rendered template becomes part of the conversation flow, representing either
        a system instruction or user input.

        :param name: The name of the agent for which the template is being rendered.
        :param prompt_args: The prompt arguments to use for rendering.
        :param message_type: The type of message to generate (defaults to CONVERSATION).
        :return: A ChatMessage object containing the rendered content.
        """
        ...

    def render_system_template(
        self, name: str, prompt_argument: PromptArgument, return_json: bool = True
    ) -> ChatMessage:
        """
        Renders the system template with the provided prompt arguments.

        This method generates a system message that provides context and instructions
        to guide the language model's responses. It can optionally append JSON formatting
        instructions to help the model produce structured outputs.

        :param name: The name of the agent for which the template is being rendered.
        :param prompt_argument: The prompt arguments to use for rendering.
        :param return_json: Whether to include JSON formatting instructions.
        :return: A ChatMessage object containing the rendered system prompt.
        """
        ...

    def render_json_formatting_prompt(self, prompt_argument: PromptArgument) -> str:
        """
        Renders instructions to format responses as structured JSON.

        This method generates a string containing JSON format instructions that help
        the language model produce responses in a valid, structured JSON format matching
        the expected prompt argument structure.

        :param prompt_args: Dictionary of prompt arguments to shape the formatting instructions.
        :return: A string containing the JSON formatting instructions.
        """
        ...

    def parse_user_prompt(
        self,
        name: str,
        llm_response: str,
        old_prompt_argument: PromptArgument,
        message_type: MessageType = MessageType.CONDUCTOR,
    ) -> PromptArgument | ChatMessage:
        """
        Parses an LLM response into either a prompt argument or error message.

        This method attempts to parse the LLM's raw text response into a structured prompt
        argument. If successful, the resulting prompt argument is returned. If the response
        fails validation, a ChatMessage with an error description is returned instead.

        :param name: Name of the agent parsing the response.
        :param llm_response: Raw text response from the language model.
        :param old_prompt_argument: Previous prompt argument, used for merging partial updates.
        :param message_type: Message type to use if creating an error message.
        :return: Either a valid PromptArgument or a ChatMessage containing an error message.
        """
        ...

    def is_final(self, prompt_argument: PromptArgument) -> bool:
        """
        Determines if the prompt argument represents the final state of a conversation.

        This method evaluates whether the current prompt argument indicates that the
        conversation has reached its conclusion and no further processing is needed.

        :param prompt_argument: The prompt argument to evaluate.
        :return: True if the conversation should be considered complete, False otherwise.
        """
        ...

    def is_valid(self, prompt_argument: PromptArgument) -> bool:
        """
        Validates the prompt argument against expected criteria.

        This method checks if the prompt argument contains all required fields with
        appropriate values. It may raise a PromptValidationError if validation fails.

        :param prompt_argument: The prompt argument to validate.
        :return: True if the prompt argument is valid, False otherwise.
        """
        ...

    def initialize_prompt(
        self,
        agent_name: str,
        conversation: "Conversation",
        locked_fields: Optional[dict[str, Any]] = None,
        init_from_longterm_memory: bool = True,
        reset_prompt: bool = True,
        message_type: MessageType = MessageType.CONVERSATION,
        render_system_prompt: bool = True,
    ) -> "Conversation":
        """
        Initializes the prompt for the conversation.

        This method sets up the initial state of the prompt for the conversation,
        including locked fields, long-term memory, and system prompt rendering.

        :param agent_name: The name of the agent initializing the prompt.
        :param conversation: The conversation object to initialize the prompt for.
        :param locked_fields: Optional dictionary of fields to lock during initialization.
        :param init_from_longterm_memory: Whether to initialize from long-term memory.
        :param reset_prompt: Whether to reset the prompt state.
        :param message_type: The type of message to generate (defaults to CONVERSATION).
        :param render_system_prompt: Whether to render the system prompt.
        :return: The updated conversation object with the initialized prompt.
        """
        ...


class MultistepPrompt(Prompt):
    """
    Protocol for multi-step prompts.

    MultistepPrompt extends the base Prompt protocol to provide structured handling of
    conversations that require multiple steps or follow a specific progression. It's designed
    for agent interactions that follow a defined sequence rather than free-form conversation.

    This type of prompt is particularly useful when:
    - Conversations need to follow a specific workflow or decision tree
    - Information needs to be collected in a structured, sequential manner
    - The agent needs to maintain a clear state progression through the conversation
    """

    def render_system_template(
        self, name: str, prompt_argument: PromptArgument, return_json: bool = True
    ) -> ChatMessage:
        """
        Renders the system template with the provided prompt arguments.

        This method generates a system message that provides context and instructions
        to guide the language model's responses. It can optionally append JSON formatting
        instructions to help the model produce structured outputs.

        :param name: The name of the agent for which the template is being rendered.
        :param prompt_argument: The prompt arguments to use for rendering the template.
        :param return_json: Whether to include JSON formatting instructions.
        :return: A ChatMessage object containing the rendered system prompt.
        """
        ...


class ConductorPrompt(Prompt):
    """
    Protocol for conductor prompts.

    This protocol defines the methods required for handling conductor prompts in a conversation.
    It includes methods for initializing, finalizing, and validating long-term memory, as well as
    determining the final state of the conversation. Conductor prompts are central to the routing
    and orchestration of agents in multi-agent conversations.
    """

    longterm_memory: Type[LongtermMemory]

    def render_system_template(
        self, name: str, prompt_argument: PromptArgument, return_json: bool = True
    ) -> ChatMessage:
        """
        Renders the system template with the provided prompt arguments.

        This method is responsible for rendering the system template using the given prompt arguments.
        It can optionally append JSON formatting instructions to the rendered template.

        :param name: The name of the agent for which the template is being rendered.
        :param prompt_argument: The prompt arguments to be used for rendering the template.
        :param return_json: Whether to include JSON formatting instructions.
        :return: A ChatMessage object containing the rendered template.
        """
        ...

    def can_finalize(self, longterm_memory: LongtermMemory) -> bool:
        """
        Determines if the conductor can finalize based on the long-term memory.

        This method evaluates the provided long-term memory to decide whether the
        conversation can be concluded. It checks if the necessary conditions are met
        for finalizing the conversation.

        :param longterm_memory: The long-term memory to be evaluated.
        :return: True if the conversation can be finalized, False otherwise.
        """
        ...

    def finalize(self, longterm_memory: LongtermMemory) -> Any:
        """
        Finalizes the conversation based on the long-term memory.

        This method processes the provided long-term memory to generate a final response
        for the conversation. It is responsible for concluding the conversation by
        utilizing the accumulated long-term memory data.

        :param longterm_memory: The long-term memory to be used for finalizing the conversation.
        :return: A dictionary containing the final response data.
        """
        ...

    def fail(self, longterm_memory: LongtermMemory) -> Any:
        """
        Handles the case of failure when the long-term memory does not meet the criteria defined
        for finalization.

        This method processes the provided long-term memory to generate a failure response
        for the conversation. It is responsible for concluding the conversation in a failure
        state by utilizing the accumulated long-term memory data.

        :param longterm_memory: The long-term memory to be used for generating the failure response.
        :return: A dictionary containing the failure response data.
        """
        ...

    def init_longterm_memory(self, **kwargs: Any) -> LongtermMemory:
        """
        Initializes the long-term memory.

        This method is responsible for creating and initializing the long-term memory
        for the conductor agent. It uses the provided keyword arguments to set up the
        initial state of the long-term memory.

        :param kwargs: Keyword arguments used to initialize the long-term memory.
        :return: An instance of the LongtermMemory class.
        """
        ...


class CallTool(Protocol):
    def __call__(self, function_name: str, arguments: dict[str, Any]) -> Any: ...


class HeuristicSequence(Protocol):
    async def __call__(
        self, conversation: "Conversation", call_tool: Optional[CallTool], llm_client: Optional["LLMClient"] = None
    ) -> "Conversation": ...


class HeuristicPrompt(Protocol):
    """
    Protocol for heuristic prompts.

    This protocol defines the structure and methods required for creating and handling heuristic prompts
    in a conversation. Implementations of this protocol are responsible for generating user prompt arguments,
    rendering user templates, and executing heuristic sequences to process conversations.

    Heuristic prompts provide a way to implement custom logic for processing conversations rather than
    relying solely on LLM responses. They enable programmatic control of conversation flow with detailed
    step-by-step processing.
    """

    prompt_argument: Type[PromptArgument]
    agent_description: str
    user_template: Optional[Template]

    async def heuristic_sequence(
        self, conversation: "Conversation", call_tool: Optional[CallTool], llm_client: Optional["LLMClient"] = None
    ) -> "Conversation":
        """
        Executes a heuristic sequence on the given conversation.

        This method processes the conversation using a series of heuristic steps,
        utilizing a tool caller and optionally an LLM client. The heuristic sequence
        is designed to guide the conversation towards a desired outcome based on
        predefined rules and logic.

        :param conversation: The conversation object to be processed.
        :param call_tool: A callable tool function that can be used during the heuristic sequence.
        :param llm_client: Optional LLM client that can be used for generating responses within the sequence.
        :return: The updated conversation object after processing the heuristic sequence.
        """
        ...

    def create_prompt_argument(self, **prompt_args) -> PromptArgument:
        """
        Creates an instance of the user prompt argument dataclass.

        This method is responsible for generating the user prompt argument
        based on the provided keyword arguments. The user prompt argument
        is used to configure the user's input and context for the conversation.

        :param prompt_args: Keyword arguments used to initialize the user prompt argument.
        :return: An instance of the user prompt argument dataclass.
        """
        ...

    def render_user_template(
        self,
        name: str,
        prompt_args: PromptArgument,
        message_type: MessageType = MessageType.CONVERSATION,
    ) -> ChatMessage:
        """
        Renders the user template with the provided prompt arguments.

        This method is responsible for rendering the user template using the given prompt arguments.
        It generates a `ChatMessage` object that represents the rendered template.

        :param name: The name of the template to be rendered.
        :param prompt_args: The prompt arguments to be used for rendering the template.
        :param message_type: The type of the message to be rendered. Defaults to `MessageType.CONVERSATION`.
        :return: A `ChatMessage` object containing the rendered template.
        """
        ...

    ...


class Conversation(Protocol):
    """
    Protocol for conversation management.

    This protocol defines the interface for immutable conversation objects that maintain
    the state of a conversation between agents. Conversations contain:
    - Chat history (messages exchanged)
    - Prompt arguments (structured data used by agents)
    - System and user prompts (context for LLMs)
    - Long-term memory (persistent state across conversation turns)
    - Metadata (additional conversation context)

    All operations on conversation objects return new instances rather than modifying
    the original, ensuring immutability and enabling safe concurrent processing.
    """

    final_result: dict[str, Any]
    conversation_id: Optional[str]
    conversation_store: Optional["ConversationStore"]

    @property
    def chat(self) -> List[ChatMessage]:
        """
        Returns a deep copy of the chat history.

        The chat history consists of all messages exchanged during the conversation,
        including messages from users, assistants, and system messages.

        :return: A deep copy of the chat message list.
        """
        ...

    @property
    def system_prompt(self) -> Optional[ChatMessage]:
        """
        Returns the current system prompt.

        The system prompt provides context and instructions to the LLM about how to
        process and respond to the conversation. It's typically updated by each agent
        that processes the conversation.

        :return: The current system prompt if available, otherwise None.
        """
        ...

    @property
    def user_prompt(self) -> Optional[ChatMessage | list[ChatMessage]]:
        """
        Returns the current user prompt.

        The user prompt represents the immediate query or input that needs to be
        addressed by the LLM. It's typically updated by each agent that processes
        the conversation.

        :return: The current user prompt if available, otherwise None.
        """
        ...

    @property
    def prompt_argument(self) -> Optional[PromptArgument]:
        """
        Returns the current prompt argument.

        The prompt argument is a structured dataclass containing the information
        required to render templates and process the conversation. It serves as
        a shared data structure between agents.

        :return: The current prompt argument if available, otherwise None.
        """
        ...

    @property
    def metadata(self) -> Dict[str, str]:
        """
        Returns a deep copy of the conversation metadata.

        Metadata provides additional context about the conversation that isn't part
        of the core conversation flow, such as user identifiers, session information,
        or processing flags.

        :return: A deep copy of the metadata dictionary.
        """
        ...

    @property
    def last_message(self) -> ChatMessage:
        """
        Returns the last message in the chat history.

        This is a convenience property for accessing the most recent message, which
        is often needed to determine the current state of the conversation.

        :return: The last message in the chat history.
        :raises: IndexError if the chat history is empty.
        """
        ...

    @property
    def active_agent(self) -> str:
        """
        Returns the name of the agent currently processing the conversation.

        This property indicates which agent is currently responsible for handling
        the conversation, which is important for routing and processing decisions.

        :return: The name of the active agent.
        """
        ...

    @property
    def longterm_memory(self) -> LongtermMemory: ...  # unified global memory

    def update_longterm_memory(self, prompt_argument: PromptArgument) -> "Conversation":
        """
        Returns a new Conversation with global long-term memory updated from OutputFields
        of the given prompt argument.
        """
        ...

    def append(
        self,
        message: Union[ChatMessage, List[ChatMessage], str],
        role: Optional[Role] = None,
        name: Optional[str] = None,
    ) -> "Conversation":
        """
        Creates a new conversation with the specified message(s) appended.

        This method adds one or more new messages to the chat history and returns
        a new Conversation instance. If a string is provided instead of a ChatMessage,
        a new ChatMessage is created using the provided role and name.

        :param message: The message(s) to append (ChatMessage, list of ChatMessages, or string).
        :param role: The role of the message sender (used only when message is a string).
        :param name: The name of the message sender (used only when message is a string).
        :return: A new Conversation instance with the appended message(s).
        """
        ...

    def update(self, **kwargs) -> "Conversation":
        """
        Creates a new conversation with updated fields.

        This method creates a new Conversation instance with specified fields updated,
        while preserving immutability of the original instance.

        :param kwargs: Key-value pairs of fields to update (chat, prompt_argument, system_prompt,
                      user_prompt, longterm_memory, metadata, active_agent, conversation_id).
        :return: A new Conversation instance with the updated fields.
        """
        ...

    def render_chat(self, message_type: MessageType = MessageType.CONVERSATION) -> List[ChatMessage]:
        """
        Returns a rendered view of the complete chat history.

        This method assembles the full chat stream for sending to an LLM, including
        the system prompt, chat history, and user prompt. It can filter messages
        based on the specified message type.

        :param message_type: Filter for messages to include (default: CONVERSATION).
        :return: List of ChatMessage objects representing the complete rendered chat.
        """
        ...

    def is_empty(self) -> bool:
        """
        Checks if the conversation's chat history is empty.

        :return: True if the chat history contains no messages, False otherwise.
        """
        ...

    def has_pending_tool_call(self) -> bool:
        """
        Checks if there is a pending tool call in the conversation.

        This method determines whether the last message in the conversation contains
        a tool call that needs to be executed before the conversation can continue.

        :return: True if a tool call is pending, False otherwise.
        """
        ...

    def has_pending_tool_response(self) -> bool:
        """
        Checks if the conversation is awaiting a tool response.

        This method determines whether the last message was a tool call from the LLM
        that needs to be responded to before the conversation can continue.

        :return: True if awaiting a tool response, False otherwise.
        """
        ...

    async def maybe_persist(self) -> None:
        """
        Persists the conversation to the associated conversation store if available.

        This method safely saves the conversation state if a conversation store is
        configured, allowing for conversation history to be retrieved later.

        :return: None
        """
        ...

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        agents: list[Any],
        longterm_memory_class: Type[LongtermMemory],
        conversation_store: Optional["ConversationStore"] = None,
    ) -> "Conversation":
        """
        Creates a Conversation instance from a dictionary representation.

        This factory method reconstructs a conversation from a serialized dictionary
        representation, typically used when loading conversations from storage.

        :param data: Dictionary containing the serialized conversation.
        :param agents: List of agent instances needed for context reconstruction.
        :param longterm_memory_class: The class of the longterm memory..
        :param conversation_store: Optional store for persistence operations.
        :return: A new Conversation instance.
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Conversation instance to a dictionary representation.

        This method serializes the conversation into a dictionary structure that can
        be stored or transmitted, typically used when persisting conversations.

        :return: Dictionary representation of the Conversation.
        """
        ...


class LLMClient(Protocol):
    """
    Protocol for interacting with a Language Model (LLM) client.

    This protocol defines the interface for communicating with language model services.
    Implementations handle:
    - Initializing connections to LLM providers (OpenAI, Azure, local models, etc.)
    - Managing token counting and context limits
    - Generating responses to conversation prompts
    - Supporting tool usage within conversations
    - Handling error conditions and retries

    The protocol ensures consistent interaction with different LLM providers
    through a unified interface, allowing the system to work with any compatible
    language model service.
    """

    max_tokens: int

    @classmethod
    def create(cls, **kwargs) -> Self:
        """
        Creates a new instance of the LLM client.

        This factory method initializes a new LLM client with the provided configuration.
        It handles authentication, connection setup, and any other initialization needed
        to establish a working connection to the language model service.

        :param kwargs: Configuration parameters for the client, such as API keys,
                      endpoint URLs, model names, and other provider-specific settings.
        :return: A properly initialized instance of the LLM client.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a text string.

        This method uses provider-specific tokenization to determine how many tokens
        would be consumed by the given text. This is crucial for managing context
        windows and calculating costs for API calls.

        :param text: The text string to tokenize.
        :return: The number of tokens in the text string.
        """
        ...

    async def use_as_tool(self, prompt: str, content: str) -> str:
        """
        Uses the LLM as a tool for processing content.

        This method leverages the language model to process content according to
        a specific prompt, often for summarization, reformatting, or extraction tasks.
        It's particularly useful for handling large content that needs to be condensed
        to fit within token limits.

        :param prompt: Instructions for how to process the content.
        :param content: The content to be processed.
        :return: The processed content from the LLM.
        """
        ...

    async def generate(
        self,
        conversation: Conversation,
        tools: Optional[list[ToolDescription]] = None,
        message_type=MessageType.CONVERSATION,
    ) -> Conversation:
        """
        Generates a response from the language model based on conversation context.

        This core method sends the conversation history to the LLM and retrieves a response,
        which is then integrated into the conversation. When tools are provided, the LLM
        may choose to call them rather than generating a direct text response.

        The method handles:
        - Preparing the conversation history for the LLM
        - Managing token limits and context windows
        - Sending the request to the LLM provider
        - Processing the response (text or tool calls)
        - Updating the conversation with the result

        :param conversation: The current conversation state.
        :param tools: Optional list of tools the LLM can use in its response.
        :param message_type: Type of messages to include from the conversation history.
        :return: Updated conversation with the LLM's response.
        """
        ...


class ConversationParticipant(Protocol):
    """
    Protocol for conversation participants.

    This protocol defines the methods and properties required for an entity to participate in a conversation.
    Implementations of this protocol are responsible for processing conversations, registering dispatchers,
    and handling topics of interest. It ensures that any participant can actively engage in and manage
    conversations within the system.
    """

    name: str
    topics: list[str]

    async def process_conversation(self, conversation: Conversation) -> None:
        """
        Actively participate in a conversation.

        This method is responsible for processing the given conversation. The implementation should  handle
        the conversation appropriately, updating its state and generating responses as needed.
        This method is called by the conversation dispatcher.

        :param conversation: The conversation to be processed.
        """
        ...

    def register_dispatcher(self, dispatcher: "ConversationDispatcher") -> None:
        """
        Registers a dispatcher with the conversation participant.

        This method is responsible for associating a `ConversationDispatcher` with the
        conversation participant. The dispatcher will handle the distribution and management
        of conversations involving this participant.

        :param dispatcher: The `ConversationDispatcher` instance to be registered.
        """
        ...


class ConversationFinalizer(Protocol):
    """
    Protocol for finalizing conversations.

    This protocol defines the interface for components that generate the final output of a conversation.
    Finalizers are responsible for:
    - Extracting relevant information from the conversation state
    - Formatting this information into a coherent final response
    - Setting the final_result property on the conversation

    Finalizers typically run at the end of a conversation flow and prepare the conversation
    output for delivery to the end user or calling application.
    """

    name: str
    final_properties: list[str]

    async def finalize_conversation(self, conversation: Conversation) -> None:
        """
        Finalizes the conversation by setting the final_result property.

        This method extracts the relevant properties from the conversation's prompt_argument
        based on the final_properties list, and sets them as the final_result on the conversation.
        It represents the last step in the conversation processing pipeline.

        :param conversation: The conversation to be finalized.
        :return: None - The method modifies the conversation in-place by setting final_result.
        """
        ...


class ConversationResponder(Protocol):
    """
    Protocol for responding to a conversation request.

    This protocol defines the interface for components that respond directly to conversation requests.
    Responders are responsible for:
    - Processing incoming conversation requests
    - Generating appropriate responses based on the conversation context
    - Returning updated conversations with the response included

    Unlike conversation participants that may modify and publish conversations to other agents,
    responders focus on providing direct responses to specific requests, often used in synchronous
    communication patterns where an immediate response is expected.
    """

    name: str

    async def respond(self, conversation: Conversation) -> Conversation:
        """
        Responds to a conversation request.

        This method processes the given conversation and generates a direct response.
        It's designed for scenarios where a synchronous response is needed, such as
        when one agent needs information from another agent to proceed with processing.

        :param conversation: The conversation to process and respond to.
        :return: The updated conversation with the response included.
        """
        ...


class ConversationDispatcher(Protocol):
    """
    Protocol for  dispatching conversations.

    This protocol defines the methods required for subscribing participants to topics,
    publishing conversations to subscribed participants, and finalizing conversations.
    Implementations of this protocol are responsible for coordinating the flow of conversations
    between agents and participants, ensuring that conversations are properly managed and
    dispatched to the appropriate entities based on their topics of interest.
    """

    def subscribe(self, topic: str, participant: ConversationParticipant) -> None:
        """
        Subscribe a participant to a specific topic.

        This method registers a `ConversationParticipant` to receive conversations
        related to the specified topic. When a conversation is published to the topic,
        all subscribed participants will be notified and can process the conversation.

        :param topic: The topic to which the participant will be subscribed.
        :param participant: The participant to be subscribed to the topic.
        """
        ...

    def unsubscribe(self, topic: str, participant: ConversationParticipant) -> None:
        """
        Unsubscribe a participant from a specific topic.

        This method removes a `ConversationParticipant` from the list of subscribers for the given topic.
        Once unsubscribed, the participant will no longer receive conversations related to that topic.

        :param topic: The topic from which the participant will be unsubscribed.
        :param participant: The participant to be unsubscribed from the topic.
        """
        ...

    async def publish(self, topic: str, conversation: Conversation) -> None:
        """
        Dispatch a conversation to all participants subscribed to the topic.

        This method sends the given conversation to all participants who are subscribed
        to the specified topic. Each participant will receive the conversation and can
        process it accordingly.

        :param topic: The topic to which the conversation will be published.
        :param conversation: The conversation to be dispatched.
        """
        ...

    async def publish_final(self, topic: str, conversation: Conversation) -> None:
        """
        Dispatch a conversation to a finalizing agent

        This method sends the given conversation to a finalizing agent that will
        create the final answer to the conversation.

        :param topic: The name of the finalizing agent.
        :param conversation: The conversation to be dispatched.
        """
        ...

    async def request_response(self, topic: str, conversation: Conversation) -> Conversation:
        """
        Method for requesting a direct response from a participant.

        This is used if you need an answer to a question from a participant, e.g. to ask an agent whether we can
        finalize the conversation.

        :param topic: The `ConversationParticipant` to ask the question.
        :param conversation: The `Conversation` object representing the current state of the conversation.
        :return: The updated `Conversation` object with the response from the participant.
        """
        ...

    async def run(self, participant: ConversationParticipant, conversation: Conversation) -> Conversation:
        """
        Entry point for starting a conversation with a participant.

        This method starts a conversation by dispatching it to the participant, passed into it.
        It also handles the finalization of the conversation, as soon as the future object is set.
        returns a dictionary containing the final response data.

        :param participant: The `ConversationParticipant` that is involved in the conversation.
        :param conversation: The `Conversation` object representing the current state of the conversation.
        :return: A dictionary containing the final response data.
        """
        ...


class ConversationStore(Protocol):
    """
    Protocol for managing the persistence of conversations.

    This protocol defines the methods required for creating, persisting, fetching, deleting, and checking
    the existence of conversations.
    Implementations of this protocol are responsible for handling the storage and retrieval of conversation data,
    ensuring that conversations can be reliably saved and accessed as needed.
    """

    is_persistent: bool

    @classmethod
    def create(cls, **kwargs) -> "ConversationStore": ...

    async def persist(self, conversation: Conversation) -> None:
        """
        Persists the given conversation.

        This method is responsible for saving the state of the provided conversation
        to a persistent storage. Implementations of this method should ensure that
        the conversation data is reliably stored and can be retrieved later.

        :param conversation: The `Conversation` object representing the current state of the conversation.
        """
        ...

    async def fetch(self, conversation_id: str) -> Conversation:
        """
        Fetches a conversation by its unique identifier.

        This method retrieves the conversation associated with the given conversation ID from the persistent storage.
        It ensures that the conversation data is accurately fetched and returned as a `Conversation` object.

        :param conversation_id: The unique identifier of the conversation to be fetched.
        :return: The `Conversation` object representing the fetched conversation.
        """
        ...

    async def delete(self, conversation_id: str) -> None:
        """
        Deletes a conversation by its unique identifier.

        This method removes the conversation associated with the given conversation ID from the persistent storage.
        It ensures that the conversation data is permanently deleted and can no longer be retrieved.

        :param conversation_id: The unique identifier of the conversation to be deleted.
        :return: None
        """
        ...

    async def exists(self, conversation_id: str) -> bool:
        """
        Checks if a conversation with the given unique identifier exists in the persistent storage.

        This method is responsible for verifying the existence of a conversation by its unique ID.
        It returns a boolean value indicating whether the conversation is present in the storage.

        :param conversation_id: The unique identifier of the conversation to check.
        :return: True if the conversation exists, False otherwise.
        """
        ...


class Agent(Protocol):
    """
    The `Agent` protocol defines the core interface for agents within the Diskurs framework.

    Agents are central to the Diskurs system, responsible for processing conversations and
    generating responses based on the current state of the conversation. Implementations of
    this protocol must provide methods for creating agents and invoking their logic on
    conversations. Agents interact with various components such as prompts, long-term memory,
    and language model clients to facilitate meaningful and context-aware dialogues.

    Key Responsibilities:
    - Creating and initializing agents with specific configurations.
    - Processing conversations by invoking the agent's logic.
    - Interacting with prompts and long-term memory to maintain conversation context.
    - Generating responses using language model clients.

    This protocol ensures that any implementing class can seamlessly integrate into the
    Diskurs framework, enabling flexible and extensible conversation management.
    """

    name: str

    @classmethod
    def create(cls, name: str, prompt: Prompt, llm_client: LLMClient, **kwargs) -> "Agent": ...

    async def invoke(
        self,
        conversation: Union[Conversation, str],
        message_type: MessageType = MessageType.CONVERSATION,
        reset_prompt: bool = True,
    ) -> Conversation:
        """
        Run the agent on a conversation.

        This method processes the given conversation by invoking the agent's logic.
        It takes a `Conversation` object representing the conversation
        and returns an updated `Conversation` object after processing.

        :param conversation: The current state of the conversation, represented as a `Conversation` object
        :param message_type: The type of message to use when rendering templates and processing the conversation
        :param init_prompt: Whether to initialize the prompt before processing the conversation
        :return: An updated `Conversation` object with the processed state.
        """
        ...


@runtime_checkable
class ConductorAgent(Protocol):
    """
    The `ConductorAgent` protocol defines the interface for agents that manage and coordinate conversations.

    Conductor agents are responsible for orchestrating conversations by:
    - Creating and updating long-term memory to maintain conversation context
    - Routing conversations to appropriate specialized agents
    - Evaluating conversation state to determine when finalization is appropriate
    - Coordinating the finalization process when a conversation is complete

    This protocol ensures that implementing classes can effectively manage conversation flow,
    maintain state across multiple turns, and direct conversations to specialized agents based
    on content or user needs.
    """

    name: str
    prompt: ConductorPrompt
    can_finalize_name: Optional[str]
    finalizer_name: Optional[str]
    supervisor: Optional[str]

    def evaluate_rules(self, conversation: Conversation) -> Optional[str]:
        """
        Evaluates all routing rules against the current conversation.

        This method iterates through the configured routing rules and executes their
        condition functions against the provided conversation. It returns the target
        agent of the first rule that matches, or None if no rules match.

        Exceptions in rule evaluation are caught and logged to prevent rule failures
        from crashing the routing process.

        :param conversation: The conversation to evaluate against the rules.
        :return: The name of the target agent if a rule matches, None otherwise.
        """
        ...

    async def add_routing_message_to_chat(self, conversation: Conversation, next_agent: str) -> Conversation:
        """
        Adds a routing message to the conversation chat history.

        This method updates the prompt_argument with the next agent and adds a JSON
        message to the conversation history to ensure consistency. This creates a clear
        record of routing decisions in the conversation history.

        :param conversation: The conversation to update
        :param next_agent: The name of the next agent to route to
        :return: The updated conversation with routing information
        """
        ...

    async def invoke(self, conversation: Conversation, message_type=MessageType.CONDUCTOR) -> Conversation:
        """
        Processes the conversation and determines the next agent to route to.

        This method first attempts to find a routing destination using rule-based routing.
        If no rule matches and LLM fallback is enabled, it uses the LLM to determine routing.
        The selected next agent is then stored in the prompt_argument and a JSON
        representation is appended to the conversation history.

        The method follows this sequence:
        1. Prepare the conversation with appropriate prompt arguments
        2. Evaluate routing rules against the conversation
        3. If no rule matches and fallback is enabled, attempt LLM-based routing
        4. Update the prompt_argument with the chosen next_agent
        5. Append a structured JSON message to the conversation history

        :param conversation: The conversation to process
        :param message_type: The message type, defaults to MessageType.CONDUCTOR
        :return: The updated conversation with routing decision
        """
        ...

    async def can_finalize(self, conversation: Conversation) -> bool:
        """
        Determines if the conversation is ready to be finalized.

        This method checks if the conversation has reached a state where it can be
        considered complete. It either:
        1. Delegates the decision to another agent specified by can_finalize_name,
           which should return a prompt_argument with a can_finalize property, or
        2. Uses the prompt's own can_finalize method on the longterm memory

        If we are using a function i.e. heuristics, this method evaluates the long-term memory
        of the conversation to decide whether it can be finalized.

        If we are using an LLM i.e. agent based can_finalize, this method evaluates the chat history
        of the conversation to decide whether it can be finalized. This allows for free-form
        questions as observed in open chat.

        :param conversation: The conversation to check for finalization
        :return: True if the conversation can be finalized, False otherwise
        """
        ...

    async def finalize(self, conversation: Conversation) -> None:
        """
        Finalizes the conversation based on current routing configuration.

        This method handles the finalization process for a conversation based on the
        agent's configuration:

        1. If a supervisor is configured, routes the conversation to that agent
        2. If a finalizer_name is configured, publishes the conversation to that finalizer
        3. Otherwise, sets the final_result directly using the prompt's finalize method

        The method ensures that every conversation is properly concluded and that the
        appropriate finalization logic is applied.

        :param conversation: The conversation to finalize
        """
        ...

    def fail(self, conversation: Conversation) -> dict[str, Any]:
        """
        Handles conversation failure cases.

        This method is called when a conversation cannot be successfully completed,
        such as when maximum dispatch attempts are exceeded or other failure conditions
        are met. It generates an appropriate failure response.

        :param conversation: The conversation that has failed
        :return: A dictionary containing the failure response data
        """
        ...

    async def process_conversation(self, conversation: Conversation) -> None:
        """
        Main entry point for processing a conversation by the conductor agent.

        This method orchestrates the full conversation processing workflow:
        1. Updates longterm memory with conversation context
        2. Checks if the conversation can be finalized
        3. If ready for finalization, calls finalize()
        4. If max dispatches exceeded, calls fail()
        5. Otherwise, invokes routing logic and publishes to the next agent

        :param conversation: The conversation to be processed
        """
        ...


class ToolExecutor(Protocol):
    """
    The `ToolExecutor` protocol defines the interface for executing tools within the Diskurs framework.

    Tool executors are responsible for managing and executing tools that agents can use during
    conversation processing. They handle:
    - Tool registration and dependency management
    - Tool execution based on agent requests
    - Managing the lifecycle of tool dependencies

    This protocol ensures that any implementing class can provide a consistent interface for
    tool management and execution across the Diskurs framework.
    """

    tools: dict[str, Callable]
    dependencies: dict[str, "ToolDependency"]

    def register_tools(self, tools: List[Callable] | Callable) -> None:
        """
        Registers one or more tools with the executor.

        This method allows the registration of a single tool function or a list of tool functions
        that can be executed by the executor. Tool functions are registered with their function
        name as the key in the tools dictionary.

        :param tools: A single callable or a list of callables representing the tools to be registered.
        """
        ...

    def register_dependencies(self, dependencies: list["ToolDependency"]) -> None:
        """
        Registers tool dependencies with the executor.

        This method registers dependencies that tools may need for execution, such as database
        connections or external service clients. This is the preferred way to handle shared
        resources like DB connection pools, allowing them to be created once and shared across tools.

        :param dependencies: A list of ToolDependency objects to be registered.
        """
        ...

    async def execute_tool(self, tool_call: ToolCall, metadata: Dict[str, Any]) -> ToolCallResult:
        """
        Executes a tool based on a tool call specification.

        This method handles the execution of a tool specified by a ToolCall object. It looks up the
        appropriate tool function, prepares arguments, executes the tool, and wraps the result
        in a ToolCallResult object.

        :param tool_call: The ToolCall object specifying which tool to execute and with what arguments.
        :param metadata: Additional context information for tool execution.
        :return: A ToolCallResult object containing the execution result.
        """
        ...

    def call_tool(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Directly calls a registered tool with the specified arguments.

        This method provides a simplified interface for calling tools directly from code rather
        than through the LLM-based tool call mechanism. It's particularly useful in heuristic
        sequences where programmatic tool calls are needed.

        :param function_name: The name of the tool function to call.
        :param arguments: A dictionary of arguments to pass to the tool.
        :return: The result returned by the tool function.
        """
        ...


class ToolDependency(Protocol):
    """
    The `ToolDependency` protocol defines the interface for tool dependencies within the Diskurs framework.

    Tool dependencies provide a way to manage shared resources that tools may need for execution.
    Examples include:
    - Database connection pools
    - External API clients
    - Configuration managers
    - Cache systems

    This approach ensures efficient resource utilization by allowing multiple tools to share
    the same underlying resources rather than creating new connections for each tool call.
    It also provides lifecycle management for proper resource cleanup.
    """

    name: str

    @classmethod
    def create(cls, **kwargs) -> Self:
        """
        Creates a new instance of the tool dependency.

        This factory method is responsible for creating and properly initializing a new
        instance of the tool dependency with the provided configuration. It handles any
        setup tasks such as establishing connections, loading configurations, or initializing
        resources.

        :param kwargs: Configuration parameters for the dependency.
        :return: A properly initialized instance of the tool dependency.
        """
        ...

    def close(self) -> None:
        """
        Closes the tool dependency and releases any associated resources.

        This method ensures proper cleanup of resources when the tool dependency is no longer
        needed. This might include closing database connections, freeing memory, or releasing
        other system resources.

        :return: None
        """
        ...
