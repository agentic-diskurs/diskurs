from dataclasses import dataclass
from typing import (
    List,
    Dict,
    Self,
    Callable,
    Protocol,
    Any,
    Type,
    TypeVar,
    Optional,
    Union,
    runtime_checkable,
)

from diskurs.entities import (
    ToolDescription,
    LongtermMemory,
    ToolCallResult,
    ToolCall,
    PromptArgument,
    ChatMessage,
    MessageType,
    Role,
)


class LongtermMemoryHandler(Protocol):
    def can_finalize(self, longterm_memory: Any) -> bool:
        pass


class PromptValidator(Protocol):
    """
    Protocol for validating prompt responses.

    This protocol defines methods for validating responses from a language model (LLM).
    It includes methods for validating responses as dataclasses and JSON objects.
    """

    @classmethod
    def validate_dataclass(
        cls,
        parsed_response: dict[str, Any],
        user_prompt_argument: Type[dataclass],
        strict: bool = False,
    ) -> dataclass:
        """
        Validates a parsed response dictionary against a dataclass type.

        :param parsed_response: The dictionary containing the parsed response from the LLM.
        :param user_prompt_argument: The dataclass type to validate against.
        :param strict: If True, enforce strict validation rules.

        :return: An instance of the dataclass populated with the validated data.
        """
        pass

    @classmethod
    def validate_json(cls, llm_response: str) -> dict:
        """
        Validates a JSON response from a language model (LLM).

        This method takes a JSON string response from an LLM and validates it,
        ensuring it conforms to the expected structure and content.

        :param llm_response: The JSON string response from the LLM.
        :return: A dictionary representation of the validated JSON response.
        """
        pass


UserPromptArg = TypeVar("UserPromptArg", bound=PromptArgument)
SystemPromptArg = TypeVar("SystemPromptArg", bound=PromptArgument)


class Prompt(Protocol):
    """
    Protocol for prompt implementations.

    This protocol defines the structure and methods required for creating and handling prompts
    in a conversation. Implementations of this protocol are responsible for generating prompt
    arguments, rendering templates, and parsing responses from a language model (LLM).
    """

    user_prompt_argument: Type[UserPromptArg]

    def create_user_prompt_argument(self, **prompt_args: Any) -> UserPromptArg:
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

    def parse_user_prompt(
        self,
        name: str,
        llm_response: str,
        old_user_prompt_argument: PromptArgument,
        message_type: MessageType = MessageType.CONDUCTOR,
    ) -> Union[PromptArgument, ChatMessage]:
        """
        Parses the LLM response into a prompt argument or ChatMessage.

        This method takes the response from a language model (LLM) and parses it into either a
        `PromptArgument` or a `ChatMessage` object. It uses the old user prompt argument and
        the message type to guide the parsing process.

        :param name: Name of the agent.
        :param llm_response: The response string from the language model.
        :param old_user_prompt_argument: The previous user prompt argument to be used as a reference.
        :param message_type: The type of message to be parsed. Defaults to `MessageType.ROUTING`.
        :return: A `PromptArgument` or `ChatMessage` object based on the parsed response.
        """
        ...

    def is_final(self, user_prompt_argument: PromptArgument) -> bool:
        """
        Determines if the user prompt argument indicates the final state.

        This method checks the provided user prompt argument to determine if it represents
        the final state in the conversation. It is used to decide whether the conversation
        can be concluded based on the user's input.

        :param user_prompt_argument: The user prompt argument to be evaluated.
        :return: True if the user prompt argument indicates the final state, False otherwise.
        """
        ...

    def is_valid(self, user_prompt_argument: PromptArgument) -> bool:
        """
        Validates the user prompt argument.

        This method checks if the provided user prompt argument meets the required
        criteria for validity. It ensures that the user prompt argument is correctly
        structured and contains the necessary information for further processing.

        :param user_prompt_argument: The user prompt argument to be validated.
        :return: True if the user prompt argument is valid, False otherwise.
        """
        ...


class MultistepPrompt(Prompt):
    system_prompt_argument: Type[SystemPromptArg]

    def create_system_prompt_argument(self, **prompt_args: Any) -> SystemPromptArg:
        """
        Creates an instance of the system prompt argument dataclass.

        This method is responsible for generating the system prompt argument
        based on the provided keyword arguments. The system prompt argument
        is used to configure the initial state and context for the conversation.

        :param prompt_args: Keyword arguments used to initialize the system prompt argument.
        :return: An instance of the system prompt argument dataclass.
        """
        ...

    def render_system_template(self, name: str, prompt_args: PromptArgument, return_json: bool = True) -> ChatMessage:
        """
        Renders the system template with the provided prompt arguments.

        This method is responsible for rendering the system template using the given prompt arguments.
        It can optionally return the rendered template as a JSON object.

        :param name: The name of the template to be rendered.
        :param prompt_args: The prompt arguments to be used for rendering the template.
        :param return_json: If True, the rendered template will be returned as a JSON object. Defaults to True.
        :return: A ChatMessage object containing the rendered template.
        """
        ...


class ConductorPrompt(Prompt):
    """
    Protocol for conductor prompts.

    This protocol defines the methods required for handling conductor prompts in a conversation.
    It includes methods for initializing, finalizing, and validating long-term memory, as well as
    determining the final state of the conversation.
    """

    system_prompt_argument: Type[SystemPromptArg]
    longterm_memory: Type[LongtermMemory]

    def create_system_prompt_argument(self, **prompt_args: Any) -> SystemPromptArg:
        """
        Creates an instance of the system prompt argument dataclass.

        This method is responsible for generating the system prompt argument
        based on the provided keyword arguments. The system prompt argument
        is used to configure the initial state and context for the conversation.

        :param prompt_args: Keyword arguments used to initialize the system prompt argument.
        :return: An instance of the system prompt argument dataclass.
        """
        ...

    def render_system_template(self, name: str, prompt_args: PromptArgument, return_json: bool = True) -> ChatMessage:
        """
        Renders the system template with the provided prompt arguments.

        This method is responsible for rendering the system template using the given prompt arguments.
        It can optionally return the rendered template as a JSON object.

        :param name: The name of the template to be rendered.
        :param prompt_args: The prompt arguments to be used for rendering the template.
        :param return_json: If True, the rendered template will be returned as a JSON object. Defaults to True.
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

    async def finalize(self, longterm_memory: LongtermMemory) -> dict[str, Any]:
        """
        Finalizes the conversation based on the long-term memory.

        This method processes the provided long-term memory to generate a final response
        for the conversation. It is responsible for concluding the conversation by
        utilizing the accumulated long-term memory data.

        :param longterm_memory: The long-term memory to be used for finalizing the conversation.
        :return: A dictionary containing the final response data.
        """
        ...

    def fail(self, longterm_memory: LongtermMemory) -> dict[str, Any]:
        """
        Handles the case failure i.e. when the long-term memory does not meet the criteria defined
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
    async def __call__(self, conversation: "Conversation", call_tool: Optional[CallTool]) -> "Conversation": ...


class HeuristicPrompt(Protocol):
    """
    Protocol for heuristic prompts.

    This protocol defines the structure and methods required for creating and handling heuristic prompts
    in a conversation. Implementations of this protocol are responsible for generating user prompt arguments,
    rendering user templates, and executing heuristic sequences to process conversations.
    """

    user_prompt_argument: Type[PromptArgument]

    async def heuristic_sequence(self, conversation: "Conversation", call_tool: CallTool) -> "Conversation":
        """
        Executes a heuristic sequence on the given conversation.

        This method processes the conversation using a series of heuristic steps,
        optionally utilizing a tool for specific operations. The heuristic sequence
        is designed to guide the conversation towards a desired outcome based on
        predefined rules and logic.

        :param conversation: The conversation object to be processed.
        :param call_tool: An optional callable tool that can be used during the heuristic sequence.
        :return: The updated conversation object after processing the heuristic sequence.
        """
        ...

    def create_user_prompt_argument(self, **prompt_args) -> UserPromptArg:
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
        prompt_args: UserPromptArg,
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


class Conversation(Protocol[SystemPromptArg, UserPromptArg]):
    """
    Protocol for conversation management.

    This protocol defines the structure and methods required for handling conversations
    between agents and users. It includes properties and methods for accessing and updating
    chat messages, prompt arguments, long-term memory, and metadata. Implementations of this
    protocol are responsible for managing the state and flow of conversations, ensuring
    immutability, and providing mechanisms for rendering and parsing prompts.
    """

    final_result: dict[str, Any]

    @property
    def chat(self) -> List[ChatMessage]:
        """
        Provides a deep copy of the chat messages to ensure immutability.

        This method returns a list of `ChatMessage` objects representing the conversation's chat history.
        The returned list is a deep copy, ensuring that the original chat messages remain unchanged.

        :return: A deep copy of the list of chat messages.
        :rtype: List[ChatMessage]
        """
        ...

    @property
    def system_prompt(self) -> Optional[ChatMessage]:
        """
        Retrieves the system prompt of the conversation.

        The system prompt is a message that sets the initial context or instructions for the conversation.
        It is typically used to guide the conversation's flow and provide necessary background information.
        Each time an agent takes a turn in the conversation, it updates the system prompt accordingly.

        :return: The system prompt message if available, otherwise None.
        """
        ...

    @property
    def user_prompt(self) -> Optional[ChatMessage]:
        """
        Retrieves the user prompt of the conversation.

        The user prompt is a message that represents the user's input or query in the conversation.
        It is typically used to direct the agent's next response.
        Each time an agent takes a turn in the conversation, it updates the user prompt accordingly.

        :return: The user prompt message if available, otherwise None.
        """
        ...

    @property
    def system_prompt_argument(self) -> Optional[SystemPromptArg]:
        """
        Retrieves the system prompt arguments.

        The system prompt argument is used to render the system prompt message.
        Each time an agent takes a turn in the conversation, it updates the system prompt argument.

        :return: The system prompt arguments if available, otherwise None.
        """
        ...

    @property
    def user_prompt_argument(self) -> Optional[UserPromptArg]:
        """
        Retrieves the user prompt arguments.

        The user prompt argument is used to render the user prompt message.
        Each time an agent takes a turn in the conversation, it updates the user prompt argument.

        :return: The user prompt arguments if available, otherwise None.
        """
        ...

    @property
    def metadata(self) -> Dict[str, str]:
        """
        Provides a deep copy of the metadata dictionary to ensure immutability.

        The metadata dictionary contains additional information about the conversation,
        and can be freely defined and updated by the implementing system.

        :return: A deep copy of the metadata dictionary.
        """
        ...

    @property
    def last_message(self) -> ChatMessage:
        """
        Retrieves the last message in the conversation.

        This property returns the most recent `ChatMessage` object from the conversation's chat history.
        It is used to access the latest message exchanged in the conversation, which can be useful for
        determining the current state or context of the conversation.

        :return: The last message in the chat.
        :rtype: ChatMessage
        """
        ...

    @property
    def active_agent(self) -> str:
        """
        Retrieves the name of the active agent.

        The active agent is the agent currently responsible for processing the conversation.
        This method returns the name of the active agent as a string.

        :return: The name of the active agent.
        :rtype: str
        """
        ...

    @property
    def conversation_id(self) -> str:
        """
        Retrieves the unique identifier for the conversation.

        This method returns the conversation ID, which is a unique string
        used to identify the conversation instance. The conversation ID
        is typically used for tracking and managing conversations within
        the system.

        :return: The unique identifier for the conversation.
        """
        ...

    def append(
        self,
        message: ChatMessage | list[ChatMessage],
        role: Optional[Role] = "",
        name: Optional[str] = "",
    ) -> "Conversation":
        """
        Appends a new chat message and returns a new instance of Conversation.

        :param message: The ChatMessage object to be added to the conversation, alternatively a string can be provided.
        :param role: Only needed if message is str, the role (system, user, assistant)
        :param name: Only needed if message is str, name of the agent
        :return: A new instance of Conversation with the appended message.
        """

    def get_agent_longterm_memory(self, agent_name: str) -> Optional[LongtermMemory]:
        """
        Provides a deep copy of the long-term memory for the specified agent.

        This method retrieves the long-term memory associated with a given agent name.
        Allowing the caller to safely modify the returned memory without affecting the original.

        :param agent_name: The name of the agent whose long-term memory is to be retrieved.
        :return: A deep copy of the agent's long-term memory, or None if no memory is found.
        """
        ...

    def update_agent_longterm_memory(self, agent_name: str, longterm_memory: LongtermMemory) -> "Conversation":
        """
        Updates the long-term memory for a specific agent.

        This method updates the long-term memory associated with the given agent name.
        It returns a new instance of the Conversation with the updated long-term memory,
        ensuring immutability of the conversation state.

        :param agent_name: The name of the agent whose long-term memory is to be updated.
        :param longterm_memory: The new long-term memory for the agent.
        :return: A new instance of the Conversation with updated long-term memory.
        """
        ...

    def update_prompt_argument_with_longterm_memory(self, conductor_name: str) -> "Conversation":
        """
        Updates the prompt arguments with the long-term memory of the conductor agent.

        This method retrieves the long-term memory associated with the specified conductor agent
        and updates the prompt arguments of the conversation accordingly. It ensures that the
        conversation's state is updated with the relevant long-term memory data, preserving
        immutability by returning a new instance of the Conversation.

        :param conductor_name: The name of the conductor agent whose long-term memory is to be used.
        :return: A new instance of the Conversation with updated prompt arguments.
        """
        ...

    def update_prompt_argument_with_previous_agent(self, prompt_argument: UserPromptArg) -> "Conversation":
        """
        Updates the provided prompt arguments with the previous agent's prompt argument.

        This method uses the prompt argument of the last agent from the conversation
        and updates the provided prompt argument with the relevant data. It ensures that the
        conversation's state is updated with the necessary information, preserving immutability

        :param prompt_argument: The prompt argument to be updated.
        :return: A new instance of the Conversation with updated prompt arguments.
        """
        ...

    def update(
        self,
        chat: Optional[List[ChatMessage]] = None,
        system_prompt_argument: Optional[SystemPromptArg] = None,
        user_prompt_argument: Optional[UserPromptArg] = None,
        system_prompt: Optional[ChatMessage] = None,
        user_prompt: Optional[Union[ChatMessage, List[ChatMessage]]] = None,
        longterm_memory: Optional[Dict[str, LongtermMemory]] = None,
        metadata: Optional[Dict[str, str]] = None,
        active_agent: Optional[str] = None,
        conversation_id=None,
    ) -> "Conversation":
        """
        Returns a new instance of Conversation with updated fields, preserving immutability.

        :param chat: Optional list of ChatMessage objects representing the conversation's chat history.
        :param system_prompt_argument: Optional system prompt argument to update.
        :param user_prompt_argument: Optional user prompt argument to update.
        :param system_prompt: Optional system prompt message to update.
        :param user_prompt: Optional user prompt message(s) to update.
        :param longterm_memory: Optional dictionary of long-term memory to update.
        :param metadata: Optional dictionary of metadata to update.
        :param active_agent: Optional name of the active agent to update.
        :return: A new instance of the Conversation class with updated fields.
        """
        ...

    def append(
        self,
        message: Union[ChatMessage, List[ChatMessage], str],
        role: Optional[str] = "",
        name: Optional[str] = "",
    ) -> "Conversation":
        """
        Appends a new chat message and returns a new instance of Conversation.

        :param message: The message to be added.
        :param role: The role of the message sender.
        :param name: The name of the message sender.
        :return: A new instance of Conversation with the appended message.
        """
        ...

    def render_chat(self, message_type: MessageType = MessageType.CONVERSATION) -> List[ChatMessage]:
        """
        Returns the complete chat with the system prompt prepended and the user prompt appended.

        This method generates a list of `ChatMessage` objects representing the full chat history.
        It includes the system prompt at the beginning and the user prompt at the end of the chat.

        :param message_type: The type of message to be rendered. Defaults to `MessageType.CONVERSATION`.
        :return: A list representing the full chat.
        """
        ...

    def is_empty(self) -> bool:
        """
        Checks if the chat is empty.

        :return: True if the chat is empty, False otherwise.
        """
        ...

    def has_pending_tool_call(self) -> bool:
        """
        Checks if there is a pending tool call in the conversation.

        :return: True if there is a pending tool call, False otherwise.
        """
        ...

    def has_pending_tool_response(self) -> bool:
        """
        Checks if there is a pending tool response in the conversation,
        this would be the case if the last message was a tool call from
        the LLM.

        :return: True if there is a pending tool response, False otherwise.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any], agents: list[Any]) -> "Conversation":
        """
        Creates a Conversation instance from a dictionary.

        :param data: The data dictionary.
        :param agents: A list of agent instances.
        :return: A new instance of Conversation.
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Conversation instance to a dictionary.

        :return: A dictionary representation of the Conversation.
        """
        ...


class LLMClient(Protocol):
    """
    Protocol for interacting with a Language Model (LLM) client.
    This protocol defines the methods required for creating and generating responses
    from a language model within a conversation. Implementations of this protocol
    are responsible for initializing the LLM client and generating responses based
    on the current state of the conversation and optional tools.
    """

    @classmethod
    def create(cls, **kwargs) -> Self: ...

    async def generate(
        self, conversation: Conversation, tools: Optional[list[ToolDescription]] = None
    ) -> Conversation:
        """
        Generates a response from the language model (LLM) based on the current state of the conversation.

        This method takes the current conversation and optionally a list of tools, and generates a response
        from the language model. The response is integrated into the conversation, updating its state.

        :param conversation: The current state of the conversation, represented as a `Conversation` object.
        :param tools: An optional list of `ToolDescription` objects that can be used by the LLM to generate the response.
        :return: An updated `Conversation` object with the generated response.
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

    This protocol defines the methods and properties required for an entity to finalize a conversation.
    Implementations of this protocol are responsible for generating the final response to a conversation.

    :param final_properties: Specify the fields from the prompt argument that should be included in the final response.
    """

    name: str
    final_properties: list[str]

    async def finalize_conversation(self, conversation: Conversation) -> None:
        """
        Finalizes the conversation by setting the final response property on the conversation.
        This method is meant to be the last method called in a conversation.

        :param conversation: The conversation to be finalized.
        :return: A dictionary containing the final response data.
        """
        ...


class ConversationResponder(Protocol):
    """
    Protocol for responding to a conversation request.

    This protocol defines the methods required for responding to a conversation request.
    Implementations of this protocol are responsible for processing the request, generating a response,
    and returning the response to the requester.
    """

    name: str

    async def respond(self, conversation: Conversation) -> Conversation:
        """
        Responds to a conversation request.

        This method processes the given request and generates a response to be sent back to the requester.
        It is responsible for handling the request and generating the appropriate response data.

        :param conversation: The request data to be processed.
        :return: A dictionary containing the response data.
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
        pass

    def unsubscribe(self, topic: str, participant: ConversationParticipant) -> None:
        """
        Unsubscribe a participant from a specific topic.

        This method removes a `ConversationParticipant` from the list of subscribers for the given topic.
        Once unsubscribed, the participant will no longer receive conversations related to that topic.

        :param topic: The topic from which the participant will be unsubscribed.
        :param participant: The participant to be unsubscribed from the topic.
        """
        pass

    async def publish(self, topic: str, conversation: Conversation) -> None:
        """
        Dispatch a conversation to all participants subscribed to the topic.

        This method sends the given conversation to all participants who are subscribed
        to the specified topic. Each participant will receive the conversation and can
        process it accordingly.

        :param topic: The topic to which the conversation will be published.
        :param conversation: The conversation to be dispatched.
        """
        pass

    async def publish_final(self, topic: str, conversation: Conversation) -> None:
        """
        Dispatch a conversation to a finalizing agent

        This method sends the given conversation to a finalizing agent that will
        create the final answer to the conversation.

        :param topic: The name of the finalizing agent.
        :param conversation: The conversation to be dispatched.
        """
        pass

    async def request_response(self, topic: str, conversation: Conversation) -> Conversation:
        """
        Method for requesting a direct response from a participant.

        This is used if you need an answer to a question from a participant, e.g. to ask an agent whether we can
        finalize the conversation.

        :param topic: The `ConversationParticipant` to ask the question.
        :param conversation: The `Conversation` object representing the current state of the conversation.
        :return: The updated `Conversation` object with the response from the participant.
        """
        pass

    async def run(self, participant: ConversationParticipant, conversation: Conversation) -> dict:
        """
        Entry point for starting a conversation with a participant.

        This method starts a conversation by dispatching it to the participant, passed into it.
        It also handles the finalization of the conversation, as soon as the future object is set.
        returns a dictionary containing the final response data.

        :param participant: The `ConversationParticipant` that is involved in the conversation.
        :param conversation: The `Conversation` object representing the current state of the conversation.
        :return: A dictionary containing the final response data.
        """
        pass


class ConversationStore(Protocol):
    """
    Protocol for managing the persistence of conversations.

    This protocol defines the methods required for creating, persisting, fetching, deleting, and checking
    the existence of conversations.
    Implementations of this protocol are responsible for handling the storage and retrieval of conversation data,
    ensuring that conversations can be reliably saved and accessed as needed.
    """

    @classmethod
    def create(cls, **kwargs) -> "ConversationStore": ...

    def persist(self, conversation: Conversation) -> None:
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
    def create(cls, name: str, prompt: Prompt, llm_client: LLMClient, **kwargs): ...

    async def invoke(self, conversation: Conversation) -> Conversation:
        """
        Run the agent on a conversation.

        This method processes the given conversation by invoking the agent's logic.
        It takes a `Conversation` object representing the conversation
        and returns an updated `Conversation` object after processing.

        :param conversation: The current state of the conversation, represented as a `Conversation` object
        :return: An updated `Conversation` object with the processed state.
        """
        ...


@runtime_checkable
class ConductorAgent(Protocol):
    """
    The `ConductorAgent` protocol defines the interface for agents that manage and coordinate conversations.
    Conductor agents are responsible for creating or updating long-term memory, invoking conversation logic,
    and ensuring the conversation progresses smoothly. They are responsible for dispatching conversation to the
    most appropriate agents to maintain context and generate responses. This protocol ensures that any
    implementing class can effectively manage and direct conversations.
    """

    name: str
    prompt: ConductorPrompt
    can_finalize_name: Optional[str]

    def create_or_update_longterm_memory(self, conversation: Conversation, overwrite: bool = False) -> Conversation:
        """
        Creates or updates the long-term memory for the conductor agent.

        This method is responsible for either creating a new long-term memory instance
        or updating an existing one based on the provided conversation. It ensures that
        the long-term memory is synchronized with the current state of the conversation.

        :param conversation: The current state of the conversation, represented as a `Conversation` object.
        :param overwrite: A boolean flag indicating whether to overwrite existing memory fields. Defaults to False.
        :return: An updated `Conversation` object with the new or updated long-term memory.
        """
        ...

    async def can_finalize(self, conversation: Conversation) -> bool:
        """
        Determines if the conversation can be finalized based on the long-term memory i.e. the conversation history
        (in the case of an LLM based can_finalize)

        If we are using a function i.e. heuristics, this method evaluates the long-term memory of the conversation
        to decide whether it can be finalized. It checks if the necessary conditions are met for finalizing the conversation.

        If we are using an LLM i.e. agent based can_finalize, this method evaluates the chat history of the conversation
        to decide whether it can be finalized. This allows for free-form questions as observed in open chat.

        :param conversation: The current state of the conversation, represented as a `Conversation` object.
        :return: True if the conversation can be finalized, False otherwise.
        """
        ...

    async def finalize(self, conversation: Conversation) -> None:
        """
        Finalizes the conversation based on the long-term memory.
        Finalizing refers to the process of generating the final answer to be returned by diskurs.
        The function can work in two ways:
            1. If a "finalizer_name" has been provided, the function will call an agent with that name, to finalize.
            2. If no "finalizer_name" has been provided, the function will call the "finalize" function on the prompt.

        :param conversation: The current state of the conversation, represented as a `Conversation` object.
        """
        ...


class ToolExecutor(Protocol):
    """
    The `ToolExecutor` protocol defines the interface for executing tools within the Diskurs framework.
    Implementations of this protocol are responsible for registering tools, executing them based on tool calls,
    and providing a mechanism to directly call specific tool functions. This protocol ensures that any implementing
    class can manage and execute tools effectively, facilitating the integration of various tools into the conversation
    processing workflow.
    """

    tools: dict[str, Callable]
    dependencies: dict[str, "ToolDependency"]

    def register_tools(self, tool_list: List[Callable] | Callable) -> None:
        """
        Registers one or more tools with the executor.

        This method allows the registration of a single tool or a list of tools
        that can be executed by the executor. Each tool is a callable that can
        be invoked with specific arguments.

        :param tool_list: A single callable or a list of callables representing the tools to be registered.
        """
        ...

    def register_dependencies(self, dependencies: list["ToolDependency"]) -> None:
        """
        Registers tool dependencies with the executor.

        This method allows the registration of tool dependencies that are required for the execution of tools.
        This is the prefered way to handle things like DB connection pools, as it allows for central creation of
        a single connection pool that can be shared across multiple tools.

        :param dependencies: A list of ToolDependency objects representing the dependencies to be registered.
        """
        ...

    async def execute_tool(self, tool_call: ToolCall, metadata: Dict[str, Any]) -> ToolCallResult:
        """
        Executes a registered tool based on the provided tool call and metadata.

        This method is responsible for invoking a specific tool using the details
        provided in the `tool_call` object. It utilizes the metadata to provide
        additional context or parameters required for the tool execution. The result
        of the tool execution is returned as a `ToolCallResult` object.

        :param tool_call: The `ToolCall` object containing the details of the tool to be executed.
        :param metadata: A dictionary containing additional context or parameters for the tool execution.
        :return: A `ToolCallResult` object containing the result of the tool execution.
        """
        ...

    def call_tool(self, function_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Calls a registered tool with the specified function name and arguments.

        This method is responsible for invoking a tool that has been registered with the executor.
        It uses the provided function name and arguments to execute the tool and returns the result.
        It is meant to be used by a developer to directly call a specific tool function withing a
        heuristic sequence.

        :param function_name: The name of the tool function to be called.
        :param arguments: A dictionary containing the arguments to be passed to the tool function.
        :return: The result of the tool function execution.
        """
        ...


class ToolDependency(Protocol):
    """
    The `ToolDependency` protocol defines the interface for tool dependencies within the Diskurs framework.
    Implementations of this protocol are responsible for managing the lifecycle of tool dependencies,
    such as database connections or external services. This protocol ensures that any implementing class
    can effectively handle tool dependencies and provide a mechanism for registering and accessing them.
    """

    name: str

    def create(self, **kwargs) -> Self:
        """
        Creates a new instance of the tool dependency.

        This method is responsible for creating a new instance of the tool dependency
        based on the provided keyword arguments. It ensures that the dependency is
        properly initialized and ready for use by the tool executor.

        :param kwargs: Additional keyword arguments used to configure the tool dependency.
        :return: An instance of the tool dependency.
        """
        ...

    def close(self) -> None:
        """
        Closes the tool dependency and releases any associated resources.

        This method is responsible for closing the tool dependency and releasing any
        resources that are associated with it. It ensures that the dependency is properly
        cleaned up and ready for disposal.

        :return: None
        """
        ...
