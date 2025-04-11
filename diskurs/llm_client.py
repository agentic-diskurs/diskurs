import json
from abc import abstractmethod
from typing import Any, Callable, Optional

import tiktoken
from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    UnprocessableEntityError,
)
from openai.types.chat import ChatCompletion
from typing_extensions import Self

from diskurs import Conversation
from diskurs.entities import ChatMessage, MessageType, Role, ToolCall, ToolDescription
from diskurs.logger_setup import get_logger
from diskurs.protocols import LLMClient
from diskurs.registry import register_llm
from diskurs.tools import map_python_type_to_json

TOOL_RESPONSE_MAX_FRACTION = 4


class BaseOaiApiLLMClient(LLMClient):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        tokenizer: Callable[[str], int] | tiktoken.Encoding,
        max_tokens: int,
        max_repeat: int = 3,
    ):
        """
        :param client: The OpenAI client instance used to interact
            with the OpenAI API.
        :param model: The model identifier string that specifies which
            version/model of the OpenAI API to use for generating responses.
        """
        self.client = client
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_repeat = max_repeat
        self.logger = get_logger(f"diskurs.llm.{self.__class__.__name__}")

        self.logger.info(f"LLM client '{self.__class__.__name__}' initialized.")

    @classmethod
    @abstractmethod
    def create(cls, **kwargs) -> Self:
        # Abstract create method to be implemented in subclasses
        pass

    async def send_request(
        self,
        body: dict[str, Any],
    ) -> ChatCompletion:
        self.logger.debug("Sending request to API.")
        completion = await self.client.chat.completions.create(**body)
        self.logger.debug("Received response from API.")
        return completion

    @staticmethod
    def format_tool_description_for_llm(tool: ToolDescription) -> dict[str, Any]:
        """
        Formats a ToolDescription object into a dictionary that can be sent to the LLM model.
        :param tool: Tool description to be formatted
        :return: JSON-serializable dictionary containing the tool data
        """
        properties = {
            arg_name: {
                "type": map_python_type_to_json(arg_data["type"]),
                "description": arg_data["description"],
            }
            for arg_name, arg_data in tool.arguments.items()
        }

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description.strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(tool.arguments.keys()),
                },
            },
        }

    @staticmethod
    def format_message_for_llm(message: ChatMessage) -> dict[str, str]:
        """
        Formats a ChatMessage object into a dictionary that can be sent to the LLM model.
        Used by the format_conversation_for_llm method to prepare individual messages for the LLM.

        :param message: Message to be formatted
        :return: JSON-serializable dictionary containing the message data
        """
        tool_calls = (
            {
                "tool_calls": [
                    {
                        "id": tool_call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function_name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            }
            if message.tool_calls
            else {}
        )
        tool_call_id = {"tool_call_id": message.tool_call_id} if message.tool_call_id else {}
        return {
            "role": str(message.role),
            "content": str(message.content),
            **tool_calls,
            **tool_call_id,
        }

    def count_tokens_tool_responses(self, user_prompt_tool_responses) -> tuple[int, list[tuple[ChatMessage, int]]]:
        if not isinstance(user_prompt_tool_responses, list):
            user_prompt_tool_responses = [user_prompt_tool_responses]

        tool_responses_tokens = 0
        tool_responses = []
        for msg in user_prompt_tool_responses:
            if msg.role == Role.TOOL:
                tool_tokens = self.count_tokens(str(msg.content))
                tool_responses.append((msg, tool_tokens))
                tool_responses_tokens += tool_tokens

        return tool_responses_tokens, tool_responses

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        # Helper to truncate text so that its token count is at most max_tokens.
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def water_filling_truncate_responses(
        self, tool_responses_with_counts, total_allowed_tokens, truncation_message: str = "\n[Response truncated]"
    ):
        """
        Given a list of tuples (message, token_count), return a new list of messages where the
        token counts have been reduced using a water-filling / iterative thresholding approach
        to meet a total_allowed_tokens budget.

        This function doesn't directly alter the text. It assumes that you have a function
        like _truncate_text(content, new_token_count) that returns a version of the content
        limited to new_token_count tokens.

        :param tool_responses_with_counts: List of tuples (ChatMessage, token_count)
        :param total_allowed_tokens: The total number of tokens allowed after reduction
        :param truncation_message: Message to append to truncated responses
        :return: List of truncated ChatMessages
        """
        truncated_responses_token_count = self.count_tokens(truncation_message)
        min_message_tokens = 5  # Minimum tokens to keep a message meaningful

        # Sort responses by token count (largest first)
        sorted_responses = sorted(tool_responses_with_counts, key=lambda x: x[1], reverse=True)

        total_tokens = sum(token_count for _, token_count in sorted_responses)
        excess = total_tokens - total_allowed_tokens

        # If within limit, no need to truncate.
        if excess <= 0:
            return [msg for msg, _ in sorted_responses]

        # Special case: protect very small messages from truncation
        original_tokens = [token_count for _, token_count in sorted_responses]
        very_small_indices = [i for i, count in enumerate(original_tokens) if count <= min_message_tokens]
        very_small_tokens = sum(original_tokens[i] for i in very_small_indices)

        # For minimal truncation scenario, consider preserving the smallest message as well
        if excess < total_tokens * 0.1:  # If we need to truncate less than 10%
            if len(sorted_responses) > 1 and not very_small_indices:
                # Add the smallest message to the protected list if it's not already there
                smallest_idx = len(sorted_responses) - 1
                very_small_indices.append(smallest_idx)
                very_small_tokens += original_tokens[smallest_idx]

        # Adjust the total allowed tokens to account for preserving very small messages
        remaining_allowed = total_allowed_tokens - very_small_tokens if very_small_tokens > 0 else total_allowed_tokens

        # If remaining allowed tokens is negative, we have to truncate something
        if remaining_allowed <= 0:
            # In this extreme case, preserve just the smallest message
            if very_small_indices:
                smallest_idx = max(very_small_indices)  # Get the smallest message index
                preserved_tokens = original_tokens[smallest_idx]
                remaining_allowed = total_allowed_tokens - preserved_tokens
                # Remove all other small indices except the smallest one
                very_small_indices = [smallest_idx]
            else:
                remaining_allowed = total_allowed_tokens

        # Create a list to hold the new token allocations
        # Initially, each response is allocated its original token count
        new_token_alloc = list(original_tokens)

        # Remove small messages from the waterfall calculation
        remaining_indices = [i for i in range(len(sorted_responses)) if i not in very_small_indices]
        if not remaining_indices:
            # All messages are very small, can't do waterfall algorithm
            return [msg for msg, _ in sorted_responses]

        remaining_excess = excess - (total_tokens - very_small_tokens - remaining_allowed)

        # If remaining excess is negative, we've already solved the problem by preserving small messages
        if remaining_excess <= 0:
            truncated_messages = []
            for idx, (message, original_token_count) in enumerate(sorted_responses):
                if idx in very_small_indices:
                    # Keep small messages as they are
                    truncated_messages.append(message)
                else:
                    # Truncate larger messages
                    new_limit = max(1, remaining_allowed // len(remaining_indices))
                    truncated_text = self._truncate_text(
                        str(message.content), new_limit - truncated_responses_token_count
                    )
                    content = f"{truncated_text}{truncation_message}"
                    truncated_message = ChatMessage(
                        role=message.role,
                        content=content,
                        type=message.type,
                        name=message.name,
                        tool_call_id=message.tool_call_id,
                    )
                    truncated_messages.append(truncated_message)
            return truncated_messages

        # Apply waterfall algorithm only to non-small messages
        i = 0  # Start with the largest message
        while remaining_excess > 0 and i < len(remaining_indices):
            group_start_idx = i
            current_idx = remaining_indices[i]
            current_count = new_token_alloc[current_idx]

            # Find group of equal-sized messages
            while i < len(remaining_indices) and new_token_alloc[remaining_indices[i]] == current_count:
                i += 1

            group_end_idx = i  # The group is [group_start_idx, group_end_idx)
            group_indices = remaining_indices[group_start_idx:group_end_idx]
            group_size = len(group_indices)

            # Determine the next lower level among remaining responses
            if i < len(remaining_indices):
                next_idx = remaining_indices[i]
                next_level = new_token_alloc[next_idx]
            else:
                # Don't go below minimum meaningful token count
                next_level = min_message_tokens + truncated_responses_token_count

            # Maximum tokens that can be removed without reducing below next level
            potential_removal_per_message = current_count - next_level
            total_possible_removal = potential_removal_per_message * group_size

            if total_possible_removal <= remaining_excess:
                # Remove the full potential amount from each message in the group
                for idx in group_indices:
                    new_token_alloc[idx] -= potential_removal_per_message
                remaining_excess -= total_possible_removal
            else:
                # Not enough excess to remove down to next_level
                # Remove an equal share from each message
                removal_each = remaining_excess // group_size
                remainder = remaining_excess % group_size

                for j, idx in enumerate(group_indices):
                    new_token_alloc[idx] -= removal_each
                    if j < remainder:  # Distribute remainder evenly
                        new_token_alloc[idx] -= 1

                remaining_excess = 0  # Budget exhausted

        # Create truncated messages with the new token allocations
        truncated_messages = []
        for idx, (message, original_token_count) in enumerate(sorted_responses):
            original_content = str(message.content)

            if idx in very_small_indices:
                # Keep small messages as they are
                truncated_messages.append(message)
                continue

            new_limit = new_token_alloc[idx]
            was_truncated = new_limit < original_token_count

            # Ensure we have enough tokens for meaningful content plus truncation message
            if was_truncated:
                if new_limit <= truncated_responses_token_count + 1:
                    # If allocation is too small, just keep a minimal part of the message
                    truncated_text = self._truncate_text(original_content, min_message_tokens)
                else:
                    truncated_text = self._truncate_text(original_content, new_limit - truncated_responses_token_count)
                content = f"{truncated_text}{truncation_message}"
            else:
                content = original_content

            truncated_message = ChatMessage(
                role=message.role,
                content=content,
                type=message.type,
                name=message.name,
                tool_call_id=message.tool_call_id,
            )
            truncated_messages.append(truncated_message)

        return truncated_messages

    def truncate_tool_responses(
        self, tool_responses: ChatMessage | list[ChatMessage], fraction: int = 2
    ) -> list[ChatMessage]:
        """
        Truncates tool responses to fit within the maximum token limit.
        We first obtain the token count for each tool response sorted by size.
        Then we truncate the largest tool responses until we fit within the limit.
        We intelligently estimate the number of tokens that can be removed in each turn.
        :param tool_responses: The tool responses to truncate
        :param fraction: The fraction the tool response should be reduced by in relation to the max tokens
        """
        _, tool_responses_with_counts = self.count_tokens_tool_responses(user_prompt_tool_responses=tool_responses)
        total_allowed_tokens = self.max_tokens // fraction
        return self.water_filling_truncate_responses(
            tool_responses_with_counts=tool_responses_with_counts, total_allowed_tokens=total_allowed_tokens
        )

    def should_truncate_tool_response(self, tool_responses: ChatMessage | list[ChatMessage], fraction=4) -> bool:
        """
        Determine if we should attempt to truncate a tool response as a first strategy.
        Only applies when over token limits and the tool responses contain significant tokens.

        :param tool_responses: The tool response messages to check
        :param fraction: The fraction of max tokens to use as a threshold for truncation
        :return: True if the tool responses contain enough tokens to make truncation worthwhile
        """
        if self.is_tool_response(tool_responses):
            responses = tool_responses if isinstance(tool_responses, list) else [tool_responses]
            total_tokens = sum(self.count_tokens(str(msg.content)) for msg in responses if msg.role == Role.TOOL)

            return total_tokens > (self.max_tokens / fraction)

        return False

    @staticmethod
    def is_tool_response(user_prompt: ChatMessage | list[ChatMessage]) -> bool:
        """
        Check if the conversation contains tool responses.
        """
        if isinstance(user_prompt, list):
            return any(msg.role == Role.TOOL for msg in user_prompt)
        elif isinstance(user_prompt, ChatMessage):
            return user_prompt.role == Role.TOOL
        else:
            raise ValueError("Invalid user prompt type. Expected list or ChatMessage.")

    def format_messages_for_llm(self, conversation, message_type):
        messages = []
        for message in conversation.render_chat(message_type=message_type):
            if isinstance(message, list):
                # If we executed multiple tools in a single pass, we have to flatten the list
                # containing the tool call responses
                for m in message:
                    messages.append(self.format_message_for_llm(m))
            else:
                messages.append(self.format_message_for_llm(message))
        return messages

    def format_conversation_for_llm(
        self,
        conversation: Conversation,
        tools: Optional[list[ToolDescription]] = None,
        message_type=MessageType.CONVERSATION,
    ) -> dict[str, Any]:
        """
        Formats the conversation object into a dictionary that can be sent to the LLM model.
        This comprises the user prompt, chat history, and tool descriptions.
        :param conversation: Contains all interactions so far
        :param tools: The descriptions of all tools that the agent can use
        :param message_type: The message type used to filter the chat history. If MessageType.CONDUCTOR,
          all messages will be rendered
        :return: A JSON-serializable dictionary containing the conversation data ready for the LLM
        """
        self.logger.debug(f"Formatting conversation for LLM")

        formatted_tools = {"tools": [self.format_tool_description_for_llm(tool) for tool in tools]} if tools else {}

        messages = self.format_messages_for_llm(conversation, message_type)

        n_tokens_tool_descriptions = (
            self.count_tokens_of_tool_descriptions(formatted_tools["tools"]) if formatted_tools else 0
        )

        tokens_in_conversation = self.count_tokens_in_conversation(messages)

        if (tokens_in_conversation + n_tokens_tool_descriptions) > self.max_tokens:
            if self.should_truncate_tool_response(conversation.user_prompt):
                truncated_tool_responses = self.truncate_tool_responses(tool_responses=conversation.user_prompt)
                conversation = conversation.update(user_prompt=truncated_tool_responses)
                messages = self.format_messages_for_llm(conversation=conversation, message_type=message_type)

            messages = self.truncate_chat_history(messages, n_tokens_tool_descriptions)

        return {
            **formatted_tools,
            "model": self.model,
            "messages": messages,
        }

    @classmethod
    def is_tool_call(cls, completion: ChatCompletion) -> bool:
        return completion.choices[0].finish_reason == "tool_calls"

    @classmethod
    def llm_response_to_chat_message(
        cls, completion: ChatCompletion, agent_name: str, message_type: MessageType
    ) -> ChatMessage:
        """
        Converts the message returned by the LLM to a typed ChatMessage.
        :param completion: The response from the LLM model
        :param agent_name: The name of the agent whose question the completion is a response to
        :param message_type: The type of message to be created
        :return: A ChatMessage object containing the structured response
        """
        if cls.is_tool_call(completion):
            tool_calls = [
                ToolCall(
                    tool_call_id=tool_call.id,
                    function_name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                for tool_call in completion.choices[0].message.tool_calls
            ]
            return ChatMessage(
                role=Role.ASSISTANT,
                tool_calls=tool_calls,
                type=message_type,
                name=agent_name,
            )
        else:
            return ChatMessage(
                role=Role(completion.choices[0].message.role),
                content=completion.choices[0].message.content,
                type=message_type,
                name=agent_name,
            )

    @classmethod
    def concatenate_user_prompt_with_llm_response(
        cls, conversation: Conversation, completion: ChatCompletion
    ) -> list[ChatMessage]:
        """
        Creates a list of ChatMessages that combines the user prompt with the LLM response.
        Ensures a flat list, even if there are multiple messages in the user prompt (as is the case when
        multiple tools are executed in a single pass).

        :param conversation: the conversation containing the user prompt
        :param completion: the response from the LLM model
        :return: Flat list of ChatMessages containing the user prompt and LLM response
        """
        user_prompt = (
            conversation.user_prompt if isinstance(conversation.user_prompt, list) else [conversation.user_prompt]
        )
        agent_name, message_type = next(((m.name, m.type) for m in user_prompt))
        return user_prompt + [
            cls.llm_response_to_chat_message(completion=completion, agent_name=agent_name, message_type=message_type)
        ]

    def count_tokens_in_conversation(self, messages: list[dict]) -> int:
        """
        Count the number of tokens used by a list of messages i.e. chat history.
        The implementation is based on OpenAI's token counting guidelines.
        """
        self.logger.debug(f"Counting tokens in conversation")

        if self.model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4",
            "gpt-4o",  # verify for 4-0
        }:
            tokens_per_message = 3
            tokens_per_name = 1
            tokens_per_function_call = 3  # tokens per function call
        elif self.model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {self.model}.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == "name":
                    num_tokens += tokens_per_name + self.count_tokens(value)
                elif key == "content":
                    num_tokens += self.count_tokens(value)
                elif key == "function_call" or key == "tool_calls":
                    num_tokens += tokens_per_function_call
                    num_tokens += self.count_tokens_recursively(value)
                else:
                    num_tokens += self.count_tokens_recursively(value)
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        self.logger.debug(f"Counted {num_tokens} tokens in conversation")
        return num_tokens

    def count_tokens_recursively(self, value):
        if isinstance(value, str):
            return self.count_tokens(value)
        elif isinstance(value, dict):
            total = 0
            for k, v in value.items():
                total += self.count_tokens_recursively(k)
                total += self.count_tokens_recursively(v)
            return total
        elif isinstance(value, list):
            total = 0
            for item in value:
                total += self.count_tokens_recursively(item)
            return total
        else:
            return self.count_tokens(str(value))

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a text string.
        :param text: The text string to tokenize.
        :return: The number of tokens in the text string.
        """
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

    def count_tokens_of_tool_descriptions(self, tool_descriptions: list[dict[str, Any]]) -> int:
        """
        Return the number of tokens used by the tool i.e. function description.
        Unfortunately, there's no documented way of counting those tokens, therefore we resort to best effort approach,
        hoping this implementation is a true upper bound.
        The implementation is taken from:
        https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11

        :param tool_descriptions: The description of all the tools
        :return: The number of tokens used by the tools
        """

        num_tokens = 0
        for tool_description in tool_descriptions:
            function = tool_description["function"]
            function_tokens = self.count_tokens(function["name"])
            function_tokens += self.count_tokens(function["description"])

            if "parameters" in function:
                parameters = function["parameters"]
                if "properties" in parameters:
                    for propertiesKey in parameters["properties"]:
                        function_tokens += self.count_tokens(propertiesKey)
                        v = parameters["properties"][propertiesKey]
                        for field in v:
                            if field == "type":
                                function_tokens += 2
                                function_tokens += self.count_tokens(v["type"])
                            elif field == "description":
                                function_tokens += 2
                                function_tokens += self.count_tokens(v["description"])
                            elif field == "enum":
                                function_tokens -= 3
                                for o in v["enum"]:
                                    function_tokens += 3
                                    function_tokens += self.count_tokens(o)
                            else:
                                print(f"Warning: not supported field {field}")
                    function_tokens += 11

            num_tokens += function_tokens

        num_tokens += 12

        self.logger.debug(f"Counted {num_tokens} tokens for tool descriptions")
        return num_tokens

    def truncate_chat_history(self, messages, n_tokens_tool_descriptions) -> list[dict]:
        """
        Truncate the chat history to fit within the maximum token limit while preserving
        context and essential messages.
        """
        self.logger.warning(f"Max tokens exceeded, truncating chat history")

        # Always keep system message and last message
        system_message = messages[0]
        last_message = messages[-1]

        # Calculate available tokens for middle messages
        tokens_for_edges = self.count_tokens_in_conversation([system_message, last_message])
        available_tokens = self.max_tokens - tokens_for_edges - n_tokens_tool_descriptions

        if available_tokens <= 0:
            self.logger.warning("Not enough tokens for chat history, keeping only system and last message")
            return [system_message, last_message]

        # Try to keep as many recent messages as possible
        middle_messages = messages[1:-1]
        truncated_messages = []

        # Work backwards from most recent
        for message in reversed(middle_messages):
            message_tokens = self.count_tokens_in_conversation([message])
            if available_tokens - message_tokens > 0:
                truncated_messages.insert(0, message)
                available_tokens -= message_tokens
            else:
                break

        return [system_message] + truncated_messages + [last_message]

    async def generate(
        self,
        conversation: Conversation,
        tools: Optional[ToolDescription] = None,
        message_type=MessageType.CONVERSATION,
    ) -> Conversation:
        """
        Generates a response from the LLM model for the given conversation.
        Handles conversion from Conversation to LLM request format, sending the request to the LLM model,
        and converting the response back to a Conversation object.

        :param conversation: The conversation object containing the user prompt and chat history.
        :param tools: Description of all the tools that the agent can use
        :param message_type: The message type used to filter the chat history. If MessageType.CONDUCTOR,
          all messages will be rendered
        :return: Updated conversation object with the LLM response appended to the chat history.
        """
        request_body = self.format_conversation_for_llm(
            conversation=conversation, tools=tools, message_type=message_type
        )
        fail_counter = 0

        while fail_counter < self.max_repeat:
            try:
                completion = await self.send_request(request_body)
                return conversation.append(self.concatenate_user_prompt_with_llm_response(conversation, completion))

            except (
                UnprocessableEntityError,
                AuthenticationError,
                PermissionError,
                BadRequestError,
            ) as e:
                self.logger.error(f"Non-retryable error: {e}, aborting...")
                raise e

            except (
                APITimeoutError,
                APIError,
                RateLimitError,
            ) as e:
                fail_counter += 1
                self.logger.warning(
                    f"Retryable error encountered: {e}, retrying... ({fail_counter}/{self.max_repeat})"
                )

        raise RuntimeError("Failed to generate response after multiple attempts.")

    async def use_as_tool(self, prompt: str, content: str) -> str:
        """
        Summarizes content to fit within token limit.

        :param prompt: Prompt to use for summarization
        :param content: Content to summarize
        :param fraction: Fraction of the max tokens to use for summarization
        :return: Summarized content
        """
        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {"role": "user", "content": content},
            ],
        }
        completion = await self.send_request(request_body)
        return completion.choices[0].message.content


@register_llm(name="openai")
class OpenAILLMClient(BaseOaiApiLLMClient):
    @classmethod
    def create(cls, **kwargs) -> Self:
        api_key = kwargs.get("api_key", None)
        model = kwargs.get("model_name", "")
        max_tokens = kwargs.get("model_max_tokens", 2048)

        tokenizer = tiktoken.encoding_for_model(model)

        client = AsyncOpenAI()
        client.api_key = api_key

        return cls(client=client, model=model, tokenizer=tokenizer, max_tokens=max_tokens)
