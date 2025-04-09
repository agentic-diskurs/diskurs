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
        tokenizer: Callable[[str], int],
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

    def count_tokens_tool_responses(
        self, total_tokens, user_prompt_tool_responses
    ) -> tuple[int, list[tuple[ChatMessage, int]]]:
        # Ensure user_prompt_tool_responses is iterable
        if not isinstance(user_prompt_tool_responses, list):
            user_prompt_tool_responses = [user_prompt_tool_responses]

        tool_responses_tokens = 0
        tool_responses = []
        for msg in user_prompt_tool_responses:
            if msg.role == Role.TOOL:
                tool_tokens = self.count_tokens(str(msg.content))
                tool_responses.append((msg, tool_tokens))
                tool_responses_tokens += tool_tokens
        tool_responses.sort(key=lambda x: x[1], reverse=True)

        return tool_responses_tokens, tool_responses

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        # Helper to truncate text so that its token count is at most max_tokens.
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

    def truncate_tool_responses(
        self, user_prompt_tool_responses: list[ChatMessage], total_tokens: int
    ) -> list[ChatMessage]:
        """
        Truncate all tool responses such that the combined tokens of the responses and total_tokens
        is less than self.max_tokens.
        """
        available = self.max_tokens - total_tokens
        marker = "[Content truncated due to token limits...]"
        marker_tokens = self.count_tokens(marker)
        min_content_tokens = 10  # Minimum tokens of original content to preserve

        # Add a small buffer to ensure we stay under the limit, even with encoding differences
        buffer_tokens = 1
        effective_available = max(0, available - buffer_tokens)

        # If no available tokens, replace every response with the marker
        if effective_available <= 0:
            for msg in user_prompt_tool_responses:
                msg.content = marker
            return user_prompt_tool_responses

        # Calculate total tokens in tool responses.
        total_tool_tokens = sum(self.count_tokens(msg.content) for msg in user_prompt_tool_responses)

        # If current responses already fit into available tokens, return them unchanged.
        if total_tool_tokens <= effective_available:
            return user_prompt_tool_responses

        # Proportionally compute allowed tokens per message.
        for msg in user_prompt_tool_responses:
            orig_tokens = self.count_tokens(msg.content)
            # Allocate tokens proportionally ensuring room for marker if truncation is needed.
            new_max = int(orig_tokens * effective_available / total_tool_tokens)

            # Ensure we have enough space for minimum content + marker
            min_required = min_content_tokens + marker_tokens
            if new_max < min_required:
                new_max = min(min_required, effective_available)

            # If message exceeds the allowed token count, truncate and append marker.
            if orig_tokens > new_max:
                # Reserve space for the marker but ensure at least min_content_tokens
                effective_max = max(min_content_tokens, new_max - marker_tokens)
                truncated = self._truncate_text(msg.content, effective_max)
                msg.content = truncated + marker

        # Final check to ensure we're under the limit
        final_tokens = sum(self.count_tokens(msg.content) for msg in user_prompt_tool_responses)
        if final_tokens > effective_available:
            # If still over limit, do another pass with more aggressive truncation
            over_by = final_tokens - effective_available
            for msg in user_prompt_tool_responses:
                if over_by <= 0:
                    break
                current_tokens = self.count_tokens(msg.content)
                if current_tokens > marker_tokens + min_content_tokens:
                    # Calculate how much to reduce this message by
                    to_reduce = min(over_by + 1, current_tokens - marker_tokens - min_content_tokens)
                    if to_reduce > 0:
                        new_content_tokens = current_tokens - to_reduce - marker_tokens
                        if new_content_tokens >= min_content_tokens:
                            truncated = self._truncate_text(msg.content, new_content_tokens)
                            msg.content = truncated + marker
                            over_by -= to_reduce

        return user_prompt_tool_responses

    @staticmethod
    def is_last_msg_tool_response(conversation: Conversation) -> bool:
        """
        Check if the conversation contains tool responses.
        """
        if isinstance(conversation.user_prompt, list):
            return any(msg.role == Role.TOOL for msg in conversation.user_prompt)
        return False

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
            n_tokens_tool_response, _ = self.count_tokens_tool_responses(self.max_tokens, conversation.user_prompt)
            if (
                self.is_last_msg_tool_response(conversation)
                and n_tokens_tool_response > self.max_tokens // TOOL_RESPONSE_MAX_FRACTION
            ):
                truncated_tool_responses = self.truncate_tool_responses(
                    user_prompt_tool_responses=conversation.user_prompt,
                    total_tokens=tokens_in_conversation + n_tokens_tool_descriptions,
                )
                # Rebuild messages by replacing tool messages with truncated versions
                new_conversation = conversation.update(user_prompt=truncated_tool_responses)
                messages = self.format_messages_for_llm(conversation=new_conversation, message_type=message_type)
            else:
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
