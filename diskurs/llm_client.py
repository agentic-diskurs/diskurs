import json
import logging
from abc import abstractmethod
from typing import Any, Optional, Callable

import tiktoken
from typing_extensions import Self

from openai import APIError, APITimeoutError, RateLimitError, UnprocessableEntityError, AuthenticationError
from openai import OpenAI, BadRequestError
from openai.types.chat import ChatCompletion

from diskurs.entities import Conversation, ChatMessage, Role, ToolCall, ToolDescription, MessageType
from diskurs.protocols import LLMClient
from diskurs.registry import register_llm
from diskurs.tools import map_python_type_to_json

logger = logging.getLogger(__name__)


class BaseOaiApiLLMClient(LLMClient):
    def __init__(
            self, client: OpenAI, model: str, tokenizer: Callable[[str], int], max_tokens: int, max_repeat: int = 3
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

    @classmethod
    @abstractmethod
    def create(cls, **kwargs) -> Self:
        # Abstract create method to be implemented in subclasses
        pass

    def send_request(
            self,
            body: dict[str, Any],
    ) -> ChatCompletion:
        completion = self.client.chat.completions.create(**body)
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

    def format_conversation_for_llm(
            self, conversation: Conversation, tools: Optional[list[ToolDescription]] = None
    ) -> dict[str, Any]:
        """
        Formats the conversation object into a dictionary that can be sent to the LLM model.
        This comprises the user prompt, chat history, and tool descriptions.
        :param conversation: Contains all interactions so far
        :param tools: The descriptions of all tools that the agent can use
        :return: A JSON-serializable dictionary containing the conversation data ready for the LLM
        """

        formatted_tools = {"tools": [self.format_tool_description_for_llm(tool) for tool in tools]} if tools else {}

        messages = []
        for message in conversation.render_chat():
            if isinstance(message, list):
                # If we executed multiple tools in a single pass, we have to flatten the list
                # containing the tool call responses
                for m in message:
                    messages.append(self.format_message_for_llm(m))
            else:
                messages.append(self.format_message_for_llm(message))

        n_tokens_tool_descriptions = self.count_tokens_of_tool_descriptions(
            formatted_tools["tools"]) if formatted_tools else 0

        if (self.count_tokens_in_conversation(messages) + n_tokens_tool_descriptions) > self.max_tokens:
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
    def llm_response_to_chat_message(cls, completion: ChatCompletion, message_type: MessageType) -> ChatMessage:
        """
        Converts the message returned by the LLM to a typed ChatMessage.
        :param completion: The response from the LLM model
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
            return ChatMessage(role=Role.ASSISTANT, tool_calls=tool_calls, type=message_type)
        else:
            return ChatMessage(
                role=Role(completion.choices[0].message.role),
                content=completion.choices[0].message.content,
                type=message_type,
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
        message_type = next((m.type for m in user_prompt))
        return user_prompt + [cls.llm_response_to_chat_message(completion, message_type=message_type)]

    def count_tokens_in_conversation(self, messages: list[dict]) -> int:
        """
        Count the number of tokens used by a list of messages i.e. chat history.
        The implementation is based on OpenAI's token counting guidelines.
        """
        if self.model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4o"  # verify for 4-0
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
        return num_tokens

    def truncate_chat_history(self, messages, n_tokens_tool_descriptions) -> list[dict]:
        """
        Truncate the chat history to fit within the maximum token limit. The token limit is calculated as follows:
        We retain the first two messages i.e. system prompt and initial user prompt and the last message.
        We then truncate from left, removing messages from the chat history until the total token count is within the
        limit. We also account for the token count of the tool descriptions.

        :param messages: The list of messages in the conversation
        :param n_tokens_tool_descriptions: The number of tokens used by the tool descriptions
        :return: The truncated chat history
        """
        chat_start = messages[:2]
        user_prompt = messages[-1]

        max_tokens = (
                self.max_tokens
                - self.count_tokens_in_conversation(chat_start + [user_prompt])
                - n_tokens_tool_descriptions
        )

        truncated_chat = messages[2:-1]

        for _ in range(len(truncated_chat)):
            truncated_chat = truncated_chat[1:]

            if self.count_tokens_in_conversation(truncated_chat) > max_tokens:
                break

        return chat_start + truncated_chat + [user_prompt]

    def generate(self, conversation: Conversation, tools: Optional[ToolDescription] = None) -> Conversation:
        """
        Generates a response from the LLM model for the given conversation.
        Handles conversion from Conversation to LLM request format, sending the request to the LLM model,
        and converting the response back to a Conversation object.

        :param conversation: The conversation object containing the user prompt and chat history.
        :param tools: Description of all the tools that the agent can use
        :return: Updated conversation object with the LLM response appended to the chat history.
        """
        request_body = self.format_conversation_for_llm(conversation, tools)
        fail_counter = 0

        while fail_counter < self.max_repeat:
            try:
                completion = self.send_request(request_body)
                return conversation.append(self.concatenate_user_prompt_with_llm_response(conversation, completion))

            except (
                    UnprocessableEntityError,
                    AuthenticationError,
                    PermissionError,
                    BadRequestError,
            ) as e:
                logger.error(f"Non-retryable error: {e}, aborting...")
                raise e

            except (
                    APITimeoutError,
                    APIError,
                    RateLimitError,
            ) as e:
                fail_counter += 1
                logger.warning(f"Retryable error encountered: {e}, retrying... ({fail_counter}/{self.max_repeat})")

        raise RuntimeError("Failed to generate response after multiple attempts.")


@register_llm(name="openai")
class OpenAILLMClient(BaseOaiApiLLMClient):
    @classmethod
    def create(cls, **kwargs) -> Self:
        api_key = kwargs.get("api_key", None)
        model = kwargs.get("model_name", "")
        max_tokens = kwargs.get("model_max_tokens", 2048)

        tokenizer = tiktoken.encoding_for_model(model)

        client = OpenAI()
        client.api_key = api_key

        return cls(client=client, model=model, tokenizer=tokenizer, max_tokens=max_tokens)
