import json
import logging
import os
from abc import abstractmethod
from typing import Any, Optional
from typing_extensions import Self

from openai import APIError, APITimeoutError, RateLimitError, UnprocessableEntityError, AuthenticationError
from openai import OpenAI, BadRequestError, AzureOpenAI
from openai.types.chat import ChatCompletion

from entities import Conversation, ChatMessage, Role, ToolCall, ToolDescription, MessageType
from protocols import LLMClient
from registry import register_llm
from tools import map_python_type_to_json

logger = logging.getLogger(__name__)


class BaseOaiApiLLMClient(LLMClient):
    def __init__(self, client: OpenAI, model: str, max_repeat: int = 3):
        """
        :param client: The OpenAI client instance used to interact
            with the OpenAI API.
        :param model: The model identifier string that specifies which
            version/model of the OpenAI API to use for generating responses.
        """
        self.client = client
        self.model = model
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
            "content": message.content,
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
        model = kwargs.get("model", "")

        client = OpenAI()
        client.api_key = api_key

        return cls(client=client, model=model)


@register_llm(name="azure")
class AzureOpenAIClient(BaseOaiApiLLMClient):
    @classmethod
    def create(cls, **kwargs) -> Self:
        api_key = kwargs.get("api_key", None)
        model = kwargs.get("model_name", "")
        api_version = kwargs.get("api_version", "")
        azure_endpoint = kwargs.get("endpoint", "")

        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        return cls(client=client, model=model, max_repeat=3)
