import os

import tiktoken
from typing_extensions import Self

from openai import AzureOpenAI

from diskurs.llm_client import BaseOaiApiLLMClient
from diskurs.registry import register_llm

try:
    from azure.identity import get_bearer_token_provider, DefaultAzureCredential
except ImportError:
    DefaultAzureCredential = None
    get_bearer_token_provider = None


@register_llm(name="azure")
class AzureOpenAIClient(BaseOaiApiLLMClient):
    @classmethod
    def create(cls, **kwargs) -> Self:
        api_key = kwargs.get("api_key", None)
        model = kwargs.get("model_name", "")
        api_version = kwargs.get("api_version", "")
        azure_endpoint = kwargs.get("endpoint", "")
        use_entra_id = kwargs.get("use_entra_id", False)
        max_tokens = kwargs.get("max_tokens")

        tokenizer = (tiktoken.encoding_for_model(model),)

        client_params = {
            "api_key": api_key,
            "api_version": api_version,
            "azure_endpoint": azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
        }

        if use_entra_id:
            client_params["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), "https://api.cognitive.microsoft.com/.default"
            )

        client = AzureOpenAI(**client_params)

        return cls(client=client, model=model, tokenizer=tokenizer, max_tokens=max_tokens, max_repeat=3)
