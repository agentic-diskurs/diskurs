from unittest.mock import patch, MagicMock
import pytest
from diskurs.azure_llm_client import AzureOpenAIClient


@patch("diskurs.azure_llm_client.DefaultAzureCredential")
@patch("diskurs.azure_llm_client.get_bearer_token_provider")
def test_azure_openai_client_creation_with_entra_id(mock_get_token, mock_credential):
    mock_credential.return_value = MagicMock()
    mock_get_token.return_value = MagicMock(return_value="mock_token")

    client = AzureOpenAIClient.create(
        api_key=None,
        model_name="gpt-4",
        api_version="2023-03-15-preview",
        endpoint="https://mock-azure-openai-endpoint.com",
        use_entra_id=True,
    )

    mock_credential.assert_called_once()
    mock_get_token.assert_called_once_with(mock_credential(), "https://cognitiveservices.azure.com/.default")
    assert client.client

@patch("diskurs.azure_llm_client.get_bearer_token_provider")
def test_azure_openai_client_creation_without_entra_id(mock_credential):
    mock_credential.return_value = MagicMock()

    client = AzureOpenAIClient.create(
        api_key="mock_api_key",
        model_name="gpt-4",
        api_version="2023-03-15-preview",
        endpoint="https://mock-azure-openai-endpoint.com",
        use_entra_id=False,
    )

    mock_credential.assert_not_called()

    assert client.client
    assert client.client.api_key == "mock_api_key"
    assert client
