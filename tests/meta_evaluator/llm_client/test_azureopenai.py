"""Test suite for AzureOpenAIConfig class.

This module contains comprehensive tests for the AzureOpenAIConfig class,
covering instantiation, validation, serialization, and edge cases.
"""

from typing import Any
import pytest
from pydantic import ValidationError

from meta_evaluator.llm_client.azureopenai_client import AzureOpenAIConfig


class TestAzureOpenAIConfig:
    """Test suite for the AzureOpenAIConfig class.

    This class tests the Azure OpenAI configuration implementation,
    including required field validation, default values, and integration
    with the base LLMClientConfig class.
    """

    @pytest.fixture
    def valid_azure_config_data(self) -> dict[str, Any]:
        """Provide valid configuration data for AzureOpenAIConfig.

        Returns:
            dict[str, Any]: A dictionary containing all required fields
                with valid values for creating an AzureOpenAIConfig instance.
        """
        return {
            "api_key": "test-api-key",
            "endpoint": "https://test.openai.azure.com/",
            "api_version": "2023-12-01-preview",
            "default_model": "gpt-4",
            "default_embedding_model": "text-embedding-ada-002",
        }

    @pytest.fixture
    def valid_azure_config(
        self, valid_azure_config_data: dict[str, Any]
    ) -> AzureOpenAIConfig:
        """Provide a valid AzureOpenAIConfig instance.

        Args:
            valid_azure_config_data: A dictionary containing valid configuration data.

        Returns:
            AzureOpenAIConfig: A valid AzureOpenAIConfig instance for testing.
        """
        return AzureOpenAIConfig(**valid_azure_config_data)

    def test_happy_path_instantiation(
        self, valid_azure_config_data: dict[str, Any]
    ) -> None:
        """Test successful creation with all required fields provided.

        Verifies that an AzureOpenAIConfig instance can be created successfully
        when all required fields are provided with valid values.

        Args:
            valid_azure_config_data: A dictionary containing valid configuration data.
        """
        config = AzureOpenAIConfig(**valid_azure_config_data)

        assert config.api_key == "test-api-key"
        assert config.endpoint == "https://test.openai.azure.com/"
        assert config.api_version == "2023-12-01-preview"
        assert config.default_model == "gpt-4"
        assert config.default_embedding_model == "text-embedding-ada-002"
        assert config.supports_structured_output is True
        assert config.supports_logprobs is True

    def test_default_boolean_values(
        self, valid_azure_config_data: dict[str, Any]
    ) -> None:
        """Test that Azure-specific capabilities have correct default values.

        Verifies that the supports_structured_output and supports_logprobs
        fields are set to their expected default values for Azure OpenAI.

        Args:
            valid_azure_config_data: A dictionary containing valid configuration data.
        """
        config = AzureOpenAIConfig(**valid_azure_config_data)

        assert config.supports_structured_output is True
        assert config.supports_logprobs is True

    def test_prevent_instantiation_method_exists(
        self, valid_azure_config: AzureOpenAIConfig
    ) -> None:
        """Test that _prevent_instantiation method can be called without error.

        Verifies that the _prevent_instantiation method required by the abstract
        base class is implemented and can be called without raising exceptions.

        Args:
            valid_azure_config: A valid AzureOpenAIConfig instance.
        """
        # Should not raise an exception
        valid_azure_config._prevent_instantiation()

    def test_missing_endpoint_raises_validation_error(self) -> None:
        """Test that missing endpoint field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required endpoint field results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                api_key="test-key",
                api_version="2023-12-01-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_missing_api_version_raises_validation_error(self) -> None:
        """Test that missing api_version field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required api_version field results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_missing_api_key_raises_validation_error(self) -> None:
        """Test that missing api_key field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required api_key field (inherited from base class)
        results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                endpoint="https://test.openai.azure.com/",
                api_version="2023-12-01-preview",
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_missing_default_model_raises_validation_error(self) -> None:
        """Test that missing default_model field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required default_model field (inherited from base class)
        results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                api_version="2023-12-01-preview",
                default_embedding_model="text-embedding-ada-002",
            )

    def test_missing_default_embedding_model_raises_validation_error(self) -> None:
        """Test that missing default_embedding_model field raises ValidationError.

        Verifies that attempting to create an AzureOpenAIConfig instance
        without providing the required default_embedding_model field
        (inherited from base class) results in a ValidationError.
        """
        with pytest.raises(ValidationError):
            AzureOpenAIConfig(  # type: ignore
                api_key="test-key",
                endpoint="https://test.openai.azure.com/",
                api_version="2023-12-01-preview",
                default_model="gpt-4",
            )

    def test_empty_strings_are_allowed(self) -> None:
        """Test that empty strings are accepted for all string fields.

        Verifies that the configuration class accepts empty strings for all
        string fields, following the design decision to allow empty values
        and let validation occur at API call time.
        """
        config = AzureOpenAIConfig(
            api_key="",
            endpoint="",
            api_version="",
            default_model="",
            default_embedding_model="",
        )

        assert config.api_key == ""
        assert config.endpoint == ""
        assert config.api_version == ""
        assert config.default_model == ""
        assert config.default_embedding_model == ""

    def test_whitespace_strings_are_preserved(self) -> None:
        """Test that whitespace-only strings are preserved exactly as provided.

        Verifies that strings containing only whitespace characters are
        stored without modification, maintaining the principle that the
        configuration class acts as a simple data holder.
        """
        config = AzureOpenAIConfig(
            api_key="   ",
            endpoint="  \t  ",
            api_version=" \n ",
            default_model="    ",
            default_embedding_model=" ",
        )

        assert config.api_key == "   "
        assert config.endpoint == "  \t  "
        assert config.api_version == " \n "
        assert config.default_model == "    "
        assert config.default_embedding_model == " "

    def test_model_dump_serialization(
        self, valid_azure_config: AzureOpenAIConfig
    ) -> None:
        """Test that configuration can be serialized to a dictionary.

        Verifies that the Pydantic model_dump method correctly serializes
        the configuration instance to a dictionary containing all field values.

        Args:
            valid_azure_config: A valid AzureOpenAIConfig instance.
        """
        config_dict = valid_azure_config.model_dump()

        assert config_dict["api_key"] == "test-api-key"
        assert config_dict["endpoint"] == "https://test.openai.azure.com/"
        assert config_dict["api_version"] == "2023-12-01-preview"
        assert config_dict["default_model"] == "gpt-4"
        assert config_dict["default_embedding_model"] == "text-embedding-ada-002"
        assert config_dict["supports_structured_output"] is True
        assert config_dict["supports_logprobs"] is True

    def test_reconstruction_from_dict(
        self, valid_azure_config: AzureOpenAIConfig
    ) -> None:
        """Test that configuration can be reconstructed from a dictionary.

        Verifies that an AzureOpenAIConfig instance can be created from
        a dictionary produced by model_dump, ensuring round-trip serialization
        works correctly.

        Args:
            valid_azure_config: A valid AzureOpenAIConfig instance.
        """
        config_dict = valid_azure_config.model_dump()
        reconstructed = AzureOpenAIConfig(**config_dict)

        assert reconstructed.api_key == valid_azure_config.api_key
        assert reconstructed.endpoint == valid_azure_config.endpoint
        assert reconstructed.api_version == valid_azure_config.api_version
        assert reconstructed.default_model == valid_azure_config.default_model
        assert (
            reconstructed.default_embedding_model
            == valid_azure_config.default_embedding_model
        )
        assert (
            reconstructed.supports_structured_output
            == valid_azure_config.supports_structured_output
        )
        assert reconstructed.supports_logprobs == valid_azure_config.supports_logprobs

    def test_boolean_fields_can_be_overridden(self) -> None:
        """Test that default boolean values can be explicitly overridden.

        Verifies that the supports_structured_output and supports_logprobs
        fields can be set to non-default values when explicitly provided
        during instantiation.
        """
        config = AzureOpenAIConfig(
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
            api_version="2023-12-01-preview",
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
            supports_structured_output=False,
            supports_logprobs=False,
        )

        assert config.supports_structured_output is False
        assert config.supports_logprobs is False

    def test_inheritance_from_base_class(
        self, valid_azure_config: AzureOpenAIConfig
    ) -> None:
        """Test that AzureOpenAIConfig properly inherits from LLMClientConfig.

        Verifies that the AzureOpenAIConfig class correctly inherits all
        functionality from the LLMClientConfig base class.

        Args:
            valid_azure_config: A valid AzureOpenAIConfig instance.
        """
        from meta_evaluator.llm_client import LLMClientConfig

        assert isinstance(valid_azure_config, LLMClientConfig)

        # Test that base class fields are accessible
        assert hasattr(valid_azure_config, "api_key")
        assert hasattr(valid_azure_config, "supports_structured_output")
        assert hasattr(valid_azure_config, "default_model")
        assert hasattr(valid_azure_config, "default_embedding_model")

        # Test that Azure-specific fields are accessible
        assert hasattr(valid_azure_config, "endpoint")
        assert hasattr(valid_azure_config, "api_version")
