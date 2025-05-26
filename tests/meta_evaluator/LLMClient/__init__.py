"""File for testing the LLMClient package."""

import pytest
from pydantic import ValidationError
from meta_evaluator.LLMClient import LLMClientConfig


class TestLLMClientConfig:
    """Test the LLMClientConfig abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMClientConfig cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMClientConfig()  # type: ignore

    def test_concrete_subclass_can_instantiate(self):
        """Test that a concrete subclass can be instantiated with valid data."""

        class ConcreteConfig(LLMClientConfig):
            pass

        config = ConcreteConfig(
            api_key="test-api-key",
            supports_instructor=True,
            default_model="gpt-4",
            default_embedding_model="text-embedding-ada-002",
        )

        assert config.api_key == "test-api-key"
        assert config.supports_instructor is True
        assert config.default_model == "gpt-4"
        assert config.default_embedding_model == "text-embedding-ada-002"

    def test_validation_works_for_missing_fields(self):
        """Test that Pydantic validation is properly configured."""

        class ConcreteConfig(LLMClientConfig):
            pass

        with pytest.raises(ValidationError):
            ConcreteConfig()  # type: ignore

        with pytest.raises(ValidationError):
            ConcreteConfig(api_key="test-key")  # type: ignore

    def test_validation_works_for_wrong_types(self):
        """Test that field types are validated correctly."""

        class ConcreteConfig(LLMClientConfig):
            pass

        with pytest.raises(ValidationError):
            ConcreteConfig(
                api_key=123,  # type: ignore
                supports_instructor=True,
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )

        with pytest.raises(ValidationError):
            ConcreteConfig(
                api_key="test-key",
                supports_instructor="yes",  # type: ignore
                default_model="gpt-4",
                default_embedding_model="text-embedding-ada-002",
            )
