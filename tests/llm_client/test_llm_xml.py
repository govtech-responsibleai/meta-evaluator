"""Comprehensive tests for XML parsing functionality in LLMClient."""

from typing import TypeVar

import pytest
from pydantic import BaseModel

from meta_evaluator.llm_client.exceptions import LLMAPIError, LLMValidationError
from meta_evaluator.llm_client.models import (
    ErrorType,
    TagConfig,
)

T = TypeVar("T", bound=BaseModel)


class TestXMLParsing:
    """Comprehensive test suite for XML parsing functionality."""

    # ===== _extract_tag_values Tests =====

    def test_extract_tag_values_simple_single_tag(self, basic_llm_client):
        """Test extracting a single simple XML tag."""
        text = "<status>active</status>"
        result = basic_llm_client._extract_tag_values("status", text)
        assert result == ["active"]

    def test_extract_tag_values_multiple_same_tags(self, basic_llm_client):
        """Test extracting multiple instances of the same tag."""
        text = "<tag>value1</tag><tag>value2</tag><tag>value3</tag>"
        result = basic_llm_client._extract_tag_values("tag", text)
        assert result == ["value1", "value2", "value3"]

    def test_extract_tag_values_with_attributes(self, basic_llm_client):
        """Test extracting tags that have attributes."""
        text = '<status type="current">active</status>'
        result = basic_llm_client._extract_tag_values("status", text)
        assert result == ["active"]

    def test_extract_tag_values_case_insensitive(self, basic_llm_client):
        """Test that tag extraction is case insensitive."""
        text = "<STATUS>active</STATUS>"
        result = basic_llm_client._extract_tag_values("status", text)
        assert result == ["active"]

    def test_extract_tag_values_multiline_content(self, basic_llm_client):
        """Test extracting tags with multiline content."""
        text = """<description>
        This is a long description
        that spans multiple lines
        </description>"""
        result = basic_llm_client._extract_tag_values("description", text)
        expected = "This is a long description\n        that spans multiple lines"
        assert result == [expected]

    def test_extract_tag_values_with_whitespace_trimming(self, basic_llm_client):
        """Test that whitespace is properly trimmed."""
        text = "<tag>  content with spaces  </tag>"
        result = basic_llm_client._extract_tag_values("tag", text)
        assert result == ["content with spaces"]

    def test_extract_tag_values_empty_tag_filtered(self, basic_llm_client):
        """Test extracting empty tags are filtered out."""
        text = "<empty></empty>"
        result = basic_llm_client._extract_tag_values("empty", text)
        assert result == []  # Empty strings are filtered out

    def test_extract_tag_values_whitespace_only_filtered(self, basic_llm_client):
        """Test extracting tags with only whitespace are filtered out."""
        text = "<tag>   </tag>"
        result = basic_llm_client._extract_tag_values("tag", text)
        assert result == []  # Whitespace-only strings are filtered out

    def test_extract_tag_values_no_matching_tags(self, basic_llm_client):
        """Test when no matching tags are found."""
        text = "<other>value</other>"
        result = basic_llm_client._extract_tag_values("missing", text)
        assert result == []

    def test_extract_tag_values_nested_tags_same_name(self, basic_llm_client):
        """Test handling nested tags with the same name."""
        text = "<tag>outer<tag>inner</tag>content</tag>"
        result = basic_llm_client._extract_tag_values("tag", text)
        # The regex matches first opening tag with first closing tag
        assert result == ["outer<tag>inner"]

    def test_extract_tag_values_with_special_characters(self, basic_llm_client):
        """Test extracting content with special characters."""
        text = "<content>Special chars: !@#$%^&*()[]{}|\\:;\"'<>,./</content>"
        result = basic_llm_client._extract_tag_values("content", text)
        assert result == ["Special chars: !@#$%^&*()[]{}|\\:;\"'<>,./"]

    def test_extract_tag_values_with_unicode(self, basic_llm_client):
        """Test extracting content with Unicode characters."""
        text = "<text>Unicode: 你好世界 🌍 café naïve</text>"
        result = basic_llm_client._extract_tag_values("text", text)
        assert result == ["Unicode: 你好世界 🌍 café naïve"]

    def test_extract_tag_values_malformed_xml(self, basic_llm_client):
        """Test behavior with malformed XML."""
        # Unclosed tag
        text = "<tag>content"
        result = basic_llm_client._extract_tag_values("tag", text)
        assert result == []

        # Mismatched tags
        text = "<tag>content</other>"
        result = basic_llm_client._extract_tag_values("tag", text)
        assert result == []

    # ===== _construct_xml_tag_parsing Tests - Category A: Tag Not Found =====

    def test_tag_not_found_cardinality_one(self, basic_llm_client):
        """Test tag not found with cardinality='one'."""
        config = TagConfig(name="missing", cardinality="one")
        text = "<other>value</other>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "missing" not in result.data  # Clean separation: no data entry
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.TAG_NOT_FOUND
        assert result.errors[0].tag_name == "missing"
        assert "not found in text" in result.errors[0].message

    def test_tag_not_found_cardinality_many(self, basic_llm_client):
        """Test tag not found with cardinality='many'."""
        config = TagConfig(name="missing", cardinality="many")
        text = "<other>value</other>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "missing" not in result.data  # Clean separation: no data entry
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.TAG_NOT_FOUND

    # ===== Category B: No Allowed Values (Freeform) - Cardinality One =====

    def test_freeform_cardinality_one_single_value_success(self, basic_llm_client):
        """Test freeform single value with cardinality='one' - success path."""
        config = TagConfig(name="tag", allowed_values=None, cardinality="one")
        text = "<tag>freeform_value</tag>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is True
        assert result.data["tag"] == "freeform_value"
        assert len(result.errors) == 0

    def test_freeform_cardinality_one_multiple_values_error(self, basic_llm_client):
        """Test freeform multiple values with cardinality='one' and multiple_handling='error'."""
        config = TagConfig(
            name="tag",
            allowed_values=None,
            cardinality="one",
            multiple_handling="error",
        )
        text = "<tag>val1</tag><tag>val2</tag>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "tag" not in result.data  # Clean separation: error means no data
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.CARDINALITY_MISMATCH
        assert "Expected exactly 1 value" in result.errors[0].message
        assert result.errors[0].found_values == ["val1", "val2"]

    def test_freeform_cardinality_one_multiple_values_allow_both(
        self, basic_llm_client
    ):
        """Test freeform multiple values with cardinality='one' and multiple_handling='allow_both'."""
        config = TagConfig(
            name="tag",
            allowed_values=None,
            cardinality="one",
            multiple_handling="allow_both",
        )
        text = "<tag>val1</tag><tag>val2</tag>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is True
        assert result.data["tag"] == ["val1", "val2"]
        assert len(result.errors) == 0

    def test_freeform_cardinality_one_multiple_identical_error_if_different(
        self, basic_llm_client
    ):
        """Test freeform identical multiple values with multiple_handling='error_if_different'."""
        config = TagConfig(
            name="tag",
            allowed_values=None,
            cardinality="one",
            multiple_handling="error_if_different",
        )
        text = "<tag>same</tag><tag>same</tag>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is True
        assert result.data["tag"] == "same"  # Identical values accepted
        assert len(result.errors) == 0

    def test_freeform_cardinality_one_multiple_different_error_if_different(
        self, basic_llm_client
    ):
        """Test freeform different multiple values with multiple_handling='error_if_different'."""
        config = TagConfig(
            name="tag",
            allowed_values=None,
            cardinality="one",
            multiple_handling="error_if_different",
        )
        text = "<tag>val1</tag><tag>val2</tag>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "tag" not in result.data  # Different values = error = no data
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.MULTIPLE_VALUES_CONFLICT
        assert "Multiple different values" in result.errors[0].message
        assert result.errors[0].found_values == ["val1", "val2"]

    # ===== Category C: No Allowed Values (Freeform) - Cardinality Many =====

    def test_freeform_cardinality_many_single_value_success(self, basic_llm_client):
        """Test freeform single value with cardinality='many' - success path."""
        config = TagConfig(name="tag", allowed_values=None, cardinality="many")
        text = "<tag>single_value</tag>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is True
        assert result.data["tag"] == ["single_value"]
        assert len(result.errors) == 0

    def test_freeform_cardinality_many_multiple_values_success(self, basic_llm_client):
        """Test freeform multiple values with cardinality='many' - success path."""
        config = TagConfig(name="tag", allowed_values=None, cardinality="many")
        text = "<tag>val1</tag><tag>val2</tag><tag>val3</tag>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is True
        assert result.data["tag"] == ["val1", "val2", "val3"]
        assert len(result.errors) == 0

    # ===== Category D: With Allowed Values - All Valid =====

    def test_allowed_values_all_valid_cardinality_one_success(self, basic_llm_client):
        """Test all valid values with cardinality='one' - success path."""
        config = TagConfig(
            name="priority", allowed_values=["low", "medium", "high"], cardinality="one"
        )
        text = "<priority>high</priority>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is True
        assert result.data["priority"] == "high"
        assert len(result.errors) == 0

    def test_allowed_values_all_valid_cardinality_many_success(self, basic_llm_client):
        """Test all valid values with cardinality='many' - success path."""
        config = TagConfig(
            name="colors", allowed_values=["red", "green", "blue"], cardinality="many"
        )
        text = "<colors>red</colors><colors>blue</colors>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is True
        assert result.data["colors"] == ["red", "blue"]
        assert len(result.errors) == 0

    def test_allowed_values_all_valid_cardinality_one_multiple_error(
        self, basic_llm_client
    ):
        """Test multiple valid values with cardinality='one' and multiple_handling='error'."""
        config = TagConfig(
            name="status",
            allowed_values=["active", "inactive"],
            cardinality="one",
            multiple_handling="error",
        )
        text = "<status>active</status><status>inactive</status>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "status" not in result.data  # Multiple values = error = no data
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.CARDINALITY_MISMATCH
        assert result.errors[0].found_values == ["active", "inactive"]

    # ===== Category E: With Allowed Values - All Invalid =====

    def test_allowed_values_all_invalid_cardinality_one(self, basic_llm_client):
        """Test all invalid values with cardinality='one'."""
        config = TagConfig(
            name="priority", allowed_values=["low", "medium", "high"], cardinality="one"
        )
        text = "<priority>urgent</priority>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "priority" not in result.data  # Invalid value = no data entry
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.INVALID_VALUE
        assert result.errors[0].tag_name == "priority"
        assert "urgent" in result.errors[0].message
        assert result.errors[0].found_values == ["urgent"]
        assert result.errors[0].expected_values == ["low", "medium", "high"]

    def test_allowed_values_all_invalid_cardinality_many(self, basic_llm_client):
        """Test all invalid values with cardinality='many'."""
        config = TagConfig(
            name="colors", allowed_values=["red", "green", "blue"], cardinality="many"
        )
        text = "<colors>purple</colors><colors>yellow</colors>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "colors" not in result.data  # All invalid = no data entry
        assert len(result.errors) == 2  # Two invalid value errors
        assert all(e.error_type == ErrorType.INVALID_VALUE for e in result.errors)
        assert result.errors[0].found_values == ["purple"]
        assert result.errors[1].found_values == ["yellow"]

    def test_allowed_values_all_invalid_multiple_same_values(self, basic_llm_client):
        """Test multiple identical invalid values."""
        config = TagConfig(
            name="status", allowed_values=["active", "inactive"], cardinality="one"
        )
        text = "<status>invalid</status><status>invalid</status>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "status" not in result.data
        assert len(result.errors) == 2  # Two separate invalid value errors
        assert all(e.error_type == ErrorType.INVALID_VALUE for e in result.errors)
        assert all("invalid" in e.message for e in result.errors)

    # ===== Category F: With Allowed Values - Mixed Valid/Invalid =====

    def test_allowed_values_mixed_cardinality_one_single_valid_success(
        self, basic_llm_client
    ):
        """Test mixed valid/invalid with cardinality='one' where one valid value remains."""
        config = TagConfig(
            name="priority", allowed_values=["low", "medium", "high"], cardinality="one"
        )
        text = "<priority>invalid</priority><priority>high</priority>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False  # Has errors but partial success
        assert result.partial_success is True
        assert result.data["priority"] == "high"  # Valid value preserved
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.INVALID_VALUE
        assert "invalid" in result.errors[0].message

    def test_allowed_values_mixed_cardinality_many_some_valid_success(
        self, basic_llm_client
    ):
        """Test mixed valid/invalid with cardinality='many' where some valid values remain."""
        config = TagConfig(
            name="colors", allowed_values=["red", "green", "blue"], cardinality="many"
        )
        text = "<colors>purple</colors><colors>red</colors><colors>blue</colors>"

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False  # Has errors
        assert result.partial_success is True
        assert result.data["colors"] == ["red", "blue"]  # Only valid values
        assert len(result.errors) == 1
        assert result.errors[0].error_type == ErrorType.INVALID_VALUE
        assert "purple" in result.errors[0].message

    def test_allowed_values_mixed_cardinality_one_multiple_valid_error(
        self, basic_llm_client
    ):
        """Test mixed valid/invalid with cardinality='one' where multiple valid values cause error."""
        config = TagConfig(
            name="status",
            allowed_values=["active", "inactive"],
            cardinality="one",
            multiple_handling="error",
        )
        text = (
            "<status>invalid</status><status>active</status><status>inactive</status>"
        )

        result = basic_llm_client._construct_xml_tag_parsing([config], text)

        assert result.success is False
        assert "status" not in result.data  # Cardinality error = no data
        assert len(result.errors) == 2  # Invalid value + cardinality mismatch
        error_types = {e.error_type for e in result.errors}
        assert ErrorType.INVALID_VALUE in error_types
        assert ErrorType.CARDINALITY_MISMATCH in error_types

    # ===== Category G: Multiple Configs (Integration) =====

    def test_multiple_configs_mixed_success_and_failure(self, basic_llm_client):
        """Test multiple configs with mixed success and failure outcomes."""
        configs = [
            TagConfig(name="success", cardinality="one"),  # Will succeed
            TagConfig(name="missing", cardinality="one"),  # Will fail - not found
            TagConfig(
                name="invalid", allowed_values=["valid"], cardinality="one"
            ),  # Will fail - invalid value
        ]
        text = "<success>good</success><invalid>bad</invalid>"

        result = basic_llm_client._construct_xml_tag_parsing(configs, text)

        assert result.success is False
        assert result.partial_success is True
        # Only successful parse in data
        assert result.data == {"success": "good"}
        assert "missing" not in result.data
        assert "invalid" not in result.data
        # Two errors
        assert len(result.errors) == 2
        error_types = {e.error_type for e in result.errors}
        assert ErrorType.TAG_NOT_FOUND in error_types
        assert ErrorType.INVALID_VALUE in error_types

    def test_multiple_configs_all_success(self, basic_llm_client):
        """Test multiple configs with all successful outcomes."""
        configs = [
            TagConfig(name="status", cardinality="one"),
            TagConfig(name="tags", cardinality="many"),
            TagConfig(
                name="priority", allowed_values=["low", "high"], cardinality="one"
            ),
        ]
        text = "<status>active</status><tags>red</tags><tags>blue</tags><priority>high</priority>"

        result = basic_llm_client._construct_xml_tag_parsing(configs, text)

        assert result.success is True
        assert result.data["status"] == "active"
        assert result.data["tags"] == ["red", "blue"]
        assert result.data["priority"] == "high"
        assert len(result.errors) == 0

    def test_multiple_configs_all_failure(self, basic_llm_client):
        """Test multiple configs with all failure outcomes."""
        configs = [
            TagConfig(name="missing1", cardinality="one"),
            TagConfig(name="missing2", cardinality="many"),
            TagConfig(name="invalid", allowed_values=["valid"], cardinality="one"),
        ]
        text = "<invalid>bad_value</invalid>"

        result = basic_llm_client._construct_xml_tag_parsing(configs, text)

        assert result.success is False
        assert result.partial_success is False
        assert result.data == {}  # No successful parses
        assert len(result.errors) == 3  # All three configs failed
        error_types = [e.error_type for e in result.errors]
        assert error_types.count(ErrorType.TAG_NOT_FOUND) == 2
        assert error_types.count(ErrorType.INVALID_VALUE) == 1

    # ===== Category H: Edge Cases =====

    def test_empty_config_list(self, basic_llm_client):
        """Test parsing with empty config list."""
        result = basic_llm_client._construct_xml_tag_parsing([], "<tag>value</tag>")

        assert result.success is True
        assert result.data == {}
        assert len(result.errors) == 0

    def test_empty_text(self, basic_llm_client):
        """Test parsing with empty text."""
        configs = [
            TagConfig(name="tag1", cardinality="one"),
            TagConfig(name="tag2", cardinality="many"),
        ]

        result = basic_llm_client._construct_xml_tag_parsing(configs, "")

        assert result.success is False
        assert result.data == {}
        assert len(result.errors) == 2
        assert all(e.error_type == ErrorType.TAG_NOT_FOUND for e in result.errors)

    # ===== ParseResult Helper Methods Tests =====

    def test_parse_result_get_errors_by_tag(self, basic_llm_client):
        """Test getting errors filtered by tag name."""
        configs = [
            TagConfig(name="tag1", allowed_values=["valid"], cardinality="one"),
            TagConfig(name="tag2", allowed_values=["valid"], cardinality="one"),
        ]
        text = "<tag1>invalid1</tag1><tag2>invalid2</tag2>"

        result = basic_llm_client._construct_xml_tag_parsing(configs, text)

        tag1_errors = result.get_errors_by_tag("tag1")
        tag2_errors = result.get_errors_by_tag("tag2")
        nonexistent_errors = result.get_errors_by_tag("nonexistent")

        assert len(tag1_errors) == 1
        assert len(tag2_errors) == 1
        assert len(nonexistent_errors) == 0
        assert tag1_errors[0].tag_name == "tag1"
        assert tag2_errors[0].tag_name == "tag2"

    def test_parse_result_get_errors_by_type(self, basic_llm_client):
        """Test getting errors filtered by error type."""
        configs = [
            TagConfig(name="missing", cardinality="one"),
            TagConfig(name="invalid", allowed_values=["valid"], cardinality="one"),
        ]
        text = "<invalid>bad_value</invalid>"

        result = basic_llm_client._construct_xml_tag_parsing(configs, text)

        not_found_errors = result.get_errors_by_type(ErrorType.TAG_NOT_FOUND)
        invalid_value_errors = result.get_errors_by_type(ErrorType.INVALID_VALUE)
        nonexistent_errors = result.get_errors_by_type(ErrorType.CARDINALITY_MISMATCH)

        assert len(not_found_errors) == 1
        assert len(invalid_value_errors) == 1
        assert len(nonexistent_errors) == 0
        assert not_found_errors[0].tag_name == "missing"
        assert invalid_value_errors[0].tag_name == "invalid"

    def test_parse_result_properties(self, basic_llm_client):
        """Test ParseResult success and partial_success properties."""
        # Test success = True
        config_success = TagConfig(name="good", cardinality="one")
        result_success = basic_llm_client._construct_xml_tag_parsing(
            [config_success], "<good>value</good>"
        )
        assert result_success.success is True
        assert result_success.partial_success is True

        # Test success = False, partial_success = True
        configs_partial = [
            TagConfig(name="good", cardinality="one"),
            TagConfig(name="missing", cardinality="one"),
        ]
        result_partial = basic_llm_client._construct_xml_tag_parsing(
            configs_partial, "<good>value</good>"
        )
        assert result_partial.success is False
        assert result_partial.partial_success is True

        # Test success = False, partial_success = False
        config_failure = TagConfig(name="missing", cardinality="one")
        result_failure = basic_llm_client._construct_xml_tag_parsing(
            [config_failure], "<other>value</other>"
        )
        assert result_failure.success is False
        assert result_failure.partial_success is False

    # ===== prompt_with_xml_tags Integration Tests =====

    def test_prompt_with_xml_tags_success(
        self, basic_llm_client, simple_user_message, mock_usage, mocker
    ):
        """Test successful prompt with XML tag parsing integration."""
        xml_response = "<status>completed</status><priority>high</priority>"
        mocker.patch.object(
            basic_llm_client, "_prompt", return_value=(xml_response, mock_usage)
        )

        configs = [
            TagConfig(name="status", cardinality="one"),
            TagConfig(
                name="priority",
                allowed_values=["low", "medium", "high"],
                cardinality="one",
            ),
        ]

        parse_result, llm_response = basic_llm_client.prompt_with_xml_tags(
            simple_user_message, configs
        )

        assert parse_result.success is True
        assert parse_result.data["status"] == "completed"
        assert parse_result.data["priority"] == "high"
        assert len(parse_result.errors) == 0

        assert llm_response.content == xml_response
        assert llm_response.usage == mock_usage
        assert len(llm_response.messages) == 2

    def test_prompt_with_xml_tags_parsing_errors(
        self, basic_llm_client, simple_user_message, mock_usage, mocker
    ):
        """Test prompt with XML parsing errors."""
        xml_response = "<status>invalid_status</status>"
        mocker.patch.object(
            basic_llm_client, "_prompt", return_value=(xml_response, mock_usage)
        )

        configs = [
            TagConfig(
                name="status", allowed_values=["active", "inactive"], cardinality="one"
            )
        ]

        parse_result, llm_response = basic_llm_client.prompt_with_xml_tags(
            simple_user_message, configs
        )

        assert parse_result.success is False
        assert "status" not in parse_result.data  # Clean separation
        assert len(parse_result.errors) == 1
        assert parse_result.errors[0].error_type == ErrorType.INVALID_VALUE

        assert llm_response.content == xml_response

    def test_prompt_with_xml_tags_partial_success(
        self, basic_llm_client, simple_user_message, mock_usage, mocker
    ):
        """Test prompt with XML tags resulting in partial success."""
        xml_response = "<good>value</good><bad>invalid</bad>"
        mocker.patch.object(
            basic_llm_client, "_prompt", return_value=(xml_response, mock_usage)
        )

        configs = [
            TagConfig(name="good", cardinality="one"),
            TagConfig(name="bad", allowed_values=["valid"], cardinality="one"),
            TagConfig(name="missing", cardinality="one"),
        ]

        parse_result, llm_response = basic_llm_client.prompt_with_xml_tags(
            simple_user_message, configs
        )

        assert parse_result.success is False
        assert parse_result.partial_success is True
        assert parse_result.data == {"good": "value"}  # Only successful parse
        assert len(parse_result.errors) == 2  # Invalid value + missing tag

    def test_prompt_with_xml_tags_validation_error(
        self, basic_llm_client, mock_usage, mocker
    ):
        """Test prompt with XML tags when message validation fails."""
        configs = [TagConfig(name="test", cardinality="one")]

        with pytest.raises(LLMValidationError):
            basic_llm_client.prompt_with_xml_tags([], configs)

    def test_prompt_with_xml_tags_api_error(
        self, basic_llm_client, simple_user_message, mocker
    ):
        """Test prompt with XML tags when LLM API call fails."""
        mocker.patch.object(
            basic_llm_client, "_prompt", side_effect=RuntimeError("API Error")
        )

        configs = [TagConfig(name="test", cardinality="one")]

        with pytest.raises(LLMAPIError) as exc_info:
            basic_llm_client.prompt_with_xml_tags(simple_user_message, configs)

        assert "Failed to get response" in str(exc_info.value)
        assert isinstance(exc_info.value.original_error, RuntimeError)

    def test_prompt_with_xml_tags_explicit_model(
        self, basic_llm_client, simple_user_message, mock_usage, mocker
    ):
        """Test prompt with XML tags using explicit model parameter."""
        xml_response = "<result>success</result>"
        mock_prompt = mocker.patch.object(
            basic_llm_client, "_prompt", return_value=(xml_response, mock_usage)
        )

        configs = [TagConfig(name="result", cardinality="one")]
        explicit_model = "custom-model"

        parse_result, llm_response = basic_llm_client.prompt_with_xml_tags(
            simple_user_message, configs, model=explicit_model
        )

        mock_prompt.assert_called_once_with(
            model=explicit_model, messages=simple_user_message, get_logprobs=False
        )
        assert parse_result.success is True
        assert parse_result.data["result"] == "success"

    def test_prompt_with_xml_tags_logging_success(
        self, basic_llm_client, simple_user_message, mock_usage, mocker
    ):
        """Test that prompt_with_xml_tags logs appropriately on success."""
        xml_response = "<status>completed</status>"
        mocker.patch.object(
            basic_llm_client, "_prompt", return_value=(xml_response, mock_usage)
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        configs = [TagConfig(name="status", cardinality="one")]

        parse_result, llm_response = basic_llm_client.prompt_with_xml_tags(
            simple_user_message, configs
        )

        # Verify key logging calls
        mock_logger.info.assert_any_call(
            f"Using model: {basic_llm_client.config.default_model}"
        )
        mock_logger.info.assert_any_call("XML Tag Configurations: ['status']")
        mock_logger.info.assert_any_call("Successfully parsed 1 XML tags")
        mock_logger.info.assert_any_call("XML parsing completed successfully")

    def test_prompt_with_xml_tags_logging_with_errors(
        self, basic_llm_client, simple_user_message, mock_usage, mocker
    ):
        """Test logging when XML parsing has errors."""
        xml_response = "<status>invalid</status>"
        mocker.patch.object(
            basic_llm_client, "_prompt", return_value=(xml_response, mock_usage)
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        configs = [
            TagConfig(name="status", allowed_values=["valid"], cardinality="one")
        ]

        parse_result, llm_response = basic_llm_client.prompt_with_xml_tags(
            simple_user_message, configs
        )

        # Verify error logging
        mock_logger.warning.assert_any_call("XML parsing encountered 1 errors:")
        mock_logger.warning.assert_any_call(f"  - {parse_result.errors[0]}")

    def test_prompt_with_xml_tags_logging_partial_success(
        self, basic_llm_client, simple_user_message, mock_usage, mocker
    ):
        """Test logging when XML parsing has partial success."""
        xml_response = "<good>value</good><bad>invalid</bad>"
        mocker.patch.object(
            basic_llm_client, "_prompt", return_value=(xml_response, mock_usage)
        )
        mock_logger = mocker.patch.object(basic_llm_client, "logger")

        configs = [
            TagConfig(name="good", cardinality="one"),
            TagConfig(name="bad", allowed_values=["valid"], cardinality="one"),
        ]

        parse_result, llm_response = basic_llm_client.prompt_with_xml_tags(
            simple_user_message, configs
        )

        # Verify partial success logging
        mock_logger.info.assert_any_call("Successfully parsed 1 XML tags")
        mock_logger.info.assert_any_call("XML parsing completed with partial success")
        mock_logger.warning.assert_any_call("XML parsing encountered 1 errors:")

    # ===== Performance and Security Edge Cases =====

    def test_xml_parsing_large_content_performance(self, basic_llm_client):
        """Test that XML parsing handles large content efficiently."""
        large_content = "x" * 50000
        text = f"<content>{large_content}</content>"

        import time

        start_time = time.time()
        result = basic_llm_client._extract_tag_values("content", text)
        end_time = time.time()

        assert (end_time - start_time) < 1.0  # Should complete quickly
        assert result == [large_content]

    def test_xml_parsing_many_tags_performance(self, basic_llm_client):
        """Test parsing many tags performs reasonably."""
        # Create 1000 tags
        tags = "".join(f"<item>value{i}</item>" for i in range(1000))

        import time

        start_time = time.time()
        result = basic_llm_client._extract_tag_values("item", tags)
        end_time = time.time()

        assert (end_time - start_time) < 2.0  # Should complete reasonably quickly
        assert len(result) == 1000
        assert result[0] == "value0"
        assert result[999] == "value999"
