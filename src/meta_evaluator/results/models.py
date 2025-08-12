"""Data models for the results domain."""

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, List, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class FieldTags:
    """Metadata for field tags."""

    tags: list[str]


class BaseResultRow(BaseModel):
    """Base class for result row structures."""

    sample_example_id: Annotated[
        str,
        Field(..., description="Unique identifier for the example within this run"),
        FieldTags(tags=["metadata"]),
    ]
    original_id: Annotated[
        str | int,
        Field(..., description="Original identifier from the source data"),
        FieldTags(tags=["metadata"]),
    ]
    run_id: Annotated[
        str,
        Field(..., description="Unique identifier for this evaluation run"),
        FieldTags(tags=["metadata"]),
    ]
    status: Annotated[
        str,
        Field(..., description="Status of the evaluation for this example"),
        FieldTags(tags=["metadata"]),
    ]
    error_message: Annotated[
        Optional[str],
        Field(None, description="Error message if evaluation failed"),
        FieldTags(tags=["error"]),
    ]
    error_details_json: Annotated[
        Optional[str],
        Field(None, description="JSON string with error details"),
        FieldTags(tags=["error"]),
    ]

    model_config = ConfigDict(extra="allow")

    @classmethod
    def get_fields_by_tag(cls, tag: str) -> list[str]:
        """Get field names that have a specific tag.

        Args:
            tag: The tag to filter by

        Returns:
            list[str]: List of field names with the specified tag
        """
        field_names = []
        for field_name, field_info in cls.model_fields.items():
            # Look for FieldTags in the metadata
            for metadata in field_info.metadata:
                if isinstance(metadata, FieldTags) and tag in metadata.tags:
                    field_names.append(field_name)
                    break
        return field_names

    @classmethod
    def get_metadata_fields(cls) -> list[str]:
        """Get all metadata field names.

        Returns:
            list[str]: List of field names tagged as metadata
        """
        return cls.get_fields_by_tag("metadata")

    @classmethod
    def get_error_fields(cls) -> list[str]:
        """Get all error field names.

        Returns:
            list[str]: List of field names tagged as error
        """
        return cls.get_fields_by_tag("error")

    @classmethod
    def get_all_base_fields(cls) -> list[str]:
        """Get all base field names.

        Returns:
            list[str]: List of all base field names
        """
        return list(cls.model_fields.keys())

    @classmethod
    def get_required_columns_with_tasks(cls, task_names: List[str]) -> List[str]:
        """Get all required columns including task columns.

        Args:
            task_names: List of task names to include as columns.

        Returns:
            List[str]: Combined list of base columns and task columns.
        """
        base_columns = list(cls.model_fields.keys())
        return base_columns + task_names


class JudgeResultRow(BaseResultRow):
    """Result row for Judge evaluation with LLM-specific fields."""

    judge_id: Annotated[
        str,
        Field(description="ID of the judge configuration used"),
        FieldTags(tags=["metadata"]),
    ]

    llm_raw_response_content: Annotated[
        Optional[str],
        Field(default=None, description="Raw response content from LLM"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    llm_prompt_tokens: Annotated[
        Optional[int],
        Field(default=None, description="Number of tokens used in the prompt"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    llm_completion_tokens: Annotated[
        Optional[int],
        Field(default=None, description="Number of tokens used in the completion"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    llm_total_tokens: Annotated[
        Optional[int],
        Field(default=None, description="Total number of tokens used"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    llm_call_duration_seconds: Annotated[
        Optional[float],
        Field(default=None, description="Duration of the LLM call in seconds"),
        FieldTags(tags=["llm_diagnostic"]),
    ]

    model_config = ConfigDict(extra="allow", frozen=False)

    @classmethod
    def get_llm_diagnostic_fields(cls) -> list[str]:
        """Get all LLM diagnostic field names.

        Returns:
            list[str]: List of LLM diagnostic field names.
        """
        return cls.get_fields_by_tag("llm_diagnostic")


class HumanAnnotationResultRow(BaseResultRow):
    """Result row for human annotation with annotation-specific fields."""

    annotator_id: Annotated[
        str,
        Field(description="ID of the human annotator"),
        FieldTags(tags=["metadata"]),
    ]

    annotation_timestamp: Annotated[
        Optional[datetime],
        Field(default=None, description="Timestamp when annotation was completed"),
        FieldTags(tags=["annotation_diagnostic"]),
    ]

    model_config = ConfigDict(extra="allow", frozen=False)

    @classmethod
    def get_annotation_diagnostic_fields(cls) -> list[str]:
        """Get all annotation diagnostic field names.

        Returns:
            list[str]: List of annotation diagnostic field names.
        """
        return cls.get_fields_by_tag("annotation_diagnostic")
