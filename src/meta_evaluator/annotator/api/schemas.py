"""Pydantic request/response models for annotation API."""

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request to create a new annotation session."""

    annotator_name: str = Field(..., min_length=1)


class CreateSessionResponse(BaseModel):
    """Response after creating a session."""

    run_id: str
    annotator_id: str
    total_samples: int
    resumed: bool = False
    annotated_count: int = 0


class TaskConfigResponse(BaseModel):
    """Response containing task configuration."""

    task_schemas: dict[str, list[str] | None]
    prompt_columns: list[str] | None
    response_columns: list[str]
    annotation_prompt: str
    required_tasks: list[str]


class SampleResponse(BaseModel):
    """Response containing a single sample's data."""

    index: int
    total: int
    sample_id: str
    prompt_data: dict[str, str] | None
    response_data: dict[str, str]
    previous_annotation: dict[str, str] | None = None


class SubmitAnnotationRequest(BaseModel):
    """Request to submit an annotation."""

    run_id: str
    sample_index: int = Field(..., ge=0)
    outcomes: dict[str, str]


class SubmitAnnotationResponse(BaseModel):
    """Response after submitting annotation."""

    success: bool
    annotated_count: int
    auto_saved: bool


class ProgressResponse(BaseModel):
    """Response containing annotation progress."""

    run_id: str
    annotated_count: int
    total_samples: int
    incomplete_indices: list[int]


class ExportRequest(BaseModel):
    """Request to export annotations."""

    run_id: str


class ExportResponse(BaseModel):
    """Response after export."""

    metadata_file: str
    data_file: str
    total_count: int
    succeeded_count: int
    error_count: int
