"""Pydantic models for Evaluation Task serialization."""

from typing import Literal

from pydantic import BaseModel, model_validator

from .exceptions import TaskSchemaError

# The reserved sentinel marking a not-selected slot in a multi-label vector.
# It cannot be declared as an outcome because a selected slot stores the
# outcome's own name, so a slot literally named "FALSE" would be ambiguous.
MULTILABEL_FALSE_SENTINEL = "FALSE"


class MultiLabelSchema(BaseModel):
    """Wrapper marking a task as multi-label (pick several).

    A multi-label task's value is a fixed-length ordered vector with one slot
    per declared outcome. Each slot holds either that outcome's own name
    (selected) or the sentinel ``"FALSE"`` (not selected). The order of
    ``outcomes`` defines canonical slot order and is load-bearing: it must
    survive every serialize/judge/storage round-trip.

    Attributes:
        outcomes: The ordered list of outcome names. Must contain at least 2
            values and must not include the reserved ``"FALSE"`` sentinel.
    """

    outcomes: list[str]

    @model_validator(mode="after")
    def validate_outcomes(self) -> "MultiLabelSchema":
        """Validate the wrapped outcomes.

        Returns:
            MultiLabelSchema: The validated instance.

        Raises:
            TaskSchemaError: If fewer than 2 outcomes are given, or if the
                reserved ``"FALSE"`` sentinel is declared as an outcome.
        """
        if len(self.outcomes) < 2:
            raise TaskSchemaError(
                f"A multi-label task must define at least 2 outcomes, "
                f"got {self.outcomes}."
            )
        if MULTILABEL_FALSE_SENTINEL in self.outcomes:
            raise TaskSchemaError(
                f"'{MULTILABEL_FALSE_SENTINEL}' is reserved as the not-selected "
                f"sentinel for multi-label tasks and cannot be declared as an "
                f"outcome."
            )
        return self

    def validate_value(self, task_name: str, value: object) -> list[str]:
        """Validate one submitted value against the ordered slot contract.

        Args:
            task_name: Task name used in validation errors.
            value: Candidate multi-label value.

        Returns:
            list[str]: The validated value.

        Raises:
            ValueError: If the value is not a correctly aligned vector.
        """
        if not isinstance(value, list):
            raise ValueError(
                f"Multi-label task '{task_name}' expects a list, got "
                f"{type(value).__name__}."
            )
        if len(value) != len(self.outcomes):
            raise ValueError(
                f"Multi-label task '{task_name}' expects a vector of length "
                f"{len(self.outcomes)} (one slot per outcome), got length "
                f"{len(value)}: {value}."
            )
        for i, slot in enumerate(value):
            if slot != self.outcomes[i] and slot != MULTILABEL_FALSE_SENTINEL:
                raise ValueError(
                    f"Multi-label task '{task_name}' slot {i} must be "
                    f"'{self.outcomes[i]}' or '{MULTILABEL_FALSE_SENTINEL}', "
                    f"got '{slot}'."
                )
        return value


class EvalTaskState(BaseModel):
    """Serialized state for Evaluation Task.

    Contains all information needed to reconstruct an EvalTask object.
    """

    task_schemas: dict[str, list[str] | MultiLabelSchema | None]
    required_tasks: list[str] | None = None
    prompt_columns: list[str] | None
    response_columns: list[str]
    answering_method: Literal["structured", "instructor", "xml"]
    structured_outputs_fallback: bool = False
    annotation_prompt: str = "Please evaluate the following response:"
