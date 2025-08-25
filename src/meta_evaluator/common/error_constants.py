"""Centralized error message constants used across the meta_evaluator package."""

# File-related error messages
STATE_FILE_NOT_FOUND_MSG = "State file not found"
INVALID_JSON_STRUCTURE_MSG = "Invalid JSON structure in state file"
INVALID_JSON_MSG = "Invalid JSON in state file"

# Aggregation mode validation error messages
INVALID_SINGLE_AGGREGATION_MSG = (
    "Aggregation mode 'single' can only be used with exactly 1 task name"
)
INVALID_MULTILABEL_MULTITASK_AGGREGATION_MSG = "Aggregation mode 'multilabel' and 'multitask' can only be used with more than 1 task name"
