"""Common fixtures for MetaEvaluator tests.

This conftest provides high-level fixtures used across multiple test modules.
Module-specific fixtures are defined in their respective conftest files.
"""

import logging
import pytest
from unittest.mock import MagicMock
import polars as pl
from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.data import EvalData, SampleEvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.common.models import Prompt


@pytest.fixture
def meta_evaluator(tmp_path) -> MetaEvaluator:
    """Provides a fresh MetaEvaluator instance for testing.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        MetaEvaluator: A new MetaEvaluator instance with temporary project directory.
    """
    return MetaEvaluator(str(tmp_path / "test_project"))


@pytest.fixture
def logger():
    """Provides a mock logger for testing.

    Returns:
        MagicMock: A mock logger instance.
    """
    return MagicMock(spec=logging.Logger)


# ==== EVAL TASK FIXTURES ====


@pytest.fixture
def basic_eval_task() -> EvalTask:
    """Provides a basic evaluation task for common testing scenarios.

    Returns:
        EvalTask: A basic evaluation task with sentiment analysis schema.
    """
    return EvalTask(
        task_schemas={"sentiment": ["positive", "negative", "neutral"]},
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def multi_task_eval_task() -> EvalTask:
    """Provides a multi-task evaluation task for testing.

    Returns:
        EvalTask: A multi-task evaluation task.
    """
    return EvalTask(
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "toxicity": ["toxic", "non_toxic"],
        },
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
    )


@pytest.fixture
def xml_eval_task() -> EvalTask:
    """Provides an XML-based evaluation task for testing.

    Returns:
        EvalTask: An XML-based evaluation task with sentiment analysis schema.
    """
    return EvalTask(
        task_schemas={"sentiment": ["positive", "negative", "neutral"]},
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="xml",
    )


# ==== PROMPT FIXTURES ====


@pytest.fixture
def sample_prompt():
    """Provides a sample Prompt object for testing.

    Returns:
        Prompt: A sample prompt for testing.
    """
    return Prompt(
        id="basic_prompt",
        prompt="You are a helpful evaluator. Rate the response on accuracy.",
    )


@pytest.fixture
def sentiment_prompt() -> Prompt:
    """Provides a sentiment analysis prompt for testing.

    Returns:
        Prompt: A sentiment analysis prompt.
    """
    return Prompt(
        id="sentiment_prompt",
        prompt="Analyze the sentiment of the following text. Classify it as positive, negative, or neutral.",
    )


# ==== EVAL DATA FIXTURES ====


@pytest.fixture
def basic_eval_data():
    """Provides a basic EvalData for testing.

    Returns:
        EvalData: A basic EvalData instance with sample data.
    """
    test_df = pl.DataFrame(
        {
            "id": ["1", "2", "3"],
            "text": [
                "This movie is fantastic!",
                "The service was terrible.",
                "The weather is okay today.",
            ],
            "response": [
                "I really enjoyed watching this film.",
                "The staff was unhelpful and rude.",
                "It's neither good nor bad weather.",
            ],
            "context": ["movie review", "service review", "weather comment"],
        }
    )

    return EvalData(
        name="basic_test_data",
        data=test_df,
        id_column="id",
    )


@pytest.fixture
def sample_eval_data():
    """Provides sample evaluation data for testing sampling scenarios.

    Returns:
        SampleEvalData: Sample evaluation data for testing sampling scenarios.
    """
    test_df = pl.DataFrame(
        {
            "sample_id": ["1", "2"],
            "text": ["Great product!", "Poor quality item."],
            "response": ["Highly recommend this.", "Would not buy again."],
            "category": ["electronics", "clothing"],
        }
    )

    return SampleEvalData(
        name="sample_test_data",
        data=test_df,
        id_column="sample_id",
        sample_name="Test Sample",
        stratification_columns=["category"],
        sample_percentage=0.5,
        seed=42,
        sampling_method="stratified_by_columns",
    )


# ==== TASK SCHEMAS FIXTURES ====


@pytest.fixture
def basic_task_schemas():
    """Provides basic task schemas for testing.

    Returns:
        dict: Task schemas for testing.
    """
    return {
        "sentiment": ["positive", "negative", "neutral"],
        "quality": ["high", "medium", "low"],
    }


@pytest.fixture
def single_task_schemas():
    """Provides single task schemas for testing.

    Returns:
        dict: Single task schema for basic testing.
    """
    return {"sentiment": ["positive", "negative", "neutral"]}
