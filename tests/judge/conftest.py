"""Fixtures for Judge and JudgeResults testing.

This conftest provides judge-specific fixtures used across judge test modules.
Common fixtures are inherited from the main conftest.py.
"""

from unittest.mock import Mock

import pytest

from meta_evaluator.common.models import Prompt
from meta_evaluator.judge.judge import Judge
from meta_evaluator.llm_client import LLMClient, LLMClientEnum
from meta_evaluator.llm_client.async_client import AsyncLLMClient
from meta_evaluator.llm_client.enums import AsyncLLMClientEnum
from meta_evaluator.llm_client.models import (
    LLMResponse,
    LLMUsage,
    Message,
    RoleEnum,
)
from meta_evaluator.results import JudgeResults, JudgeResultsBuilder

# ==== JUDGE FIXTURES ====


@pytest.fixture
def sentiment_judge_prompt() -> Prompt:
    """Provides a sentiment-specific prompt for judge testing.

    Returns:
        Prompt: A sentiment-specific prompt for judge testing.
    """
    return Prompt(
        id="sentiment_prompt",
        prompt="Evaluate the sentiment of the given text.",
    )


@pytest.fixture
def basic_judge(basic_eval_task, sentiment_judge_prompt) -> Judge:
    """Provides a basic judge configuration for testing.

    Args:
        basic_eval_task: Basic evaluation task from main conftest.
        sentiment_judge_prompt: Sentiment-specific prompt.

    Returns:
        Judge: A basic judge configuration with structured output.
    """
    return Judge(
        id="test_judge_1",
        eval_task=basic_eval_task,
        llm_client_enum=LLMClientEnum.OPENAI,
        model="gpt-4",
        prompt=sentiment_judge_prompt,
    )


@pytest.fixture
def xml_judge(xml_eval_task, sentiment_judge_prompt) -> Judge:
    """Provides an XML-based judge configuration for testing.

    Args:
        xml_eval_task: XML-based evaluation task from main conftest.
        sentiment_judge_prompt: Sentiment-specific prompt.

    Returns:
        Judge: An XML-based judge instance.
    """
    return Judge(
        id="xml_judge_1",
        eval_task=xml_eval_task,
        llm_client_enum=LLMClientEnum.OPENAI,
        model="gpt-4",
        prompt=sentiment_judge_prompt,
    )


@pytest.fixture
def multi_task_judge(multi_task_eval_task, sentiment_judge_prompt) -> Judge:
    """Provides a multi-task judge configuration for testing.

    Args:
        multi_task_eval_task: Multi-task evaluation task from main conftest.
        sentiment_judge_prompt: Sentiment-specific prompt.

    Returns:
        Judge: A multi-task judge configuration.
    """
    return Judge(
        id="multi_task_judge",
        eval_task=multi_task_eval_task,
        llm_client_enum=LLMClientEnum.OPENAI,
        model="gpt-4",
        prompt=sentiment_judge_prompt,
    )


# ==== JUDGE RESULTS BUILDER FIXTURES ====


@pytest.fixture
def base_judge_results_builder(single_task_schemas) -> JudgeResultsBuilder:
    """Provides a basic JudgeResultsBuilder for testing.

    Args:
        single_task_schemas: Single task schemas from main conftest.

    Returns:
        JudgeResultsBuilder: A basic builder instance.
    """
    return JudgeResultsBuilder(
        run_id="test_run_123",
        judge_id="test_judge_1",
        llm_client_enum=LLMClientEnum.OPENAI,
        model_used="gpt-4",
        task_schemas=single_task_schemas,
        expected_ids=["id1", "id2", "id3"],
        is_sampled_run=False,
    )


@pytest.fixture
def multi_task_judge_results_builder() -> JudgeResultsBuilder:
    """Provides a builder with multiple tasks for testing.

    Returns:
        JudgeResultsBuilder: A builder with multiple tasks.
    """
    return JudgeResultsBuilder(
        run_id="multi_task_run",
        judge_id="multi_judge",
        llm_client_enum=LLMClientEnum.OPENAI,
        model_used="gpt-4",
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "toxicity": ["toxic", "non_toxic"],
        },
        expected_ids=["id1", "id2", "id3", "id4", "id5"],
        is_sampled_run=True,
    )


@pytest.fixture
def single_task_judge_results_builder() -> JudgeResultsBuilder:
    """Provides a builder with single task for testing.

    Returns:
        JudgeResultsBuilder: A builder with single task.
    """
    return JudgeResultsBuilder(
        run_id="single_task_run",
        judge_id="single_judge",
        llm_client_enum=LLMClientEnum.ANTHROPIC,
        model_used="claude-3",
        task_schemas={"sentiment": ["positive", "negative"]},
        expected_ids=["id1"],
        is_sampled_run=False,
    )


@pytest.fixture
def serialization_task_schemas():
    """Provides task schemas for serialization testing.

    Returns:
        dict: Task schemas with task1 and task2 for serialization tests.
    """
    return {"task1": ["yes", "no"], "task2": ["good", "bad"]}


@pytest.fixture
def serialization_judge_results_builder(
    serialization_task_schemas,
) -> JudgeResultsBuilder:
    """Provides a JudgeResultsBuilder for serialization testing.

    Args:
        serialization_task_schemas: Task schemas with task1 and task2.

    Returns:
        JudgeResultsBuilder: A builder for serialization testing.
    """
    return JudgeResultsBuilder(
        run_id="run_001",
        judge_id="judge_001",
        llm_client_enum=LLMClientEnum.OPENAI,
        model_used="gpt-4",
        task_schemas=serialization_task_schemas,
        expected_ids=["id1", "id2", "id3", "id4"],
        is_sampled_run=False,
    )


@pytest.fixture
def serialization_sample_judge_results(
    serialization_judge_results_builder,
) -> JudgeResults:
    """Create sample JudgeResults for serialization testing.

    Args:
        serialization_judge_results_builder: Builder with task1/task2 schemas.

    Returns:
        JudgeResults: Sample results with task1 and task2 data.
    """
    builder = serialization_judge_results_builder

    # Add successful results
    builder.create_success_row(
        sample_example_id="sample_1",
        original_id="id1",
        outcomes={"task1": "yes", "task2": "good"},
        llm_raw_response_content="Response 1",
        llm_prompt_tokens=100,
        llm_completion_tokens=50,
        llm_total_tokens=150,
        llm_call_duration_seconds=2.5,
    )

    builder.create_success_row(
        sample_example_id="sample_2",
        original_id="id2",
        outcomes={"task1": "no", "task2": "bad"},
        llm_raw_response_content="Response 2",
        llm_prompt_tokens=120,
        llm_completion_tokens=60,
        llm_total_tokens=180,
        llm_call_duration_seconds=3.0,
    )

    # Add partial result
    builder.create_partial_row(
        sample_example_id="sample_3",
        original_id="id3",
        outcomes={"task1": "yes"},  # Only partial outcomes
        error_message="Could not parse task2",
        llm_raw_response_content="Response 3",
        llm_prompt_tokens=110,
        llm_completion_tokens=55,
        llm_total_tokens=165,
        llm_call_duration_seconds=2.8,
    )

    # Add error results
    builder.create_llm_error_row(
        sample_example_id="sample_4",
        original_id="id4",
        error=Exception("LLM API failed"),
    )

    return builder.complete()


@pytest.fixture
def sample_judge_results(base_judge_results_builder) -> JudgeResults:
    """Create complete sample JudgeResults for testing.

    Args:
        base_judge_results_builder: Base builder from this conftest.

    Returns:
        JudgeResults: A complete sample JudgeResults for testing.
    """
    builder = base_judge_results_builder

    # Add successful results
    builder.create_success_row(
        sample_example_id="sample_1",
        original_id="id1",
        outcomes={"sentiment": "positive"},
        llm_raw_response_content="Response 1",
        llm_prompt_tokens=100,
        llm_completion_tokens=50,
        llm_total_tokens=150,
        llm_call_duration_seconds=2.5,
    )

    builder.create_success_row(
        sample_example_id="sample_2",
        original_id="id2",
        outcomes={"sentiment": "negative"},
        llm_raw_response_content="Response 2",
        llm_prompt_tokens=120,
        llm_completion_tokens=60,
        llm_total_tokens=180,
        llm_call_duration_seconds=3.0,
    )

    # Add partial result
    builder.create_partial_row(
        sample_example_id="sample_3",
        original_id="id3",
        outcomes={},  # Empty outcomes for partial result
        error_message="Could not parse response",
        llm_raw_response_content="Response 3",
        llm_prompt_tokens=110,
        llm_completion_tokens=55,
        llm_total_tokens=165,
        llm_call_duration_seconds=2.8,
    )

    return builder.complete()


# ==== MOCK FIXTURES ====


@pytest.fixture
def mock_llm_client() -> Mock:
    """Provides a mock LLM client for testing.

    Returns:
        Mock: A mock LLM client configured for testing.
    """
    client = Mock(spec=LLMClient)
    client.enum_value = LLMClientEnum.OPENAI
    return client


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Provides a mock LLM response for testing.

    Returns:
        LLMResponse: A mock LLM response with structured output.
    """
    return LLMResponse(
        provider=LLMClientEnum.OPENAI,
        model="gpt-4",
        messages=[Message(role=RoleEnum.ASSISTANT, content="positive")],
        usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# ==== ASYNC MOCK FIXTURES ====


@pytest.fixture
def mock_async_llm_client() -> Mock:
    """Provides a mock async LLM client for testing.

    Returns:
        Mock: A mock async LLM client configured for testing.
    """
    client = Mock(spec=AsyncLLMClient)
    client.enum_value = AsyncLLMClientEnum.OPENAI
    return client


@pytest.fixture
def mock_async_llm_response() -> LLMResponse:
    """Provides a mock LLM response for testing.

    Returns:
        LLMResponse: A mock LLM response with structured output.
    """
    return LLMResponse(
        provider=AsyncLLMClientEnum.OPENAI,
        model="gpt-4",
        messages=[Message(role=RoleEnum.ASSISTANT, content="positive")],
        usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
