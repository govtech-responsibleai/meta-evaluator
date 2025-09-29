"""Fixtures for Judge and JudgeResults testing.

This conftest provides judge-specific fixtures used across judge test modules.
Common fixtures are inherited from the main conftest.py.
"""

from unittest.mock import Mock

import pytest

from meta_evaluator.common.models import Prompt
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.judge.judge import Judge
from meta_evaluator.judge.models import (
    LLMResponse,
    LLMUsage,
    Message,
    RoleEnum,
)
from meta_evaluator.results import JudgeResults, JudgeResultsBuilder

# ==== JUDGE FIXTURES ====


@pytest.fixture
def sentiment_judge_prompt() -> Prompt:
    """Provides a sentiment-specific prompt for judge testing (without template variables).

    Returns:
        Prompt: A sentiment-specific prompt for judge testing.
    """
    return Prompt(
        id="sentiment_prompt",
        prompt="Evaluate the sentiment of the given text.",
    )


@pytest.fixture
def template_sentiment_judge_prompt() -> Prompt:
    """Provides a sentiment-specific prompt with template variables for judge testing.

    Returns:
        Prompt: A sentiment-specific prompt with template variables for judge testing.
    """
    return Prompt(
        id="template_sentiment_prompt",
        prompt="Evaluate the sentiment of this text: {text}. Consider the response: {response}",
    )


@pytest.fixture
def template_basic_judge(basic_eval_task, template_sentiment_judge_prompt) -> Judge:
    """Provides a basic judge with template variables for testing.

    Args:
        basic_eval_task: Basic evaluation task from main conftest.
        template_sentiment_judge_prompt: Template-based sentiment prompt from this conftest.

    Returns:
        Judge: A basic judge with template variables for testing.
    """
    return Judge(
        id="test_basic_template_judge",
        eval_task=basic_eval_task,
        llm_client="openai",
        model="gpt-4",
        prompt=template_sentiment_judge_prompt,
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
        llm_client="openai",
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
        llm_client="openai",
        model="gpt-4",
        prompt=sentiment_judge_prompt,
    )


@pytest.fixture
def template_xml_judge(xml_eval_task, template_sentiment_judge_prompt) -> Judge:
    """Provides an XML-based judge with template variables for testing.

    Args:
        xml_eval_task: XML-based evaluation task from main conftest.
        template_sentiment_judge_prompt: Template-based sentiment prompt.

    Returns:
        Judge: An XML-based judge with template variables.
    """
    return Judge(
        id="template_xml_judge",
        eval_task=xml_eval_task,
        llm_client="openai",
        model="gpt-4",
        prompt=template_sentiment_judge_prompt,
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
        llm_client="openai",
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
        llm_client="openai",
        model_used="gpt-4",
        task_schemas=single_task_schemas,
        expected_ids=["id1", "id2", "id3"],
        required_tasks=["sentiment"],
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
        llm_client="openai",
        model_used="gpt-4",
        task_schemas={
            "sentiment": ["positive", "negative", "neutral"],
            "toxicity": ["toxic", "non_toxic"],
        },
        expected_ids=["id1", "id2", "id3", "id4", "id5"],
        required_tasks=["sentiment", "toxicity"],
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
        llm_client="anthropic",
        model_used="claude-3",
        task_schemas={"sentiment": ["positive", "negative"]},
        expected_ids=["id1"],
        required_tasks=["sentiment"],
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
        llm_client="openai",
        model_used="gpt-4",
        task_schemas=serialization_task_schemas,
        expected_ids=["id1", "id2", "id3", "id4"],
        required_tasks=["task1", "task2"],
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


# ==== MOCK LITELLM FIXTURES ====


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Provides a mock LLM response for testing.

    Returns:
        LLMResponse: A mock LLM response with structured output.
    """
    return LLMResponse(
        llm_client="openai",
        model="gpt-4",
        messages=[Message(role=RoleEnum.ASSISTANT, content="positive")],
        usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
def mock_litellm_response():
    """Create a mock litellm response.

    Returns:
        Mock: A mock litellm response.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"sentiment": "positive"}'
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    return mock_response


@pytest.fixture
def mock_xml_litellm_response():
    """Create a mock litellm response with XML content.

    Returns:
        Mock: A mock litellm response with XML content.
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "<sentiment>positive</sentiment>"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    return mock_response


@pytest.fixture
def sync_test_builder(single_task_schemas) -> JudgeResultsBuilder:
    """Create a builder for sync evaluation tests with matching basic_eval_data IDs.

    Args:
        single_task_schemas: Single task schemas from main conftest.

    Returns:
        JudgeResultsBuilder: A builder instance for sync evaluation testing.
    """
    return JudgeResultsBuilder(
        run_id="sync_test_run",
        judge_id="sync_test_judge",
        llm_client="openai",
        model_used="gpt-4",
        task_schemas=single_task_schemas,
        expected_ids=["1", "2", "3"],  # Match basic_eval_data IDs
        required_tasks=["sentiment"],
        is_sampled_run=False,
    )


@pytest.fixture
def single_row_test_builder(single_task_schemas) -> JudgeResultsBuilder:
    """Create a builder for single row sync evaluation tests.

    Args:
        single_task_schemas: Single task schemas from main conftest.

    Returns:
        JudgeResultsBuilder: A builder instance for single row testing.
    """
    return JudgeResultsBuilder(
        run_id="single_row_test_run",
        judge_id="single_row_test_judge",
        llm_client="openai",
        model_used="gpt-4",
        task_schemas=single_task_schemas,
        expected_ids=["1"],  # Single row test
        required_tasks=["sentiment"],
        is_sampled_run=False,
    )


# ==== FALLBACK FIXTURES ====


@pytest.fixture
def fallback_enabled_task():
    """Create an EvalTask with fallback enabled.

    Returns:
        EvalTask: An EvalTask with fallback enabled.
    """
    return EvalTask(
        task_schemas={"sentiment": ["positive", "negative", "neutral"]},
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
        structured_outputs_fallback=True,
    )


@pytest.fixture
def fallback_disabled_task():
    """Create an EvalTask with fallback disabled.

    Returns:
        EvalTask: An EvalTask with fallback disabled.
    """
    return EvalTask(
        task_schemas={"sentiment": ["positive", "negative", "neutral"]},
        prompt_columns=["text"],
        response_columns=["response"],
        answering_method="structured",
        structured_outputs_fallback=False,
    )


@pytest.fixture
def fallback_enabled_judge(fallback_enabled_task, template_sentiment_judge_prompt):
    """Create a judge with fallback enabled.

    Returns:
        Judge: A judge with fallback enabled.
    """
    return Judge(
        id="fallback_test_judge",
        eval_task=fallback_enabled_task,
        llm_client="openai",
        model="gpt-4",
        prompt=template_sentiment_judge_prompt,
    )


@pytest.fixture
def fallback_disabled_judge(fallback_disabled_task, sentiment_judge_prompt):
    """Create a judge with fallback disabled.

    Returns:
        Judge: A judge with fallback disabled.
    """
    return Judge(
        id="no_fallback_test_judge",
        eval_task=fallback_disabled_task,
        llm_client="openai",
        model="gpt-4",
        prompt=sentiment_judge_prompt,
    )
