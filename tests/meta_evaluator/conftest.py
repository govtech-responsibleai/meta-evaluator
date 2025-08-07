"""Fixtures for MetaEvaluator tests.

This conftest provides fixtures for MetaEvaluator testing, including client registry management,
meta_evaluator instances with different configurations, and orchestration-level data/results.
Higher-level fixtures for eval tasks, eval data, and logger are available from tests/conftest.py.
Specialized fixtures for data, judges, scorers, etc. are available from their respective conftest files.
"""

import logging
import pytest
from unittest.mock import MagicMock, Mock
from datetime import datetime

from meta_evaluator.meta_evaluator.scoring import ScoringMixin
from meta_evaluator.llm_client import LLMClientEnum
from meta_evaluator.llm_client.openai_client import OpenAIClient
from meta_evaluator.llm_client.azureopenai_client import AzureOpenAIClient
from meta_evaluator.llm_client.serialization import (
    OpenAISerializedState,
    AzureOpenAISerializedState,
)
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.results import JudgeResults, HumanAnnotationResults
from meta_evaluator.judge.judge import Judge
from meta_evaluator.data import SampleEvalData
import polars as pl


# ==== ENVIRONMENT FIXTURES ====


@pytest.fixture
def clean_environment(monkeypatch):
    """Provides a clean environment without any API key environment variables.

    Args:
        monkeypatch: pytest fixture for environment variable manipulation.
    """
    # Remove all API key related environment variables
    env_vars_to_remove = [
        "OPENAI_API_KEY",
        "OPENAI_DEFAULT_MODEL",
        "OPENAI_DEFAULT_EMBEDDING_MODEL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEFAULT_MODEL",
        "AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL",
    ]
    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def openai_environment(monkeypatch):
    """Provides environment with all OpenAI variables set.

    Args:
        monkeypatch: pytest fixture for environment variable manipulation.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4")
    monkeypatch.setenv("OPENAI_DEFAULT_EMBEDDING_MODEL", "text-embedding-3-large")


@pytest.fixture
def azure_openai_environment(monkeypatch):
    """Provides environment with all Azure OpenAI variables set.

    Args:
        monkeypatch: pytest fixture for environment variable manipulation.
    """
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4")
    monkeypatch.setenv("AZURE_OPENAI_DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002")


# ==== MOCK CLIENT FIXTURES ====


# ==== META EVALUATOR INSTANCE FIXTURES ====


@pytest.fixture
def meta_evaluator_with_task(meta_evaluator, basic_eval_task):
    """Provides a MetaEvaluator with eval_task set.

    Args:
        meta_evaluator: The MetaEvaluator instance from main conftest.
        basic_eval_task: The basic evaluation task from main conftest.

    Returns:
        MetaEvaluator: The modified MetaEvaluator instance.
    """
    meta_evaluator.add_eval_task(basic_eval_task)
    return meta_evaluator


@pytest.fixture
def meta_evaluator_with_data(meta_evaluator, sample_eval_data):
    """Provides a MetaEvaluator with data set.

    Args:
        meta_evaluator: The MetaEvaluator instance from main conftest.
        sample_eval_data: The sample evaluation data from main conftest.

    Returns:
        MetaEvaluator: The modified MetaEvaluator instance.
    """
    meta_evaluator.add_data(sample_eval_data)
    return meta_evaluator


@pytest.fixture
def meta_evaluator_with_clients(
    meta_evaluator, mock_openai_client, mock_azure_openai_client
):
    """Provides a MetaEvaluator with mock clients configured.

    Args:
        meta_evaluator: The MetaEvaluator instance from main conftest.
        mock_openai_client: Mock OpenAI client fixture.
        mock_azure_openai_client: Mock Azure OpenAI client fixture.

    Returns:
        MetaEvaluator: The modified MetaEvaluator instance.
    """
    meta_evaluator.client_registry = {
        LLMClientEnum.OPENAI: mock_openai_client,
        LLMClientEnum.AZURE_OPENAI: mock_azure_openai_client,
    }
    return meta_evaluator


@pytest.fixture
def meta_evaluator_with_judges_and_data(
    meta_evaluator, sample_prompt, sample_eval_data, mock_openai_client
):
    """Provides a MetaEvaluator with mock judges and data configured for integration testing.

    Args:
        meta_evaluator: The MetaEvaluator instance from main conftest.
        sample_prompt: Sample prompt from main conftest.
        sample_eval_data: Sample evaluation data from main conftest.
        mock_openai_client: Mock OpenAI client fixture.

    Returns:
        MetaEvaluator: The modified MetaEvaluator instance with configured judges and data.
    """
    # Create a custom eval_task that matches the sample_eval_data columns
    eval_task = EvalTask(
        task_schemas={"sentiment": ["positive", "negative", "neutral"]},
        prompt_columns=["question"],  # matches sample_eval_data
        response_columns=["answer"],  # matches sample_eval_data
        answering_method="structured",
    )

    meta_evaluator.add_eval_task(eval_task)
    meta_evaluator.add_data(sample_eval_data)

    # Configure LLM client registry
    meta_evaluator.client_registry = {LLMClientEnum.OPENAI: mock_openai_client}

    # Create mock judges
    mock_judge1 = Mock(spec=Judge)
    mock_judge1.llm_client_enum = LLMClientEnum.OPENAI
    mock_results1 = Mock(spec=JudgeResults)
    mock_results1.save_state = Mock()
    mock_results1.succeeded_count = 2
    mock_results1.total_count = 3
    mock_judge1.evaluate_eval_data.return_value = mock_results1

    mock_judge2 = Mock(spec=Judge)
    mock_judge2.llm_client_enum = LLMClientEnum.OPENAI
    mock_results2 = Mock(spec=JudgeResults)
    mock_results2.save_state = Mock()
    mock_results2.succeeded_count = 1
    mock_results2.total_count = 3
    mock_judge2.evaluate_eval_data.return_value = mock_results2

    # Add judges to the meta_evaluator
    meta_evaluator.judge_registry = {"judge1": mock_judge1, "judge2": mock_judge2}

    return meta_evaluator


# ==== ORCHESTRATION DATA FIXTURES ====


@pytest.fixture
def metaevaluator_test_eval_data():
    """Provides evaluation data specifically configured for meta_evaluator orchestration testing.

    Returns:
        SampleEvalData: Sample evaluation data with columns matching meta_evaluator test expectations.
    """
    test_df = pl.DataFrame(
        {
            "sample_id": ["1", "2", "3"],
            "question": [
                "What is 2+2?",
                "What is the capital of France?",
                "Is this statement positive?",
            ],
            "answer": ["4", "Paris", "Yes, it's positive"],
            "category": ["math", "geography", "sentiment"],
            "difficulty": ["easy", "medium", "easy"],
        }
    )

    return SampleEvalData(
        name="metaevaluator_test_data",
        data=test_df,
        id_column="sample_id",
        sample_name="MetaEvaluator Test Sample",
        stratification_columns=["category"],
        sample_percentage=1.0,  # Use all data for testing
        seed=42,
        sampling_method="stratified_by_columns",
    )


# ==== MOCK RESULTS FIXTURES ====


@pytest.fixture
def mock_judge_results():
    """Provides mock JudgeResults for meta_evaluator testing.

    Returns:
        Mock: A mock JudgeResults object configured for testing.
    """
    judge_results = Mock(spec=JudgeResults)
    judge_results.judge_id = "test_judge"
    judge_results.run_id = "test_run"
    judge_results.task_schemas = {
        "sentiment": ["positive", "negative", "neutral"],
        "quality": ["high", "medium", "low"],
    }
    judge_results.get_successful_results.return_value = pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
        }
    )
    judge_results.succeeded_count = 3
    judge_results.total_count = 3
    return judge_results


@pytest.fixture
def mock_human_results():
    """Provides mock HumanAnnotationResults for meta_evaluator testing.

    Returns:
        Mock: A mock HumanAnnotationResults object configured for testing.
    """
    human_results = Mock(spec=HumanAnnotationResults)
    human_results.annotator_id = "test_annotator"
    human_results.run_id = "test_annotation_run"
    human_results.get_successful_results.return_value = pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "sentiment": ["positive", "negative", "positive"],
            "quality": ["high", "low", "medium"],
        }
    )
    return human_results


# ==== SCORING-SPECIFIC FIXTURES ====


class MockMetaEvaluator(ScoringMixin):
    """Mock MetaEvaluator class for testing ScoringMixin."""

    def __init__(self, scores_dir=None):
        """Initialize the mock evaluator."""
        self.project_dir = "/mock/project/dir"
        self.paths = Mock()
        self.paths.results = Mock()
        self.paths.annotations = Mock()
        self.paths.scores = scores_dir or Mock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")


@pytest.fixture
def mock_evaluator(tmp_path):
    """Create a mock evaluator with ScoringMixin.

    Returns:
        MockMetaEvaluator: A mock evaluator with ScoringMixin for scoring tests.
    """
    # Use a temporary directory for scores to avoid creating Mock directories
    scores_dir = tmp_path / "scores"
    return MockMetaEvaluator(scores_dir=str(scores_dir))


@pytest.fixture
def judge_results_1():
    """Create first judge results for scoring comparisons.

    Returns:
        Mock: A mock judge results object with multi-task schemas.
    """
    judge_results = Mock(spec=JudgeResults)
    judge_results.judge_id = "judge_1"
    judge_results.run_id = "run_1"
    # Support both classification and text tasks
    judge_results.task_schemas = {
        "task1": ["A", "B", "C"],
        "task2": None,
        "safety": ["SAFE", "UNSAFE"],
    }
    judge_results.get_successful_results.return_value = pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4"],
            "task1": ["A", "B", "A", "C"],
            "task2": ["text1", "text2", "text3", "text4"],
            "safety": ["SAFE", "UNSAFE", "SAFE", "UNSAFE"],
        }
    )
    return judge_results


@pytest.fixture
def judge_results_2():
    """Create second judge results for scoring comparisons.

    Returns:
        Mock: A mock judge results object with multi-task schemas.
    """
    judge_results = Mock(spec=JudgeResults)
    judge_results.judge_id = "judge_2"
    judge_results.run_id = "run_2"
    # Support both classification and text tasks
    judge_results.task_schemas = {
        "task1": ["A", "B", "C"],
        "task2": None,
        "safety": ["SAFE", "UNSAFE"],
    }
    judge_results.get_successful_results.return_value = pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4"],
            "task1": ["A", "B", "A", "C"],
            "task2": ["text1", "text2", "text3", "text4"],
            "safety": ["SAFE", "SAFE", "SAFE", "SAFE"],
        }
    )
    return judge_results


@pytest.fixture
def judge_results_dict(judge_results_1, judge_results_2):
    """Create a dictionary of judge results for scoring comparison tests.

    Returns:
        dict: A dictionary of judge results.
    """
    return {"run_1": judge_results_1, "run_2": judge_results_2}


@pytest.fixture
def human_results_1():
    """Create first human results for scoring comparisons.

    Returns:
        Mock: A mock human results object.
    """
    human_results = Mock(spec=HumanAnnotationResults)
    human_results.annotator_id = "human_1"
    human_results.run_id = "run_1"
    human_results.get_successful_results.return_value = pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4"],
            "task1": ["A", "B", "B", "C"],
            "task2": ["text1", "text2_modified", "text3", "text4"],
            "safety": ["SAFE", "UNSAFE", "SAFE", "SAFE"],
        }
    )
    return human_results


@pytest.fixture
def human_results_2():
    """Create second human results for scoring comparisons.

    Returns:
        Mock: A mock human results object.
    """
    human_results = Mock(spec=HumanAnnotationResults)
    human_results.annotator_id = "human_2"
    human_results.run_id = "run_2"
    human_results.get_successful_results.return_value = pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4"],
            "task1": ["A", "A", "A", "C"],
            "task2": ["text1", "text2", "text3_modified", "text4"],
            "safety": ["SAFE", "SAFE", "UNSAFE", "UNSAFE"],
        }
    )
    return human_results


@pytest.fixture
def human_results_dict(human_results_1, human_results_2):
    """Create a dictionary of human results for scoring comparison tests.

    Returns:
        dict: A dictionary of human results.
    """
    return {"run_1": human_results_1, "run_2": human_results_2}


@pytest.fixture
def human_results(human_results_dict):
    """Human results dictionary for tests that expect multiple humans.

    Returns:
        dict: A dictionary of human results.
    """
    return human_results_dict


@pytest.fixture
def completed_judge_results():
    """Fixture for creating completed JudgeResults for file system tests.

    Returns:
        JudgeResults: A completed JudgeResults instance for testing.
    """
    from meta_evaluator.results import JudgeResultsBuilder
    from meta_evaluator.llm_client import LLMClientEnum

    builder = JudgeResultsBuilder(
        run_id="test_run",
        judge_id="test_judge",
        llm_client_enum=LLMClientEnum.OPENAI,
        model_used="gpt-4",
        task_schemas={"sentiment": ["positive", "negative"]},
        expected_ids=["id1"],
    )
    builder.create_success_row(
        sample_example_id="test_1",
        original_id="id1",
        outcomes={"sentiment": "positive"},
        llm_raw_response_content="positive",
        llm_prompt_tokens=10,
        llm_completion_tokens=5,
        llm_total_tokens=15,
        llm_call_duration_seconds=1.0,
    )
    return builder.complete()


@pytest.fixture
def completed_human_results():
    """Fixture for creating completed HumanAnnotationResults for file system tests.

    Returns:
        HumanAnnotationResults: A completed HumanAnnotationResults instance for testing.
    """
    from meta_evaluator.results import HumanAnnotationResultsBuilder

    builder = HumanAnnotationResultsBuilder(
        run_id="test_annotation_run",
        annotator_id="test_annotator",
        task_schemas={"accuracy": ["accurate", "inaccurate"]},
        expected_ids=["id1"],
    )
    builder.create_success_row(
        sample_example_id="test_1",
        original_id="id1",
        outcomes={"accuracy": "accurate"},
        annotation_timestamp=datetime.now(),
    )
    return builder.complete()


# ==== HELPER FUNCTIONS FOR DYNAMIC CLIENT CREATION ====


def create_mock_openai_client(**config_overrides):
    """Helper function to create a customized mock OpenAI client.

    Args:
        **config_overrides: Override default configuration values.

    Returns:
        MagicMock: A mock OpenAI client with custom configuration.
    """
    mock_client = MagicMock(spec=OpenAIClient)
    mock_config = MagicMock()
    mock_config.default_model = config_overrides.get("default_model", "gpt-4")
    mock_config.default_embedding_model = config_overrides.get(
        "default_embedding_model", "text-embedding-3-large"
    )
    mock_config.supports_structured_output = config_overrides.get(
        "supports_structured_output", True
    )
    mock_config.supports_logprobs = config_overrides.get("supports_logprobs", True)

    # Mock the serialize method to return a proper OpenAISerializedState
    serialized_state = OpenAISerializedState(
        default_model=mock_config.default_model,
        default_embedding_model=mock_config.default_embedding_model,
        supports_structured_output=mock_config.supports_structured_output,
        supports_logprobs=mock_config.supports_logprobs,
        supports_instructor=True,
    )
    mock_config.serialize.return_value = serialized_state

    mock_client.config = mock_config
    return mock_client


def create_mock_azure_openai_client(**config_overrides):
    """Helper function to create a customized mock Azure OpenAI client.

    Args:
        **config_overrides: Override default configuration values.

    Returns:
        MagicMock: A mock Azure OpenAI client with custom configuration.
    """
    mock_client = MagicMock(spec=AzureOpenAIClient)
    mock_config = MagicMock()
    mock_config.endpoint = config_overrides.get(
        "endpoint", "https://test.openai.azure.com"
    )
    mock_config.api_version = config_overrides.get("api_version", "2024-02-15-preview")
    mock_config.default_model = config_overrides.get("default_model", "gpt-4")
    mock_config.default_embedding_model = config_overrides.get(
        "default_embedding_model", "text-embedding-ada-002"
    )
    mock_config.supports_structured_output = config_overrides.get(
        "supports_structured_output", True
    )
    mock_config.supports_logprobs = config_overrides.get("supports_logprobs", True)

    # Mock the serialize method to return a proper AzureOpenAISerializedState
    serialized_state = AzureOpenAISerializedState(
        endpoint=mock_config.endpoint,
        api_version=mock_config.api_version,
        default_model=mock_config.default_model,
        default_embedding_model=mock_config.default_embedding_model,
        supports_structured_output=mock_config.supports_structured_output,
        supports_logprobs=mock_config.supports_logprobs,
        supports_instructor=True,
    )
    mock_config.serialize.return_value = serialized_state

    mock_client.config = mock_config
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Create a properly mocked OpenAI client for testing.

    Returns:
        MagicMock: A mock OpenAI client with configured attributes.
    """
    return create_mock_openai_client()


@pytest.fixture
def mock_azure_openai_client():
    """Create a properly mocked Azure OpenAI client for testing.

    Returns:
        MagicMock: A mock Azure OpenAI client with configured attributes.
    """
    return create_mock_azure_openai_client(supports_structured_output=False)
