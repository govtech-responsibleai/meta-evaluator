"""Fixtures for MetaEvaluator tests.

This conftest provides fixtures for MetaEvaluator testing, including client registry management,
meta_evaluator instances with different configurations, and orchestration-level data/results.
Higher-level fixtures for eval tasks, eval data, and logger are available from tests/conftest.py.
Specialized fixtures for data, judges, scorers, etc. are available from their respective conftest files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

from meta_evaluator.data import SampleEvalData
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.judge.judge import Judge
from meta_evaluator.meta_evaluator.scoring import ScoringMixin
from meta_evaluator.results import (
    HumanAnnotationResults,
    HumanAnnotationResultsBuilder,
    JudgeResults,
    JudgeResultsBuilder,
)
from meta_evaluator.scores import (
    AccuracyScorer,
    AltTestScorer,
    BaseScoringResult,
    CohensKappaScorer,
    MetricConfig,
)
from meta_evaluator.scores.enums import TaskAggregationMode

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
def meta_evaluator_with_judges_and_data(
    meta_evaluator,
    sample_prompt,
    sample_eval_data,
):
    """Provides a MetaEvaluator with mock judges and data configured for integration testing.

    Args:
        meta_evaluator: The MetaEvaluator instance from main conftest.
        sample_prompt: Sample prompt from main conftest.
        sample_eval_data: Sample evaluation data from main conftest.
        mock_openai_client: Mock OpenAI client fixture.
        mock_async_openai_client: Mock async OpenAI client fixture.

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

    # Create mock judges
    mock_judge1 = Mock(spec=Judge)
    mock_judge1.llm_client = "openai"
    mock_results1 = Mock(spec=JudgeResults)
    mock_results1.save_state = Mock()
    mock_results1.succeeded_count = 2
    mock_results1.total_count = 3
    mock_judge1.evaluate_eval_data.return_value = mock_results1

    mock_judge2 = Mock(spec=Judge)
    mock_judge2.llm_client = "openai"
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
    successful_results_df = pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "sentiment": ["positive", "negative", "neutral"],
            "quality": ["high", "medium", "low"],
        }
    )
    judge_results.get_successful_results.return_value = successful_results_df
    # Add results_data attribute needed by _validate_judge_success_rate
    judge_results.results_data = successful_results_df
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
    successful_results_df = pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4"],
            "task1": ["A", "B", "A", "C"],
            "task2": ["text1", "text2", "text3", "text4"],
            "safety": ["SAFE", "UNSAFE", "SAFE", "UNSAFE"],
        }
    )
    judge_results.get_successful_results.return_value = successful_results_df
    # Add results_data attribute needed by _validate_judge_success_rate
    judge_results.results_data = successful_results_df
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
    successful_results_df = pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4"],
            "task1": ["A", "B", "A", "C"],
            "task2": ["text1", "text2", "text3", "text4"],
            "safety": ["SAFE", "SAFE", "SAFE", "SAFE"],
        }
    )
    judge_results.get_successful_results.return_value = successful_results_df
    # Add results_data attribute needed by _validate_judge_success_rate
    judge_results.results_data = successful_results_df
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
def basic_judge(basic_eval_task, sample_prompt):
    """Provides a basic judge configuration for testing.

    Args:
        basic_eval_task: Basic evaluation task fixture.
        sample_prompt: Sample prompt fixture.

    Returns:
        Judge: A basic judge instance for testing.
    """
    return Judge(
        id="test_judge",
        eval_task=basic_eval_task,
        llm_client="openai",
        model="gpt-4",
        prompt=sample_prompt,
    )


@pytest.fixture
def xml_judge(xml_eval_task, sample_prompt):
    """Provides an XML-based judge configuration for testing.

    Args:
        xml_eval_task: XML evaluation task fixture.
        sample_prompt: Sample prompt fixture.

    Returns:
        Judge: An XML judge instance for testing.
    """
    return Judge(
        id="test_xml_judge",
        eval_task=xml_eval_task,
        llm_client="openai",
        model="gpt-4",
        prompt=sample_prompt,
    )


@pytest.fixture
def completed_judge_results():
    """Fixture for creating completed JudgeResults for file system tests.

    Returns:
        JudgeResults: A completed JudgeResults instance for testing.
    """
    builder = JudgeResultsBuilder(
        run_id="test_run",
        judge_id="test_judge",
        llm_client="openai",
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


# ==== DUPLICATE PREVENTION TEST FIXTURES ====


@pytest.fixture
def mock_judge_state_template():
    """Provides a template for creating mock judge result state files.

    Returns:
        dict: Template dictionary for judge result state JSON.
    """
    return {
        "task_schemas": {"sentiment": ["positive", "negative"]},
        "llm_client": "openai",
        "model_used": "gpt-4",
        "timestamp_local": "2024-01-01T00:00:00",
        "total_count": 10,
        "succeeded_count": 8,
        "skipped_count": 0,
        "partial_count": 1,
        "llm_error_count": 1,
        "parsing_error_count": 0,
        "other_error_count": 0,
        "is_sampled_run": False,
        "data_format": "json",
    }


def create_mock_judge_state_file(
    results_dir: Path, judge_id: str, run_id: str, state_template: dict
) -> Path:
    """Helper function to create a mock judge results state file.

    Args:
        results_dir: Directory to create the file in.
        judge_id: Judge ID for the state file.
        run_id: Run ID for the state file.
        state_template: Template dictionary for the state.

    Returns:
        Path: Path to the created state file.
    """
    state_file = results_dir / f"{run_id}_{judge_id}_state.json"

    # Create the full state from template
    mock_state = state_template.copy()
    mock_state.update(
        {
            "run_id": run_id,
            "judge_id": judge_id,
            "data_file": f"{run_id}_{judge_id}_results.json",
        }
    )

    with open(state_file, "w") as f:
        json.dump(mock_state, f)

    return state_file


@pytest.fixture
def mock_human_state_template():
    """Provides a template for creating mock human annotation metadata files.

    Returns:
        dict: Template dictionary for human annotation metadata JSON.
    """
    return {
        "task_schemas": {"sentiment": ["positive", "negative"]},
        "timestamp_local": "2024-01-01T00:00:00",
        "total_count": 10,
        "succeeded_count": 9,
        "skipped_count": 0,
        "partial_count": 0,
        "annotation_error_count": 1,
        "other_error_count": 0,
        "data_format": "json",
    }


def create_mock_human_metadata_file(
    annotations_dir: Path, annotator_id: str, run_id: str, metadata_template: dict
) -> Path:
    """Helper function to create a mock human annotation metadata file.

    Args:
        annotations_dir: Directory to create the file in.
        annotator_id: Annotator ID for the metadata file.
        run_id: Run ID for the metadata file.
        metadata_template: Template dictionary for the metadata.

    Returns:
        Path: Path to the created metadata file.
    """
    metadata_file = annotations_dir / f"{run_id}_{annotator_id}_metadata.json"

    # Create the full metadata from template
    mock_metadata = metadata_template.copy()
    mock_metadata.update(
        {
            "run_id": run_id,
            "annotator_id": annotator_id,
            "data_file": f"{run_id}_{annotator_id}_annotations.json",
        }
    )

    with open(metadata_file, "w") as f:
        json.dump(mock_metadata, f)

    return metadata_file


# ==== SCORER CONFIGURATION FIXTURES ====


@pytest.fixture
def accuracy_scorer_config():
    """Create AccuracyScorer with MetricConfig for testing.

    Returns:
        tuple: (scorer_instance, MetricConfig)
    """
    scorer = AccuracyScorer()
    config = MetricConfig(
        scorer=scorer, task_names=["task1"], aggregation_name="single"
    )
    return scorer, config


@pytest.fixture
def cohens_kappa_scorer_config():
    """Create CohensKappaScorer with MetricConfig for testing.

    Returns:
        tuple: (scorer_instance, MetricConfig)
    """
    scorer = CohensKappaScorer()
    config = MetricConfig(
        scorer=scorer, task_names=["task1"], aggregation_name="single"
    )
    return scorer, config


@pytest.fixture
def alt_test_scorer_config():
    """Create AltTestScorer with MetricConfig for testing.

    Returns:
        tuple: (scorer_instance, MetricConfig)
    """
    scorer = AltTestScorer()
    scorer.min_instances_per_human = 1
    scorer.min_humans_per_instance = 1
    config = MetricConfig(
        scorer=scorer, task_names=["task1"], aggregation_name="single"
    )
    return scorer, config


@pytest.fixture
def multi_aggregation_configs():
    """Create same scorer with different aggregation modes for testing.

    Returns:
        dict: Dictionary mapping unique names to (config, expected_results) tuples
    """
    configs = {}

    # Single task config
    scorer1 = AccuracyScorer()
    config1 = MetricConfig(
        scorer=scorer1,
        task_names=["task1"],
        aggregation_name="single",
    )
    result1 = BaseScoringResult(
        scorer_name="accuracy",
        task_name="task1",
        judge_id="judge_1",
        scores={"accuracy": 0.8},
        metadata={},
        aggregation_mode=TaskAggregationMode.SINGLE,
        num_comparisons=10,
        failed_comparisons=0,
    )
    configs[config1.get_unique_name()] = (config1, [result1])

    # Multi-task config
    scorer2 = AccuracyScorer()
    config2 = MetricConfig(
        scorer=scorer2,
        task_names=["task1", "task2"],
        aggregation_name="multitask",
    )
    result2 = BaseScoringResult(
        scorer_name="accuracy",
        task_name="2_tasks_avg",
        judge_id="judge_1",
        scores={"accuracy": 0.85},
        metadata={},
        aggregation_mode=TaskAggregationMode.MULTITASK,
        num_comparisons=20,
        failed_comparisons=0,
    )
    configs[config2.get_unique_name()] = (config2, [result2])

    # Multilabel config
    scorer3 = AccuracyScorer()
    config3 = MetricConfig(
        scorer=scorer3,
        task_names=["task1", "task2"],
        aggregation_name="multilabel",
    )
    result3 = BaseScoringResult(
        scorer_name="accuracy",
        task_name="multilabel_2_tasks",
        judge_id="judge_1",
        scores={"accuracy": 0.9},
        metadata={},
        aggregation_mode=TaskAggregationMode.MULTILABEL,
        num_comparisons=10,
        failed_comparisons=0,
    )
    configs[config3.get_unique_name()] = (config3, [result3])

    return configs


@pytest.fixture
def different_scorer_configs():
    """Create different scorer configurations for testing.

    Returns:
        dict: Dictionary mapping unique names to (config, expected_results) tuples
    """
    configs = {}

    # Accuracy scorer config
    accuracy_scorer = AccuracyScorer()
    accuracy_config = MetricConfig(
        scorer=accuracy_scorer,
        task_names=["task1"],
        aggregation_name="single",
    )
    accuracy_results = [
        BaseScoringResult(
            scorer_name="accuracy",
            task_name="task1",
            judge_id="judge_1",
            scores={"accuracy": 0.8},
            metadata={},
            aggregation_mode=TaskAggregationMode.SINGLE,
            num_comparisons=10,
            failed_comparisons=0,
        ),
        BaseScoringResult(
            scorer_name="accuracy",
            task_name="task1",
            judge_id="judge_2",
            scores={"accuracy": 0.9},
            metadata={},
            aggregation_mode=TaskAggregationMode.SINGLE,
            num_comparisons=10,
            failed_comparisons=0,
        ),
    ]
    configs["accuracy_single"] = (accuracy_config, accuracy_results)

    # Cohen's Kappa scorer config
    kappa_scorer = CohensKappaScorer()
    kappa_config = MetricConfig(
        scorer=kappa_scorer,
        task_names=["task1"],
        aggregation_name="single",
    )
    kappa_results = [
        BaseScoringResult(
            scorer_name="cohens_kappa",
            task_name="task1",
            judge_id="judge_1",
            scores={"kappa": 0.7},
            metadata={},
            aggregation_mode=TaskAggregationMode.SINGLE,
            num_comparisons=10,
            failed_comparisons=0,
        )
    ]
    configs["cohens_kappa_single"] = (kappa_config, kappa_results)

    # AltTest scorer config
    alt_test_scorer = AltTestScorer()
    alt_test_scorer.min_instances_per_human = 1
    alt_test_scorer.min_humans_per_instance = 1
    alt_test_config = MetricConfig(
        scorer=alt_test_scorer,
        task_names=["task1"],
        aggregation_name="single",
    )
    alt_test_results = [
        BaseScoringResult(
            scorer_name="alt_test",
            task_name="task1",
            judge_id="judge_1",
            scores={
                "winning_rate": {"0.10": 0.0, "0.20": 0.3333333, "0.30": 0.6666667},
                "advantage_probability": 0.6,
            },
            metadata={
                "scoring_function": "accuracy",
                "human_advantage_probabilities": {"human_1": (0.3, 0.7)},
            },
            aggregation_mode=TaskAggregationMode.SINGLE,
            num_comparisons=10,
            failed_comparisons=0,
        )
    ]
    configs["alt_test_single"] = (alt_test_config, alt_test_results)

    return configs
