"""Fixtures for scores package tests.

This conftest provides scorer instances, sample data, and results for testing
different scoring scenarios across all scorer types.
"""

import pytest
import polars as pl
from meta_evaluator.scores import (
    AccuracyScorer,
    TextSimilarityScorer,
    CohensKappaScorer,
)
from meta_evaluator.scores.metrics.agreement.alt_test import AltTestScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult


# ==== SCORER INSTANCE FIXTURES ====


@pytest.fixture
def accuracy_scorer():
    """Provides an AccuracyScorer instance for testing.

    Returns:
        AccuracyScorer: An AccuracyScorer instance.
    """
    return AccuracyScorer()


@pytest.fixture
def text_similarity_scorer():
    """Provides a TextSimilarityScorer instance for testing.

    Returns:
        TextSimilarityScorer: A TextSimilarityScorer instance.
    """
    return TextSimilarityScorer()


@pytest.fixture
def cohens_kappa_scorer():
    """Provides a CohensKappaScorer instance for testing.

    Returns:
        CohensKappaScorer: A CohensKappaScorer instance.
    """
    return CohensKappaScorer()


@pytest.fixture
def alt_test_scorer():
    """Provides an AltTestScorer instance for testing.

    Returns:
        AltTestScorer: An AltTestScorer instance with configured min_instances_per_human.
    """
    scorer = AltTestScorer()
    scorer.min_instances_per_human = 2
    return scorer


# ==== TASK SCHEMA FIXTURES ====


@pytest.fixture
def classification_task_schemas():
    """Provides classification task schemas for testing.

    Returns:
        dict: Classification task schemas.
    """
    return {
        "sentiment": ["positive", "negative", "neutral"],
        "safety": ["SAFE", "UNSAFE"],
        "toxicity": ["TOXIC", "NON_TOXIC"],
        "quality": ["high", "medium", "low"],
    }


@pytest.fixture
def single_classification_task_schema():
    """Provides a single classification task schema for testing.

    Returns:
        dict: Single classification task schema.
    """
    return {"safety": ["SAFE", "UNSAFE"]}


@pytest.fixture
def multi_classification_task_schema():
    """Provides multi-task classification schemas for testing.

    Returns:
        dict: Multi-task classification schemas.
    """
    return {
        "safety": ["SAFE", "UNSAFE"],
        "toxicity": ["TOXIC", "NON_TOXIC"],
    }


@pytest.fixture
def text_task_schema():
    """Provides text task schema for testing (free-form).

    Returns:
        dict: Text task schema with None value for free-form text.
    """
    return {"summary": None}


# ==== BASIC DATAFRAME FIXTURES ====


@pytest.fixture
def basic_judge_df():
    """Provides a basic judge DataFrame for classification testing.

    Returns:
        pl.DataFrame: A basic judge DataFrame with classification data.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "judge_id": ["judge_1", "judge_1", "judge_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": ["A", "B", "C"],
        }
    )


@pytest.fixture
def basic_human_df():
    """Provides a basic human DataFrame for classification testing.

    Returns:
        pl.DataFrame: A basic human DataFrame with classification data.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "annotator_id": ["human_1", "human_1", "human_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": ["A", "B", "C"],
        }
    )


@pytest.fixture
def multi_annotator_human_df():
    """Provides a human DataFrame with multiple annotators for testing.

    Returns:
        pl.DataFrame: A human DataFrame with multiple annotators showing disagreement.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "1", "2", "3"],
            "annotator_id": [
                "human_1",
                "human_1",
                "human_1",
                "human_2",
                "human_2",
                "human_2",
            ],
            "task_name": ["task1", "task1", "task1", "task1", "task1", "task1"],
            "task_value": ["A", "B", "C", "A", "A", "B"],  # human_2 disagrees on some
        }
    )


@pytest.fixture
def null_values_judge_df():
    """Provides a judge DataFrame with null values for testing.

    Returns:
        pl.DataFrame: A judge DataFrame with null task values.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "judge_id": ["judge_1", "judge_1", "judge_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": [None, None, None],
        }
    )


@pytest.fixture
def null_values_human_df():
    """Provides a human DataFrame with null values for testing.

    Returns:
        pl.DataFrame: A human DataFrame with null task values.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "annotator_id": ["human_1", "human_1", "human_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": [None, None, None],
        }
    )


# ==== TEXT-SPECIFIC DATAFRAME FIXTURES ====


@pytest.fixture
def text_judge_df():
    """Provides a judge DataFrame for text similarity testing.

    Returns:
        pl.DataFrame: A judge DataFrame with text data.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "judge_id": ["judge_1", "judge_1", "judge_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": ["hello world", "test case", "example text"],
        }
    )


@pytest.fixture
def text_human_df():
    """Provides a human DataFrame for text similarity testing.

    Returns:
        pl.DataFrame: A human DataFrame with text data.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "annotator_id": ["human_1", "human_1", "human_1"],
            "task_name": ["task1", "task1", "task1"],
            "task_value": ["hello world", "test case", "example text"],
        }
    )


# ==== SINGLE TASK FIXTURES (for AltTest and specific scoring) ====


@pytest.fixture
def single_task_judge_df():
    """Provides a single classification task judge DataFrame.

    Returns:
        pl.DataFrame: A judge DataFrame for single classification task.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4"],
            "judge_id": ["judge_1", "judge_1", "judge_1", "judge_1"],
            "task_name": ["safety", "safety", "safety", "safety"],
            "task_value": ["SAFE", "UNSAFE", "SAFE", "UNSAFE"],
        }
    )


@pytest.fixture
def single_task_human_df():
    """Provides a single classification task human DataFrame with multiple annotators.

    Returns:
        pl.DataFrame: A human DataFrame for single classification task with multiple annotators.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4", "1", "2", "3", "4"],
            "annotator_id": [
                "human_1",
                "human_1",
                "human_1",
                "human_1",
                "human_2",
                "human_2",
                "human_2",
                "human_2",
            ],
            "task_name": ["safety", "safety", "safety", "safety"] * 2,
            "task_value": [
                "SAFE",
                "UNSAFE",
                "SAFE",
                "SAFE",  # human1
                "SAFE",
                "SAFE",
                "UNSAFE",
                "UNSAFE",  # human2 - shows disagreement
            ],
        }
    )


# ==== MULTI-TASK FIXTURES ====


@pytest.fixture
def multi_task_judge_df():
    """Provides a multi-task classification judge DataFrame.

    Returns:
        pl.DataFrame: A judge DataFrame with multiple classification tasks.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4", "5", "6"] * 2,
            "judge_id": ["judge_1"] * 12,
            "task_name": ["safety"] * 6 + ["toxicity"] * 6,
            "task_value": [
                "SAFE",
                "UNSAFE",
                "UNSAFE",
                "SAFE",
                "SAFE",
                "UNSAFE",  # safety
                "NON_TOXIC",
                "TOXIC",
                "TOXIC",
                "NON_TOXIC",
                "NON_TOXIC",
                "TOXIC",  # toxicity
            ],
        }
    )


@pytest.fixture
def multi_task_human_df():
    """Provides a multi-task classification human DataFrame with multiple annotators.

    Returns:
        pl.DataFrame: A human DataFrame with multiple tasks and annotators.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4", "5", "6"] * 4,  # 2 tasks x 2 annotators
            "annotator_id": ["human_1"] * 12
            + ["human_2"] * 12,  # 2 annotators for both tasks
            "task_name": ["safety"] * 6
            + ["toxicity"] * 6
            + ["safety"] * 6
            + ["toxicity"] * 6,
            "task_value": [
                # human_1 safety
                "SAFE",
                "SAFE",
                "UNSAFE",
                "UNSAFE",
                "SAFE",
                "SAFE",
                # human_1 toxicity
                "NON_TOXIC",
                "NON_TOXIC",
                "TOXIC",
                "TOXIC",
                "NON_TOXIC",
                "NON_TOXIC",
                # human_2 safety
                "SAFE",
                "UNSAFE",
                "UNSAFE",
                "SAFE",
                "SAFE",
                "UNSAFE",
                # human_2 toxicity
                "NON_TOXIC",
                "TOXIC",
                "TOXIC",
                "NON_TOXIC",
                "NON_TOXIC",
                "TOXIC",
            ],
        }
    )


# ==== TEXT TASK FIXTURES ====


@pytest.fixture
def text_task_judge_df():
    """Provides a text task judge DataFrame.

    Returns:
        pl.DataFrame: A judge DataFrame for text tasks.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3"],
            "judge_id": ["judge_1", "judge_1", "judge_1"],
            "task_name": ["summary", "summary", "summary"],
            "task_value": ["Good", "Bad", "OK"],
        }
    )


@pytest.fixture
def text_task_human_df():
    """Provides a text task human DataFrame with multiple annotators.

    Returns:
        pl.DataFrame: A human DataFrame for text tasks with multiple annotators.
    """
    return pl.DataFrame(
        {
            "original_id": ["1", "2", "3", "4", "5", "6"] * 2,  # 2 annotators
            "annotator_id": ["human_1"] * 6 + ["human_2"] * 6,
            "task_name": ["summary"] * 12,
            "task_value": [
                # human_1
                "Good",
                "Bad",
                "OK",
                "Great",
                "Poor",
                "OK",
                # human_2
                "Good",
                "Bad",
                "OK",
                "Great",
                "Poor",
                "OK",
            ],
        }
    )


# ==== SAMPLE SCORING RESULTS FIXTURES ====


@pytest.fixture
def sample_accuracy_results():
    """Provides sample accuracy scoring results for aggregation testing.

    Returns:
        list: List of BaseScoringResult instances for accuracy scoring.
    """
    return [
        BaseScoringResult(
            scorer_name="accuracy",
            task_name="sentiment",
            judge_id="judge_1",
            score=0.85,
            metadata={"accuracy_method": "exact_match", "total_samples": 100},
        ),
        BaseScoringResult(
            scorer_name="accuracy",
            task_name="sentiment",
            judge_id="judge_2",
            score=0.78,
            metadata={"accuracy_method": "exact_match", "total_samples": 100},
        ),
    ]


@pytest.fixture
def sample_text_similarity_results():
    """Provides sample text similarity scoring results for aggregation testing.

    Returns:
        list: List of BaseScoringResult instances for text similarity scoring.
    """
    return [
        BaseScoringResult(
            scorer_name="text_similarity",
            task_name="summary",
            judge_id="judge_1",
            score=0.92,
            metadata={"similarity_method": "cosine", "total_comparisons": 50},
        ),
        BaseScoringResult(
            scorer_name="text_similarity",
            task_name="summary",
            judge_id="judge_2",
            score=0.88,
            metadata={"similarity_method": "cosine", "total_comparisons": 50},
        ),
    ]


@pytest.fixture
def sample_cohens_kappa_results():
    """Provides sample Cohen's Kappa scoring results for aggregation testing.

    Returns:
        list: List of BaseScoringResult instances for Cohen's Kappa scoring.
    """
    return [
        BaseScoringResult(
            scorer_name="cohens_kappa",
            task_name="safety",
            judge_id="judge_1",
            score=0.65,
            metadata={"agreement_level": "substantial", "total_annotations": 200},
        ),
        BaseScoringResult(
            scorer_name="cohens_kappa",
            task_name="safety",
            judge_id="judge_2",
            score=0.72,
            metadata={"agreement_level": "substantial", "total_annotations": 200},
        ),
    ]


@pytest.fixture
def sample_alt_test_results():
    """Provides sample AltTest scoring results for aggregation testing.

    Returns:
        list: List of BaseScoringResult instances for AltTest scoring.
    """
    return [
        BaseScoringResult(
            scorer_name="alt_test",
            task_name="safety",
            judge_id="judge_1",
            score=0.0025,
            metadata={
                "advantage_probability": 0.75,
                "scoring_function": "alttest_score",
                "total_comparisons": 150,
            },
        ),
        BaseScoringResult(
            scorer_name="alt_test",
            task_name="safety",
            judge_id="judge_2",
            score=0.0031,
            metadata={
                "advantage_probability": 0.68,
                "scoring_function": "alttest_score",
                "total_comparisons": 150,
            },
        ),
    ]


# ==== COMMON FIXTURE ALIASES FOR BACKWARD COMPATIBILITY ====
# These maintain compatibility with existing test method signatures


@pytest.fixture
def sample_consolidated_judge_df(basic_judge_df):
    """Alias for basic_judge_df to maintain backward compatibility.

    Args:
        basic_judge_df: A basic judge DataFrame for classification testing.

    Returns:
        pl.DataFrame: A basic judge DataFrame with classification data.
    """
    return basic_judge_df


@pytest.fixture
def sample_consolidated_human_df(basic_human_df):
    """Alias for basic_human_df to maintain backward compatibility.

    Args:
        basic_human_df: A basic human DataFrame for classification testing.

    Returns:
        pl.DataFrame: A basic human DataFrame with classification data.
    """
    return basic_human_df


@pytest.fixture
def sample_judge_df(text_judge_df):
    """Alias for text_judge_df to maintain backward compatibility.

    Args:
        text_judge_df: A text judge DataFrame for similarity testing.

    Returns:
        pl.DataFrame: A text judge DataFrame with similarity data.
    """
    return text_judge_df


@pytest.fixture
def sample_human_df(text_human_df):
    """Alias for text_human_df to maintain backward compatibility.

    Args:
        text_human_df: A text human DataFrame for similarity testing.

    Returns:
        pl.DataFrame: A text human DataFrame with similarity data.
    """
    return text_human_df


@pytest.fixture
def multi_human_df(multi_annotator_human_df):
    """Alias for multi_annotator_human_df to maintain backward compatibility.

    Args:
        multi_annotator_human_df: A multi-annotator human DataFrame for testing.

    Returns:
        pl.DataFrame: A multi-annotator human DataFrame with multiple annotators.
    """
    return multi_annotator_human_df
