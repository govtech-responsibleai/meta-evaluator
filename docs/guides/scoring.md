# Scoring and Metrics

Configure and use alignment metrics to compare judge evaluations with human annotations.

## Quick Setup

=== "Single Metric"

    ```python linenums="1" hl_lines="8-17"
    from meta_evaluator import MetaEvaluator
    from meta_evaluator.scores import MetricConfig, MetricsConfig
    from meta_evaluator.scores.metrics import AccuracyScorer

    evaluator = MetaEvaluator(project_dir="my_project", load=True)

    # Configure single metric
    config = MetricsConfig(
        metrics=[
            MetricConfig(
                scorer=AccuracyScorer(),
                task_names=["rejection"],
                task_strategy="single", # One of 'single', 'multilabel', or 'multitask'
                annotator_aggregation="individual_average",  # Default
            ),
        ]
    )

    # Add metrics configuration and run comparison
    evaluator.add_metrics_config(config)
    evaluator.compare_async()
    ```

=== "Multiple Metrics"

    ```python linenums="1" hl_lines="11-52"
    from meta_evaluator.scores.metrics import (
        AccuracyScorer,
        AltTestScorer,
        CohensKappaScorer,
        TextSimilarityScorer,
    )
    alt_test_scorer = AltTestScorer(multiplicative_epsilon=True)
    cohens_kappa_scorer = CohensKappaScorer()
    text_similarity_scorer = TextSimilarityScorer()
    
    config = MetricsConfig(
        metrics=[
            MetricConfig(
                scorer=alt_test_scorer,
                task_names=[
                    "hateful",
                    "insults",
                    "sexual",
                    "physical_violence",
                    "self_harm",
                    "all_other_misconduct",
                ],
                task_strategy="multilabel",
                annotator_aggregation="individual_average",
            ),
            MetricConfig(
                scorer=cohens_kappa_scorer,
                task_names=["hateful"],
                task_strategy="single",
                annotator_aggregation="individual_average",
            ),
            MetricConfig(
                scorer=cohens_kappa_scorer,
                task_names=[
                    "hateful",
                    "insults",
                    "sexual",
                    "physical_violence",
                    "self_harm",
                    "all_other_misconduct",
                ],
                task_strategy="multitask",
                annotator_aggregation="individual_average",
            ),
            MetricConfig(
                scorer=text_similarity_scorer,
                task_names=["explanation"],
                task_strategy="single",
                annotator_aggregation="majority_vote",  # Example using majority vote
            ),
        ]
    )

    # Add metrics configuration and run comparison
    evaluator.add_metrics_config(config)
    evaluator.compare_async()
    ```
## Available Scorers

=== "Accuracy"

    ```python linenums="1"
    from meta_evaluator.scores.metrics import AccuracyScorer
    
    accuracy_scorer = AccuracyScorer()
    
    config = MetricConfig(
        scorer=accuracy_scorer,
        task_names=["classification_field"],
        task_strategy="single",
        annotator_aggregation="individual_average",
    )
    ```

    - **Purpose**: Classification accuracy between judge and human annotations
    - **Requirements**: 1 human annotator minimum
    - **Output**: Percentage accuracy (0-1)

    **Sample Results:**

    ```json
    {
      "judge_id": "gpt_4_judge",
      "scorer_name": "accuracy",
      "task_strategy": "single",
      "task_name": "rejection",
      "score": 0.87,
      "total_instances": 100,
      "correct_instances": 87,
      "metadata": {
        "human_annotators": 2,
        "scoring_date": "2025-01-15T10:30:00Z"
      }
    }
    ```

=== "Cohen's Kappa"

    ```python linenums="1"
    from meta_evaluator.scores.metrics import CohensKappaScorer
    
    kappa_scorer = CohensKappaScorer()
    
    config = MetricConfig(
        scorer=kappa_scorer,
        task_names=["classification_field"],
        task_strategy="single",
        annotator_aggregation="individual_average",
    )
    ```

    - **Purpose**: Inter-rater agreement accounting for chance
    - **Requirements**: 2 human annotators minimum
    - **Output**: Kappa coefficient (-1 to 1)
    
    !!! note "Kappa Interpretation"
        | Kappa Range | Interpretation |
        |-------------|----------------|
        | < 0.00      | Poor           |
        | 0.00-0.20   | Slight         |  
        | 0.21-0.40   | Fair           |
        | 0.41-0.60   | Moderate       |
        | 0.61-0.80   | Substantial    |
        | 0.81-1.00   | Almost Perfect |

    **Sample Results:**

    ```json
    {
      "judge_id": "claude_judge",
      "scorer_name": "cohens_kappa", 
      "task_strategy": "single",
      "task_name": "rejection",
      "score": 0.72,
      "interpretation": "substantial",
      "metadata": {
        "observed_agreement": 0.85,
        "expected_agreement": 0.42,
        "human_annotators": 3
      }
    }
    ```


=== "Alt-Test"

    ```python linenums="1"
    from meta_evaluator.scores.metrics import AltTestScorer
    
    alt_test_scorer = AltTestScorer(multiplicative_epsilon=True) # Set multiplicative_epsilon=True to modify the original hypothesis.
    
    config = MetricConfig(
        scorer=alt_test_scorer,
        task_names=["classification_field"],
        task_strategy="single",
        annotator_aggregation="individual_average",
    )
    ```

    - **Purpose**: Alt-Test
    - **Requirements**: 3 human annotators minimum, and minimally 30 instances per human. (For statistical significance.)
      ```
      # To configure min_instances_per_human, run
      alt_test_scorer.min_instances_per_human = 30
      ```
    - **Output**: Winning Rates across different epsilon values, Advantage Probability, and Human Advantage Probabilities.

    !!! note
        Alt-Test is a leave-one-annotator-out hypothesis test that measures whether an LLM judge agrees with the remaining human consensus at least as well as the left-out human does.

    **Sample Results:**

    ```json
    {
      "judge_id": "anthropic_claude_3_5_haiku_judge",
      "scorer_name": "alt_test",
      "task_strategy": "multilabel", 
      "task_name": "rejection",
      "scores": {
        "winning_rate": {
          "0.00": 0.0,
          "0.05": 0.0,
          "0.10": 0.0,
          "0.15": 0.0,
          "0.20": 0.0,
          "0.25": 0.0,
          "0.30": 0.0
        },
        "advantage_probability": 0.9
      },
      "metadata": {
        "human_advantage_probabilities": {
          "person_1": [0.9, 1.0],
          "person_2": [0.9, 0.8], 
          "person_3": [0.9, 1.0]
        },
        "scoring_function": "accuracy",
        "epsilon": 0.2,
        "multiplicative_epsilon": false,
        "min_instances_per_human": 10,
        "ground_truth_method": "alt_test_procedure"
      }
    }
    ```

=== "Text Similarity"

    ```python linenums="1"
    from meta_evaluator.scores.metrics import TextSimilarityScorer
    
    text_scorer = TextSimilarityScorer()
    
    config = MetricConfig(
        scorer=text_scorer,
        task_names=["explanation"],
        task_strategy="single",
        annotator_aggregation="majority_vote",  # Example using majority vote
    )
    ```

    - **Purpose**: String similarity for text responses. Uses SequenceMatcher.
    - **Requirements**: 1 human annotator minimum
    - **Output**: Similarity scores (0-1)

    !!! note
        SequenceMatcher uses the Ratcliff-Obershelp pattern matching algorithm, which recursively looks for the longest contiguous matching subsequence between the two sequences, and calculate similarity ratio using the formula: 
        `(2 × total_matching_characters) / (len(A) + len(B))`

    **Sample Results:**

    ```json
    {
      "judge_id": "gpt_4_judge",
      "scorer_name": "text_similarity",
      "task_strategy": "single", 
      "task_name": "explanation",
      "score": 0.76,
      "metadata": {
        "mean_similarity": 0.76,
        "median_similarity": 0.78,
        "std_similarity": 0.12,
        "total_comparisons": 100
      }
    }
    ```

=== "Semantic Similarity"

    ```python linenums="1"
    from meta_evaluator.scores.metrics import SemanticSimilarityScorer
    
    semantic_scorer = SemanticSimilarityScorer()
    
    config = MetricConfig(
        scorer=semantic_scorer,
        task_names=["explanation"],
        task_strategy="single",
        annotator_aggregation="individual_average",
    )
    ```

    - **Purpose**: Semantic similarity for text responses using OpenAI embeddings.
    - **Requirements**: 
        - 1 human annotator minimum
        - OpenAI API key (set `OPENAI_API_KEY` environment variable)
    - **Output**: Cosine similarity scores (0-1)

    !!! note
        Uses OpenAI's text embedding models to compute embeddings and calculate cosine similarity between judge and human text responses. Captures semantic meaning rather than just string matching.
        
        ```bash
        export OPENAI_API_KEY="your-openai-api-key"
        ```
    
    **Sample Results:**


    ```json
    {
      "judge_id": "claude_judge",
      "scorer_name": "semantic_similarity",
      "task_strategy": "single", 
      "task_name": "explanation",
      "score": 0.84,
      "metadata": {
        "mean_similarity": 0.84,
        "median_similarity": 0.86,
        "std_similarity": 0.08,
        "total_comparisons": 100,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
      }
    }
    ```

!!! note "Score Ranges"
    - **Accuracy**: 0-1 (higher is better)
    - **Cohen's Kappa**: -1 to 1 (higher is better, accounts for chance) 
    - **Text/Semantic Similarity**: 0-1 (higher is better, semantic similarity)
    - **Alt-Test**: Winning rates across epsilon values, advantage probabilities (0-1)


## Task Configuration Types (`task_strategy`)

The `task_strategy` parameter defines how tasks are processed and must be one of: `"single"`, `"multitask"`, or `"multilabel"`.

!!! important "Task Count Requirements"
    - **`"single"`**: Use only when `task_names` contains **exactly 1 task**
    - **`"multitask"`** and **`"multilabel"`**: Use only when `task_names` contains **2 or more tasks**

=== "Single Label (Single Task)"
    
    Evaluate one classification task:

    ```python linenums="1" hl_lines="6 7"
    # Single binary classification task
    config = MetricsConfig(
        metrics=[
            MetricConfig(
                scorer=AccuracyScorer(),
                task_names=["rejection"],  # Single task name
                task_strategy="single",  # Required: "single" for single task
                annotator_aggregation="individual_average",
            ),
        ]
    )
    ```

    !!! note "Single Task Behavior"
        - Single score for the specified task
        - **Required**: Exactly 1 task in `task_names` list
        - Use this when evaluating one task independently

=== "Multi-Task (Multiple Single Labels)"
    
    Apply the same scorer to multiple separate tasks:

    ```python linenums="1" hl_lines="6 7"
    # Same scorer applied to each task separately
    config = MetricsConfig(
        metrics=[
            MetricConfig(
                scorer=AccuracyScorer(),
                task_names=["helpful", "harmless", "honest"],  # Multiple tasks
                task_strategy="multitask",  # Required: "multitask" for multiple separate tasks
                annotator_aggregation="majority_vote",  # Example using majority vote
            ),
        ]
    )
    ```

    !!! note "Multi-Task Behavior"
        - **Required**: 2 or more tasks in `task_names` list
        - **Same scorer applied to each task individually**
        - **Result**: Aggregated score across all tasks
        - For different scorers on different tasks, create separate `MetricConfig` entries

=== "Multi-Label (Combined Classification)"
    
    Treat multiple classification tasks as a single multi-label problem:

    ```python linenums="1" hl_lines="8 9"
    # Multi-label classification (AltTestScorer only)
    alt_test_scorer = AltTestScorer()

    config = MetricsConfig(
        metrics=[
            MetricConfig(
                scorer=alt_test_scorer,
                task_names=["hateful", "violent", "sexual"],  # Combined as multi-label
                task_strategy="multilabel",  # Required: "multilabel" for combined tasks
                annotator_aggregation="individual_average",
            ),
        ]
    )
    ```


    !!! note "Multi-label Behavior"
        - **Required**: 2 or more tasks in `task_names` list
        - Each instance can have multiple labels: `["hateful", "violent"]`, and these will be passed as 1 input into the Scorer.
        - Calculation of metric depends on the Scorer. For instance AltTestScorer uses Jaccard similarity for comparison, and AccuracyScorer uses exact match.


**Example with different scorers per task:**
```python linenums="1" hl_lines="7 13 19"
config = MetricsConfig(
    metrics=[
        # Accuracy for single classification tasks
        MetricConfig(
            scorer=AccuracyScorer(),
            task_names=["helpful"],
            task_strategy="single",
            annotator_aggregation="majority_vote",
        ),
        # Accuracy for multitask classification tasks
        MetricConfig(
            scorer=AccuracyScorer(),
            task_names=["helpful", "harmless", "honest"],
            task_strategy="multitask",
            annotator_aggregation="individual_average",
        ),
        # Text similarity for text tasks, all tasks combined into one multilabel
        MetricConfig(
            scorer=TextSimilarityScorer(),
            task_names=["explanation", "reasoning"],
            task_strategy="multilabel",
            annotator_aggregation="majority_vote",
        ),
    ]
)
```

## Annotator Aggregation (`annotator_aggregation`)

Control how multiple human annotations are aggregated before comparison with judge results.

=== "Individual Average (Default)"

    Compare judge vs each human separately, then average the scores:

    ```python
    MetricConfig(
        scorer=AccuracyScorer(),
        task_names=["rejection"],
        task_strategy="single",
        annotator_aggregation="individual_average"  # Default
    )
    ```

    **How it works:**
    - Judge vs Human 1: Calculate metric score
    - Judge vs Human 2: Calculate metric score
    - Judge vs Human 3: Calculate metric score
    - **Final Score**: Average of all individual scores

=== "Majority Vote"

    Aggregate humans first using majority vote, then compare judge vs consensus:

    ```python
    MetricConfig(
        scorer=AccuracyScorer(),
        task_names=["rejection"],
        task_strategy="single",
        annotator_aggregation="majority_vote"
    )
    ```

    **How it works:**
    - Find consensus among human annotators first
    - Compare judge predictions with consensus
    - Implementation varies by metric type (see support table below)

### Metric Support Table

| Metric | individual_average | majority_vote |
|--------|-------------------|---------------|
| **AccuracyScorer** | :material-check: | :material-check: Per-position majority for multilabel, alphabetical tie-breaking |
| **TextSimilarityScorer** | :material-check: | :material-check: Best-match approach: highest similarity human per sample |
| **SemanticSimilarityScorer** | :material-check: | :material-check: Best-match approach: highest similarity human per sample |
| **CohensKappaScorer** | :material-check: | :material-close: Logs warning, falls back to individual_average (agreement metric) |
| **AltTestScorer** | :material-check: | :material-close: Logs warning, falls back to individual_average (agreement metric) |

**Why agreement metrics don't support majority_vote:**
Inter-annotator agreement metrics measure disagreement between individual humans. Majority vote eliminates this disagreement information, making the metrics less meaningful.

## Custom Scorer

You may implement your own evaluation metrics.

Here is a concrete example of custom metric that **count how many times judge has more "A"s than human** in it's response.

```python linenums="1"
from meta_evaluator.scores.base_scorer import BaseScorer
from meta_evaluator.scores.base_scoring_result import BaseScoringResult
from meta_evaluator.scores.enums import TaskAggregationMode
from typing import Any, List
import polars as pl

class MyCustomScorer(BaseScorer):
    def __init__(self):
        super().__init__(scorer_name="my_custom_scorer")
    
    def can_score_task(self, sample_label: Any) -> bool:
        """Determine if this scorer can handle the given data type.
        
        Args:
            sample_label: Sample of the actual data that will be scored
            
        Returns:
            True if scorer can handle this data type
        """
        # This example only works with a str or a list of str.
        if isinstance(sample_label, str):
            return True
        elif isinstance(sample_label, list):
            # Check if list contains str
            if len(sample_label) > 0:
                return isinstance(sample_label[0], str)
            return True  # Empty list is acceptable
        else:
            return False
    
    async def compute_score_async(
        self,
        judge_data: pl.DataFrame,
        human_data: pl.DataFrame, 
        task_name: str,
        judge_id: str,
        aggregation_mode: TaskAggregationMode,
    ) -> BaseScoringResult:
        """Compute alignment score between a single judge and all human data.
        
        Args:
            judge_data: DataFrame with 1 judge outcomes (columns: original_id, label)
            human_data: DataFrame with human outcomes (columns: original_id, human_id, label)
            task_name: Name of the task being scored
            judge_id: ID of the judge being scored
            aggregation_mode: How tasks were aggregated.
            
        Returns:
            Scoring result object
        """
        # Join judge and human data on original_id
        comparison_df = judge_data.join(human_data, on="original_id", how="inner")
        
        if comparison_df.is_empty():
            score = 0.0
            num_comparisons = 0
            failed_comparisons = 1
        else:
            judge_wins = []
            num_comparisons = 0
            failed_comparisons = 0
            
            # Compare judge vs each human annotator
            humans = comparison_df["human_id"].unique()
            for human_id in humans:
                try:
                    comparison_subset = comparison_df.filter(
                        pl.col("human_id") == human_id
                    )
                    judge_texts = comparison_subset["label"].to_list()
                    human_texts = comparison_subset["label_right"].to_list()
                    
                    judge_more_A = 0
                    human_more_A = 0
                    for judge_text, human_text in zip(judge_texts, human_texts):
                        judge_count = str(judge_text).count("A")
                        human_count = str(human_text).count("A")
                        if judge_count > human_count:
                            judge_more_A += 1
                        else:
                            human_more_A += 1
                    
                    # Judge "wins" if they have more A's in more instances
                    if judge_more_A > human_more_A:
                        judge_wins.append(1)
                    else:
                        judge_wins.append(0)
                    
                    num_comparisons += 1
                    
                except Exception as e:
                    self.logger.error(f"Error computing score: {e}")
                    failed_comparisons += 1
                    continue
            
            # Calculate win rate
            score = sum(judge_wins) / len(judge_wins) if len(judge_wins) > 0 else 0.0
            num_comparisons = len(comparison_df)
        
        return BaseScoringResult(
            scorer_name=self.scorer_name,
            task_name=task_name,
            judge_id=judge_id,
            scores={"win_rate": score},
            metadata={
                "human_annotators": len(comparison_df["human_id"].unique()) if not comparison_df.is_empty() else 0
            },
            aggregation_mode=aggregation_mode,
            num_comparisons=num_comparisons,
            failed_comparisons=failed_comparisons,
        )
    
    def aggregate_results(
        self,
        results: List[BaseScoringResult],
        scores_dir: str, 
        unique_name: str = ""
    ) -> None:
        """Optional: Takes in all judge results and generate aggregate visualizations.
        
        Args:
            results: List of scoring results
            scores_dir: Directory to save plots
            unique_name: Unique identifier for this configuration
        """
        # Optional: Create custom visualizations
        # self._create_custom_plots(results, scores_dir, unique_name)
        pass
        

# Usage
custom_scorer = MyCustomScorer()

config = MetricConfig(
    scorer=custom_scorer,
    task_names=["text"],
    task_strategy="single",
    annotator_aggregation="individual_average",
)
```

!!! tip "Custom Scorer Requirements"
    **Required methods to implement:**
    
    1. **`can_score_task(sample_label)`** - Check if scorer can handle the data type
    2. **`compute_score_async(judge_data, human_data, task_name, judge_id, aggregation_mode)`** - Core scoring logic  
    3. **`aggregate_results(results, scores_dir, unique_name)`** - Optional visualization method
    
    **Guidelines:**
    
    - Call `super().__init__(scorer_name="your_scorer_name")` in constructor
    - Return `BaseScoringResult` from `compute_score_async()`
    - Handle edge cases (empty data, mismatched IDs, etc.)
    - Add meaningful metadata for debugging and transparency


## Results Output

### Individual Metric Results

Detailed scores and charts are saved to individual metric directories in your project:

```
my_project/
└── scores/
    ├── accuracy/
    │   └── accuracy_1tasks_22e76eaf_rejection_accuracy/
    │       ├── accuracy_scores.png
    │       ├── gpt_4_judge_result.json
    │       └── claude_judge_result.json
    ├── cohens_kappa/
    │   └── cohens_kappa_1tasks_22e76eaf_rejection_agreement/
    │       ├── cohens_kappa_scores.png
    │       ├── gpt_4_judge_result.json
    │       └── claude_judge_result.json
    ├── alt_test/
    │   └── alt_test_3tasks_22e76eaf_safety_significance/
    │       ├── aggregate_advantage_probabilities.png
    │       ├── aggregate_human_vs_llm_advantage.png
    │       ├── aggregate_winning_rates.png
    │       ├── gpt_4_judge_result.json
    │       └── claude_judge_result.json
    └── text_similarity/
        └── text_similarity_1tasks_74d08617_explanation_quality/
            ├── text_similarity_scores.png
            ├── gpt_4_judge_result.json
            └── claude_judge_result.json
```

### Summary Reports

You can generate summary reports that aggregate all metrics across all judges in a single view.

```python linenums="1"
# After running evaluations and configuring metric configs
evaluator.add_metrics_config(config)
evaluator.compare_async()

# Save to files
evaluator.score_report.save("score_report.html", format="html")  # Interactive HTML with highlighting
evaluator.score_report.save("score_report.csv", format="csv")    # CSV for analysis

# Print to console
evaluator.score_report.print()
```

Summary reports are saved to the scores directory:

```
my_project/
└── scores/
    ├── score_report.html    # Interactive HTML table with best score highlighting
    ├── score_report.csv     # CSV format for analysis/Excel
    ├── accuracy/            # Detailed accuracy results...
    ├── cohens_kappa/        # Detailed kappa results...
    ├── alt_test/            # Detailed alt-test results...
    └── text_similarity/     # Detailed similarity results...
```

**Sample Console Output:**
```
Score Report:
┌─────────┬─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│ judge_id ┆ accuracy_1tasks_22e ┆ alt_test_1tasks_22e ┆ alt_test_1tasks_22e ┆ text_similarity_1ta │
│ ---     ┆ 76eaf_single        ┆ 76eaf_single_winni… ┆ 76eaf_single_advant ┆ sks_74d0861_single  │
│ str     ┆ ---                 ┆ ---                 ┆ ---                 ┆ ---                 │
│         ┆ f64                 ┆ f64                 ┆ f64                 ┆ f64                 │
╞═════════╪═════════════════════╪═════════════════════╪═════════════════════╪═════════════════════╡
│ judge1  ┆ 0.87               ┆ 0.6                 ┆ 0.75                ┆ 0.85                │
│ judge2  ┆ 0.82               ┆ 0.4                 ┆ 0.65                ┆ 0.78                │
└─────────┴─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘
```

**Sample HTML Report:**
<figure markdown="span">
  ![Score Report HTML Table](../assets/example_summary_report.png)
</figure>
