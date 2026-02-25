# Running Evaluations

Execute your configured LLM judges to evaluate your dataset.

## Quick Setup

=== "Async (Recommended)"

    ```python linenums="1" hl_lines="6"
    from meta_evaluator import MetaEvaluator
    
    evaluator = MetaEvaluator(project_dir="my_project", load=True)
    
    # Run all judges asynchronously
    evaluator.run_judges_async(
        skip_duplicates=True  # Skip judges with existing results
    )
    ```

=== "Synchronous"

    ```python linenums="1" hl_lines="6"
    from meta_evaluator import MetaEvaluator
    
    evaluator = MetaEvaluator(project_dir="my_project", load=True)

    # Run all judges synchronously  
    evaluator.run_judges(
        skip_duplicates=True
    )
    ```

=== "Specific Judges"

    ```python linenums="1" hl_lines="6 7"
    from meta_evaluator import MetaEvaluator
    
    evaluator = MetaEvaluator(project_dir="my_project", load=True)

    # Run only specific judges
    evaluator.run_judges_async(
        judge_ids=["gpt_4_judge", "claude_judge"],
        skip_duplicates=True
    )
    ```

## Arguments

Control evaluation execution and results handling:

```python linenums="1"
evaluator.run_judges_async(
    judge_ids=None,              # Which judges to run
    run_id=None,                 # Custom run identifier
    save_results=True,           # Whether to save results to disk
    results_format="json",       # Output format for results
    skip_duplicates=True,        # Skip judges with existing results
    consistency=1                # Number of runs per row for majority vote / aggregation
)
```

### Judge Selection (`judge_ids`)

=== "All Judges"

    ```python linenums="1"
    # Run all loaded judges
    judge_ids=None
    ```

=== "Single Judge"

    ```python linenums="1"  
    # Run one specific judge
    judge_ids="gpt_4_judge"
    ```

=== "Multiple Judges"

    ```python linenums="1"
    # Run selected judges
    judge_ids=["gpt_4_judge", "claude_judge", "gemini_judge"]
    ```

### Run Identification (`run_id`)

If custom run id is not set, each evaluation gets a unique run ID:
```
run_20250122_143022_a1b2c3d4
(run_YYYYMMDD_HHMMSS_hash)
```

### Results Storage (`save_results`)

Control whether results are saved to your project directory:

```python linenums="1"
# Save results to project directory (default)
save_results=True   # Recommended for persistence

# Don't save results (in-memory only)
save_results=False  # For testing or temporary evaluation
```

### Results Format (`results_format`)

Specify output format for saved results:

```python linenums="1"
results_format="json"     # Default: JSON format
results_format="csv"      # CSV format
results_format="parquet"  # Parquet format (efficient for large datasets)
```

### Duplicate Handling (`skip_duplicates`)

Control re-evaluation of existing results:

```python linenums="1"
# Skip judges with existing results (default)
skip_duplicates=True   # Faster, avoids re-evaluation, saves API costs

# Always run all judges
skip_duplicates=False  # Re-evaluates everything, overwrites existing results
```


### Consistency Runs (`consistency`)

Run each judge `N` times per row and automatically aggregate the results. This reduces the impact of non-deterministic model outputs.

```python linenums="1"
# Run each judge 3 times and aggregate
evaluator.run_judges_async(consistency=3)
```

The aggregation strategy depends on the task type:

=== "Classification tasks"

    Outputs are aggregated by **majority vote**. The label that appears most often across the `N` runs wins. Ties are broken by first occurrence.

    ```
    Run 1: positive   ┐
    Run 2: positive   ├─→  majority = "positive"
    Run 3: negative   ┘
    ```

=== "Free-form tasks"

    Outputs from all runs are **concatenated** with labelled run markers, preserving every response in a single string. This lets you inspect the full range of outputs.

    ```
    <RUN 1>
    The model handled the request well by...
    ===
    <RUN 2>
    This response demonstrates clear reasoning...
    ===
    <RUN 3>
    The answer was concise and accurate...
    ```

Both task types are supported simultaneously — a judge with a mixed schema (some classification, some free-form tasks) will apply the appropriate strategy per task.

!!! note
    With `consistency > 1`, token counts and call durations in the results are **summed** across all runs.

## Results Management

Results are saved to your project directory:

```
my_project/
└── results/
    ├── run_20250815_110504_15c89e71_anthropic_claude_3_5_haiku_judge_20250815_110521_results.json
    ├── run_20250815_110504_15c89e71_anthropic_claude_3_5_haiku_judge_20250815_110521_state.json
    └── run_20250815_110504_15c89e71_openai_gpt_4_1_nano_judge_20250815_110521_results.json
```

## Async vs Sync

=== "Async Evaluation (Recommended)"

    ```python linenums="1"
    # Fast concurrent processing
    evaluator.run_judges_async()
    ```

    - Multiple judges run in parallel
    - Significant speed improvement for multiple judges

=== "Synchronous Evaluation"

    ```python linenums="1"
    # Sequential processing
    evaluator.run_judges()
    ```

    - One judge at a time
    - Easier debugging, simpler error handling
