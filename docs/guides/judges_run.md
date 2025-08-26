# Running Evaluations

Execute your configured LLM judges to evaluate your dataset.

## Quick Setup

=== "Async (Recommended)"

    ```python
    from meta_evaluator import MetaEvaluator
    
    evaluator = MetaEvaluator(project_dir="my_project", load=True)
    
    # Run all judges asynchronously
    evaluator.run_judges_async(
        skip_duplicates=True  # Skip judges with existing results
    )
    ```

=== "Synchronous"

    ```python
    from meta_evaluator import MetaEvaluator
    
    evaluator = MetaEvaluator(project_dir="my_project", load=True)

    # Run all judges synchronously  
    evaluator.run_judges(
        skip_duplicates=True
    )
    ```

=== "Specific Judges"

    ```python
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

```python
evaluator.run_judges_async(
    judge_ids=None,              # Which judges to run
    run_id=None,                 # Custom run identifier
    save_results=True,           # Whether to save results to disk
    results_format="json",       # Output format for results  
    skip_duplicates=True         # Skip judges with existing results
)
```

### Judge Selection (`judge_ids`)

=== "All Judges"

    ```python
    # Run all loaded judges
    judge_ids=None
    ```

=== "Single Judge"

    ```python  
    # Run one specific judge
    judge_ids="gpt_4_judge"
    ```

=== "Multiple Judges"

    ```python
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

```python
# Save results to project directory (default)
save_results=True   # Recommended for persistence

# Don't save results (in-memory only)
save_results=False  # For testing or temporary evaluation
```

### Results Format (`results_format`)

Specify output format for saved results:

```python
results_format="json"     # Default: JSON format
results_format="csv"      # CSV format
results_format="parquet"  # Parquet format (efficient for large datasets)
```

### Duplicate Handling (`skip_duplicates`)

Control re-evaluation of existing results:

```python
# Skip judges with existing results (default)
skip_duplicates=True   # Faster, avoids re-evaluation, saves API costs

# Always run all judges
skip_duplicates=False  # Re-evaluates everything, overwrites existing results
```


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

    ```python
    # Fast concurrent processing
    evaluator.run_judges_async()
    ```

    - Multiple judges run in parallel
    - Significant speed improvement for multiple judges

=== "Synchronous Evaluation"

    ```python
    # Sequential processing
    evaluator.run_judges()
    ```

    - One judge at a time
    - Easier debugging, simpler error handling
