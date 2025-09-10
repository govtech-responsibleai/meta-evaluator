# Loading Results

Access judge evaluation results and human annotations.

## Load Results

```python linenums="1"
# Load both judge and human results  
judge_results = evaluator.load_all_judge_results()
human_results = evaluator.load_all_human_results()

print(f"Judges: {len(judge_results)}, Humans: {len(human_results)}")
```

## View Data Format

```python linenums="1"
# Check judge results data format
for judge_id, judge_result in judge_results.items():
    print(f"Judge: {judge_id}")
    print(judge_result.data)  # Shows the Polars DataFrame
    break

# Check human results data format  
for human_id, human_result in human_results.items():
    print(f"Human: {human_id}")
    print(human_result.data)  # Shows the Polars DataFrame
    break
```

## Result Files

Results are stored in your project directory:

```
my_project/
├── main_state.json             # Project configuration
├── data/
│   └── main_state_data.json    # Your evaluation data
├── results/                    # Judge evaluation results
│   ├── run_20250815_110504_15c89e71_anthropic_claude_3_5_haiku_judge_20250815_110521_results.json
│   ├── run_20250815_110504_15c89e71_anthropic_claude_3_5_haiku_judge_20250815_110521_state.json
│   └── run_20250815_110504_15c89e71_openai_gpt_4_1_nano_judge_20250815_110521_results.json
├── annotations/                # Human annotation results  
│   ├── annotation_run_20250715_171040_f54e00c6_person_1_Person 1_data.json
│   └── annotation_run_20250715_171040_f54e00c6_person_1_Person 1_metadata.json
└── scores/                     # Computed alignment metrics (after comparison)
    ├── accuracy/
    ├── cohens_kappa/
    ├── alt_test/
    └── text_similarity/
```

## External Data Loading

MetaEvaluator supports loading pre-existing judge and human annotation results from external sources. This enables scoring-only workflows where you can compute alignment metrics on existing data without re-running evaluations.

### Loading External Judge Results

Use `add_external_judge_results()` to load judge evaluation data from CSV files:

```python linenums="1"
evaluator.add_external_judge_results(
    file_path="path/to/judge_results.csv",
    judge_id="external_judge",
    llm_client="openai",
    model_used="gpt-4",
    run_id="external_run_1"
)
```

**Arguments:**

- `file_path`: Path to the CSV file containing judge results
- `judge_id`: Unique identifier for this judge
- `llm_client`: The LLM provider used (e.g., "openai", "anthropic"). See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for supported options
- `model_used`: The specific model name used for evaluation (e.g., "gpt-4", "claude-3-sonnet")
- `run_id`: Unique identifier for this evaluation run (optional, will be auto-generated if not provided)

**Required Judge Data Format:**

Your CSV file must contain these columns:

- `original_id`: Original identifier from your evaluation data
- **Task columns**: One column for each task defined in your `EvalTask.task_schemas`

System columns (`sample_example_id`, `run_id`, `judge_id`) are auto-generated from the function parameters.

Example judge results CSV:
```csv
original_id,rejection,explanation
sample_1,rejection,"This response shows harmful content detection."
sample_2,not rejection,"This response demonstrates proper safety measures."
sample_3,rejection,"This response contains concerning patterns."
```

### Loading External Annotation Results

Use `add_external_annotation_results()` to load human annotation data from CSV files:

```python linenums="1"
evaluator.add_external_annotation_results(
    file_path="path/to/human_results.csv",
    annotator_id="external_annotators",
    run_id="external_human_run_1"
)
```

**Arguments:**

- `file_path`: Path to the CSV file containing human annotation results
- `annotator_id`: Unique identifier for the annotator(s)
- `run_id`: Unique identifier for this annotation run (optional, will be auto-generated if not provided)

**Required Human Data Format:**

Your CSV file must contain these columns:

- `original_id`: Original identifier from your evaluation data
- **Task columns**: One column for each task defined in your `EvalTask.task_schemas`

System columns (`sample_example_id`, `run_id`, `annotator_id`) are auto-generated from the function parameters.

Example human results CSV:
```csv
original_id,rejection,explanation
sample_1,rejection,"This content appears problematic based on safety guidelines."
sample_1,rejection,"This response demonstrates concerning patterns."
sample_2,not rejection,"This content appears acceptable based on safety guidelines."
sample_2,not rejection,"This response demonstrates proper safety measures."
```

### Data Schema Requirements

**Important**: The task columns in your external data must match the task schema defined in your `EvalTask`:

```python linenums="1"
# If your EvalTask defines these schemas:
task = EvalTask(
    task_schemas={
        "rejection": ["rejection", "not rejection"],  # Classification task
        "explanation": None,  # Free-form text task
    },
    # ... other parameters
)

# Then your CSV files must have columns named:
# - "rejection" (with values like "rejection" or "not rejection")  
# - "explanation" (with free-form text values)
```

### Complete Example

See `examples/rejection/run_scoring_only_async.py` in the [GitHub Repository](https://github.com/govtech-responsibleai/meta-evaluator) for a complete example that:

1. Loads original evaluation data
2. Creates mock external judge and human results
3. Loads the external results using the methods above
4. Runs scoring metrics to compare judge vs human performance

```python linenums="1"
# Load external results
evaluator.add_external_judge_results(
    file_path="judge_results.csv",
    judge_id="rejection_judge", 
    llm_client="mock_client",
    model_used="mock_model",
    run_id="judge_run_1"
)

evaluator.add_external_annotation_results(
    file_path="human_results.csv",
    annotator_id="human_annotators",
    run_id="human_run_1"
)

# Load and run scoring
judge_results = evaluator.load_all_judge_results()
human_results = evaluator.load_all_human_results()

results = evaluator.compare_async(
    comparison_config=config,
    judge_results=judge_results,
    human_results=human_results
)
```

This approach allows you to leverage MetaEvaluator's scoring capabilities on any existing judge/human evaluation data, making it easy to compute alignment metrics without needing to re-run evaluations.