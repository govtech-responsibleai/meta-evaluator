# MetaEvaluator Base Class

The `MetaEvaluator` class is the central orchestrator for your evaluation workflow. It manages your evaluation data, task configuration, judges, and results within an organized project directory.


## Quick Setup

=== "New Project"

    ```python linenums="1" hl_lines="4"
    from meta_evaluator import MetaEvaluator

    # Create new project
    evaluator = MetaEvaluator(project_dir="my_project", load=False)

    # Add your components
    evaluator.add_eval_task(task)
    evaluator.add_data(data)
    ```

=== "Load Existing Project"

    ```python linenums="1" hl_lines="4"
    from meta_evaluator import MetaEvaluator

    # Load existing project
    evaluator = MetaEvaluator(project_dir="my_project", load=True)

    # Continue working with loaded data and task
    evaluator.run_judges_async()
    ```

## Project Directory Structure

While using MetaEvaluator, you will see this directory structure:

```
my_project/
├── main_state.json          # Project configuration and metadata
├── data/                    # Evaluation dataset
│   └── main_data.json       # Your evaluation data file
├── results/                 # LLM judge evaluation results
│   ├── run_*_judge1_*.json
│   └── run_*_judge2_*.json
├── annotations/             # Human annotation results
│   ├── annotation_run_*_annotator1_*.json
│   └── annotation_run_*_annotator2_*.json
└── scores/                  # Computed alignment metrics
    ├── score_report.html
    ├── accuracy/
    ├── cohens_kappa/
    └── text_similarity/
```

**Directory:**

- **`main_state.json`**: Stores your project configuration (task schemas, data metadata, judge configurations)
- **`data/`**: Contains your evaluation dataset, referenced by `main_state.json`
- **`results/`**: Stores judge evaluation outputs (automatically created when running judges)
- **`annotations/`**: Stores human annotation data (automatically created when using the annotation interface)
- **`scores/`**: Contains computed metrics and comparison results

## State Management: `save_state()`

```python linenums="1"
evaluator = MetaEvaluator(project_dir="my_project", load=False)
evaluator.add_eval_task(task)
evaluator.add_data(data)
evaluator.load_judges_from_yaml("judges.yaml")

# Save project state
evaluator.save_state(data_format="json")  # or "csv", "parquet"
```

**Saved by `save_state()`:**

- :material-check: Evaluation task configuration (schemas, columns, answering method)
- :material-check: Data metadata and data files
- :material-check: Judge configurations (model, provider, prompt files)

!!! note "Results Persist Independently"
    Judge results, annotations, and scores are saved to their respective directories automatically and persist across sessions.

## Loading Projects: `load=True`

```python linenums="1"
# Load existing project
evaluator = MetaEvaluator(project_dir="my_project", load=True)

# Everything from save_state() is loaded automatically
# Continue working immediately
evaluator.run_judges_async()
```

`load=True` loads the saved state, so you don't have to constantly redefine your evaluation task, data, and judges.

!!! warning "Re-add Metrics Config"
    `MetricsConfig` is not saved. After loading, re-add your metrics configuration:

    ```python linenums="1"
    evaluator = MetaEvaluator(project_dir="my_project", load=True)
    evaluator.add_metrics_config(config)
    evaluator.compare_async()
    ```
