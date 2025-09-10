# MetaEvaluator

A comprehensive Python framework for evaluating LLM-as-a-Judge systems by comparing judge outputs with human annotations and calculating alignment metrics.

## Overview

MetaEvaluator helps you assess LLM judges by:
- **Running multiple judges** (OpenAI, Anthropic, Google, AWS, etc.) over the same dataset using **LiteLLM integration**
- **Collecting human annotations** through a built-in Streamlit interface
- **Computing alignment metrics** (Accuracy, Cohen's Kappa, Alt-Test, text/semantic similarity)
- **Generating comprehensive reports** with visualizations and statistical analysis

## Quick Start

See our [**Quickstart Guide**](docs/quickstart.md) for a complete walkthrough.

### Basic Example:
Check out the full example: [`examples/rejection/run_evaluation_async.py`](examples/rejection/run_evaluation_async.py)  
The sections below provide an overview of the main components.


## Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for LLM providers (OpenAI, Anthropic, etc.)

## Installation

1. **Install the package:**
   ```bash
   pip install git+https://github.com/govtech-responsibleai/meta-evaluator#egg=meta-evaluator
   ```

2. **Set up environment variables:**
   You can either:
   - Copy the [.env.example](https://github.com/govtech-responsibleai/meta-evaluator/blob/main/.env.example) file from the GitHub repo, replace with your API keys, and use `dotenv.load_dotenv()` in your script
   - Set the environment variables directly in your shell
   
   See [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers) for all supported providers.

4. **(Optional) Set up dev tools:**
   ```bash
   uv run pre-commit install
   ```

## Development Commands

- **Run linting:** `uv tool run ruff check --preview --fix`
- **Run formatting:** `uv tool run ruff format .`
- **Run type checking:** `uv run pyright`
- **Run tests:** `uv run pytest --skip-integration`

## Core Components

MetaEvaluator provides a complete evaluation pipeline with these key components:

### 1. Data Loading
Load your evaluation datasets from CSV, JSON, or Parquet files:

```python
from meta_evaluator.data import DataLoader

data = DataLoader.load_csv(
    name="evaluation_data",
    file_path="data/samples.csv"
)
```

### 2. Task Definition  
Define what and how to evaluate using EvalTask:

```python
from meta_evaluator.eval_task import EvalTask

task = EvalTask(
    task_schemas={
        "rejection": ["rejection", "not rejection"],  # Classification
        "explanation": None,  # Free-form text
    },
    prompt_columns=["prompt"],        # Context columns
    response_columns=["llm_response"], # What to evaluate
    answering_method="structured",    # JSON output parsing
    structured_outputs_fallback=True  # Fallback support
)
```

### 3. Human Annotation Interface
Collect human ground truth using the built-in Streamlit interface:

```python
evaluator = MetaEvaluator(project_dir="my_project", load=False)
evaluator.add_eval_task(task)
evaluator.add_data(data)

# Launch annotation interface
evaluator.launch_annotator(port=8501)
```

### 4. Judge Configuration & Prompt Templates
Configure multiple LLM judges using YAML and template-based prompts:

```yaml
judges:
  - id: gpt_4_judge
    llm_client: openai
    model: gpt-4o-mini
    prompt_file: ./prompt.md # Filepath relative to YAML file
    
  - id: claude_judge
    llm_client: anthropic
    model: claude-3-5-haiku-latest
    prompt_file: ./prompt.md # Filepath relative to YAML file
```

**Prompt Template System**: MetaEvaluator uses a template-based system where you can define placeholders in your `prompt.md` files using curly braces (`{variable_name}`). These variables are automatically substituted with actual data during evaluation:

```markdown
## Instructions:
Evaluate whether the given response is a rejection.

## To Evaluate:
Prompt: {prompt}
Response: {llm_response}
```

The available template variables correspond to your `prompt_columns` and `response_columns` defined in the EvalTask. For example:
- `{prompt}` - substituted with data from the "prompt" column  
- `{llm_response}` - substituted with data from the "llm_response" column
- `{text}` - substituted with data from the "text" column

This allows for dynamic, contextual prompts that adapt to your specific data structure.

### 5. Evaluation & Metrics
Run judges and compute comprehensive alignment metrics:

```python
# Load judges and run evaluation
evaluator.load_judges_from_yaml("judges.yaml", async_mode=True)
evaluator.run_judges_async(skip_duplicates=True)

# Load results  
judge_results = evaluator.load_all_judge_results()
human_results = evaluator.load_all_human_results()

# Configure metrics
from meta_evaluator.scores import MetricConfig, MetricsConfig
from meta_evaluator.scores.metrics import (
    AccuracyScorer, CohensKappaScorer, SemanticSimilarityScorer
)

config = MetricsConfig(
    metrics=[
        MetricConfig(
            scorer=AccuracyScorer(),
            task_names=["rejection"],
            aggregation_name="single"
        ),
        MetricConfig(
            scorer=SemanticSimilarityScorer(),  # This metric requires OPENAI_API_KEY
            task_names=["explanation"], 
            aggregation_name="single"
        ),
    ]
)

# Run comparison
evaluator.compare_async(config, judge_results, human_results)
```

## External Data Loading

MetaEvaluator supports loading pre-existing judge and human annotation results for scoring-only workflows. This is useful when you:
- Have results from previous evaluation runs
- Want to compute metrics on externally generated judge/human data
- Need to re-run scoring with different metrics without re-evaluating

### Loading External Judge Results
```python
# Load external judge results from CSV
evaluator.add_external_judge_results(
    file_path="path/to/judge_results.csv",
    judge_id="external_judge",
    llm_client="openai",  
    model_used="gpt-4",
    run_id="external_run_1"
)
```

**Required CSV columns for judge results:**
- `original_id`: Unique identifier for each sample
- Task columns matching your `EvalTask.task_schemas`

System columns (`sample_example_id`, `run_id`, `judge_id`) are auto-generated from the function parameters.

### Loading External Annotation Results
```python
# Load external human annotations from CSV
evaluator.add_external_annotation_results(
    file_path="path/to/human_results.csv",
    annotator_id="external_annotators",
    run_id="external_human_run_1"
)
```

**Required CSV columns for human results:**
- `original_id`: Unique identifier for each sample  
- Task columns matching your `EvalTask.task_schemas`

System columns (`sample_example_id`, `run_id`, `annotator_id`) are auto-generated from the function parameters.

For detailed data format requirements and examples, see the [Results Guide](docs/guides/results.md#external-data-loading).

## Available Metrics

MetaEvaluator supports comprehensive alignment metrics for evaluating judge performance:

### Classification Metrics
- **Accuracy**: Basic classification accuracy between judge and human labels
- **Cohen's Kappa**: Inter-rater agreement accounting for chance agreement  
- **Alt-Test**: Statistical significance testing with leave-one-annotator-out methodology

### Text Similarity Metrics  
- **Text Similarity**: String-based similarity using sequence matching algorithms
- **Semantic Similarity**: OpenAI embedding-based semantic similarity (requires API key)

### Custom Metrics
- **Custom Scorers**: Implement domain-specific metrics by extending `BaseScorer`

See [Scoring Guide](docs/guides/scoring.md) for detailed usage examples and configuration options.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Quickstart Guide](docs/quickstart.md)** - Get started in 5 minutes
- **[Data Loading](docs/guides/evaldata.md)** - Load and manage evaluation datasets
- **[Task Definition](docs/guides/evaltask.md)** - Define evaluation schemas and parsing methods
- **[Judge Configuration](docs/guides/judges_load.md)** - Set up LLM judges with YAML
- **[Running Evaluations](docs/guides/judges_run.md)** - Execute judge evaluations
- **[Scoring & Metrics](docs/guides/scoring.md)** - Compute alignment metrics
- **[Human Annotations](docs/guides/annotation_guide/annotation.md)** - Collect human ground truth

## Project Structure (automatically generated)

```
project_dir/
├── data/                    # Serialized evaluation data
├── results/                 # Judge evaluation results  
├── annotations/             # Human annotation data
└── scores/                  # Computed alignment metrics
    ├── accuracy/
    ├── cohens_kappa/
    ├── alt_test/
    └── text_similarity/
```

## Examples

See the `examples/` directory for complete working examples:

### Rejection Detection Evaluation
- **[`examples/rejection/run_evaluation_async.py`](examples/rejection/run_evaluation_async.py)** - Complete async evaluation with multiple metrics
- **[`examples/rejection/data/sample_rejection.csv`](examples/rejection/data/sample_rejection.csv)** - Sample rejection detection dataset
- **[`examples/rejection/judges.yaml`](examples/rejection/judges.yaml)** - Judge configuration example
- **[`examples/rejection/prompt.md`](examples/rejection/prompt.md)** - Evaluation prompt template

### RabakBench Evaluation (data not included)
- **[`examples/rabakbench/run_evaluation_async.py`](examples/rabakbench/run_evaluation_async.py)** - Complete async evaluation with multiple metrics

### Scoring-Only Evaluation with External Results
- **[`examples/rejection/run_scoring_only_async.py`](examples/rejection/run_scoring_only_async.py)** - Load external judge/human results and run scoring without re-evaluation

