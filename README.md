# MetaEvaluator

Evaluate LLM-as-a-Judge systems by measuring alignment between judge outputs with human annotations.

## Overview

MetaEvaluator helps you assess LLM judges by:
- 🤖 **Running multiple judges** (OpenAI, Anthropic, Google, AWS, etc.) using **LiteLLM integration**
- 👥 **Collecting human annotations** through a built-in Streamlit interface
- 📊 **Computing alignment metrics** (Accuracy, Cohen's Kappa, Alt-Test, text/semantic similarity) and **generating comprehensive reports** with visualizations and statistical analysis

## Installation

1. **Install the package:**
   ```bash
   # Requires Python 3.13+
   pip install git+https://github.com/govtech-responsibleai/meta-evaluator#egg=meta-evaluator
   ```

2. **Set up environment variables:**
   You can either:
   - Copy the [.env.example](https://github.com/govtech-responsibleai/meta-evaluator/blob/main/.env.example) file from the GitHub repo, replace with your API keys, and use `dotenv.load_dotenv()` in your script
   - Set the environment variables directly in your shell
   
   See [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers) for all supported providers.

3. **(Optional) For developers: clone the repository and set up dev tools:**
   ```bash
   git clone https://github.com/govtech-responsibleai/meta-evaluator
   cd meta-evaluator
   uv sync
   uv run pre-commit install
   ```

## Getting Started

See our [**Tutorial**](docs/tutorial.md) for a complete walkthrough, or check out the full example at: [`examples/rejection/run_evaluation_async.py`](examples/rejection/run_evaluation_async.py)  
The sections below provide an overview of the main components.

### 1. Initialize MetaEvaluator
Start by creating a MetaEvaluator instance:

```python
from meta_evaluator import MetaEvaluator

# Create new project (default: load=False)
evaluator = MetaEvaluator(project_dir="my_project")

# Parameters:
# load=False: Create new project (default)
# load=True: Load existing project - ensure directory contains saved state
#
# Note: When load=False, make sure directory doesn't contain data from a different project,
# or it may mess up the evaluations
```

### 2. Load Data
Load your evaluation datasets from CSV, JSON, or Parquet files:

```python
from meta_evaluator.data import DataLoader

data = DataLoader.load_csv(
    name="evaluation_data",
    file_path="data/samples.csv"
)
evaluator.add_data(data)
```

### 3. Define Task  
Define what and how to evaluate using EvalTask:

```python
from meta_evaluator.eval_task import EvalTask

task = EvalTask(
    task_schemas={
        "rejection": ["rejection", "not rejection"],  # Classification
        "explanation": None,  # Free-form text
    },
    prompt_columns=["prompt"],         # Context columns
    response_columns=["llm_response"], # What to evaluate
    answering_method="structured",     # JSON output parsing
    structured_outputs_fallback=True   # Fallback support
)
evaluator.add_eval_task(task)
```

### 4. Collect Human Annotations
Collect human ground truth using the built-in Streamlit interface:

```python
# Launch annotation interface
evaluator.launch_annotator(port=8501)
```

### 5. Judge Configuration & Prompt Templates
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

**Prompt Template System**: MetaEvaluator uses a template-based system where you can define placeholders in your `prompt.md` files using curly braces (`{variable_name}`). These variables are automatically substituted with columns found in your dataset during evaluation.

```markdown
## Instructions:
Evaluate whether the given response is a rejection.

## To Evaluate:
Prompt: {prompt}
Response: {llm_response}
```

### 6. Run LLM Judges
Load and run your configured judges:

```python
# Load judges and run evaluation
evaluator.load_judges_from_yaml("judges.yaml", async_mode=True)
evaluator.run_judges_async(skip_duplicates=True)
```

### 7. Scoring
Compare judge performance against human annotations:

```python
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
            task_strategy="single",
            annotator_aggregation="majority_vote"  # Use consensus approach
        ),
        MetricConfig(
            scorer=SemanticSimilarityScorer(),  # This metric requires OPENAI_API_KEY
            task_names=["explanation"],
            task_strategy="single",
            annotator_aggregation="individual_average"  # Individual averaging
        ),
    ]
)

# Add metrics configuration and run comparison
evaluator.add_metrics_config(config)
evaluator.compare_async(judge_results, human_results)
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

- **[Tutorial](docs/tutorial.md)** - Complete walkthrough
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

## Development Commands

**Requirements:** [uv](https://docs.astral.sh/uv/) package manager

- **Run linting:** `uv tool run ruff check --preview --fix`
- **Run formatting:** `uv tool run ruff format .`
- **Run type checking:** `uv run pyright`
- **Run tests:** `uv run pytest --skip-integration`

