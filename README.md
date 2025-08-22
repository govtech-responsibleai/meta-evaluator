# MetaEvaluator

A comprehensive Python framework for evaluating LLM-as-a-Judge systems by comparing judge outputs with human annotations and calculating alignment metrics.


## Overview

MetaEvaluator helps you assess LLM judges by:
- Running multiple judges (e.g., OpenAI, Anthropic, Google, AWS, etc.) over the same dataset using **LiteLLM integration**
- Collecting results from LLM judges and human annotators
- Computing alignment metrics (Accuracy, Cohen's Kappa, Alt-Test, text similarity)


### Example:
Check out the full example: [`examples/rejection/run_evaluation.py`](examples/rejection/run_evaluation.py)  
The steps below break down what’s happening in that script.


## Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for LLM providers (OpenAI, Anthropic, etc.)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd meta-evaluator
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   ```bash
   # For OpenAI models
   export OPENAI_API_KEY="your-openai-api-key"
   
   # For Anthropic models  
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   
   # Add other provider keys as needed
   ```

4. **(Optional) Set up dev tools:**
   ```bash
   uv run pre-commit install
   ```

## Development Commands

- **Run linting:** `uv tool run ruff check --preview --fix`
- **Run formatting:** `uv tool run ruff format .`
- **Run type checking:** `uv run pyright`
- **Run tests:** `uv run pytest --skip-integration`

## Usage

Here's a complete example of how to set up and run an evaluation:

### 1. Set up judges YAML file and prompts

Examples: 
- [examples/rejection/judges.yaml](examples/rejection/judges.yaml)
- [examples/rejection/prompt.md](examples/rejection/prompt.md)

### 2. Initialize logging

```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation.log", mode="w", encoding="utf-8"),
    ],
)
```

### 3. Initialize task and data

MetaEvaluator supports two evaluation modes:

**Option 1: Evaluate LLM's Prompt + Response**
- Use when you have both prompts and pre-generated responses to evaluate
- Judges see both the original prompt and the response

**Option 2: Generate Response to Single Column** 
- Use when you want judges to generate responses to text in a single column
- Set `prompt_columns=None` and judges will only see the content from `response_columns`

```python
from meta_evaluator.data import DataLoader
from meta_evaluator.eval_task import EvalTask

# Option 1: Evaluate existing prompt + response pairs
eval_task = EvalTask(
        task_schemas={
            "rejection": ["rejection", "not rejection"], # Classification
            "explanation": None,  # Free-form text field
        },
        prompt_columns=["prompt"],     # Provide context from prompt column
        response_columns=["llm_response"],  # Evaluate this response
        answering_method="structured",
    )

# Option 2: Generate response to single text column
# eval_task = EvalTask(
#         task_schemas={
#             "toxicity": ["toxic", "non-toxic"], 
#         },
#         prompt_columns=None,              # No prompt context
#         response_columns=["text"],        # Judge responds to this text directly  
#         answering_method="structured",
#     )

# Load evaluation data
eval_data = DataLoader.load_csv(
      name="eval_rejection",
      file_path="data/test_data.csv",
   )
```

### 4. Initialize MetaEvaluator and add task and data

```python
from meta_evaluator.meta_evaluator import MetaEvaluator

# Initialize MetaEvaluator with project directory
evaluator = MetaEvaluator(project_dir="project_dir")

# Add evaluation task and data
evaluator.add_eval_task(eval_task)
evaluator.add_data(eval_data)
```

### 5. Load judges

You can load judges from YAML file or add them manually:

**Option A: Load from YAML file (Recommended)**
```python
evaluator.load_judges_from_yaml(yaml_file="judges.yaml")
```

**Option B: Add judges manually**
```python
from meta_evaluator.judge import Judge

# Create and add judges manually
judge = Judge(
    id="custom_judge",
    llm_client="openai",
    model="gpt-4.1-mini",
    prompt="Your evaluation prompt here..."
)
evaluator.add_judge(judge)
```

### 6. Run judges

You can run judges synchronously or asynchronously.

**Option A: Sync**
```python
evaluator.run_judges()
```

**Option B: Async**
```python
import asyncio

async def run_evaluation():
    await evaluator.run_judges_async()

# Run the async function
asyncio.run(run_evaluation())
```

### 7. Load results

```python
# Load judge results
judge_results = evaluator.load_all_judge_results()

# Load human annotation results (project should minimally contain 1 set of human annotation)
human_results = evaluator.load_all_human_results()
```

### 8. Initialize metrics and run comparison

```python
from meta_evaluator.scores import MetricConfig, MetricsConfig
from meta_evaluator.scores.metrics import (
    AccuracyScorer,
    AltTestScorer, 
    CohensKappaScorer,
    TextSimilarityScorer,
)

# Create scorers
accuracy_scorer = AccuracyScorer()
alt_test_scorer = AltTestScorer()
cohens_kappa_scorer = CohensKappaScorer()
text_similarity_scorer = TextSimilarityScorer()

# Create metrics configuration
config = MetricsConfig(
    metrics=[
        MetricConfig(scorer=accuracy_scorer, task_names=["rejection"]),
        MetricConfig(scorer=alt_test_scorer, task_names=["rejection"]),
        MetricConfig(scorer=cohens_kappa_scorer, task_names=["rejection"]),
        MetricConfig(scorer=text_similarity_scorer, task_names=["explanation"]),
    ]
)

# Run comparison and save results
evaluator.compare(config, judge_results=judge_results, human_results=human_results)
```


## Human Annotation Platform

MetaEvaluator includes a built-in Streamlit-based interface for collecting human ground truth annotations.

### 1. Initialize task and data

```python
from meta_evaluator.data import DataLoader
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.meta_evaluator import MetaEvaluator

# Define annotation task
eval_task = EvalTask(
    task_schemas={
        "rejection": ["rejection", "not rejection"],
        "explanation": None,  # Optional explanation field
    },
    prompt_columns=["prompt"],
    response_columns=["llm_response"], 
    answering_method="structured",
)

# Load data for annotation
eval_data = DataLoader.load_csv(
    name="annotation_data",
    file_path="data/annotation_dataset.csv",
)
```

### 2. Launch annotation interface

```python
# Initialize evaluator
evaluator = MetaEvaluator(project_dir="project_dir")

# Add task and data
evaluator.add_eval_task(eval_task)
evaluator.add_data(eval_data)

# Launch Streamlit interface
evaluator.launch_annotator(port=8501)
```

### 3. Access and use interface

1. **Run script**: `uv run python metaevaluator_annotation_demo.py`
2. **Open browser**: Navigate to `http://localhost:8501`
3. **Annotate**: 
   - Enter annotator name/ID
   - Review prompts and responses  
   - Select classifications or provide text
   - Track progress and resume sessions

## Annotation Data Management

**Data Storage**: Annotations are automatically saved to:
```
project_dir/annotations/
├── annotation_run_{timestamp}_{annotator_id}_data.json
└── annotation_run_{timestamp}_{annotator_id}_metadata.json
```

**Loading Results**: Load human annotations for metrics comparison:
```python
human_results = evaluator.load_all_human_results()
evaluator.compare(config, judge_results=judge_results, human_results=human_results)
```

**Multi-annotator Support**: 
- Unique sessions per annotator
- Separate progress tracking
- Inter-annotator agreement calculation

## Available Metrics

MetaEvaluator supports various alignment metrics for evaluating judge performance. See [docs/metrics.md](docs/metrics.md) for detailed information about available metrics, usage examples, and selection guidelines.

## Project Structure

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

### Annotation Platform Examples
- `examples/metaevaluator_annotation_demo.py` - Human annotation interface example

### Evaluation Examples  
- `examples/rejection/run_evaluation.py` - Rejection detection evaluation
- `examples/rejection/run_evaluation_async.py` - Rejection detection evaluation