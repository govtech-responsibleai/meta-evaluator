# Loading Data

Load your evaluation dataset using the DataLoader for multiple file formats. EvalData uses Polars DataFrames internally for fast, efficient data processing.

## Quick Setup

=== "CSV"

    ```python
    from meta_evaluator.data import DataLoader
    
    data = DataLoader.load_csv(
        id_column="example_id"  # Optional - auto-generated if not provided
        name="my_evaluation",
        file_path="my_data.csv",
    )
    ```

=== "JSON"

    ```python
    from meta_evaluator.data import DataLoader
    
    data = DataLoader.load_json(
        id_column="example_id"
        name="my_evaluation",
        file_path="my_data.json", 
    )
    ```

=== "Parquet"

    ```python
    from meta_evaluator.data import DataLoader
    
    data = DataLoader.load_parquet(
        id_column="example_id"
        name="my_evaluation", 
        file_path="my_data.parquet",
    )
    ```

=== "DataFrame"

    ```python
    import polars as pl
    from meta_evaluator.data import DataLoader
    
    df = pl.DataFrame({
        "prompt": ["What is AI?", "Explain ML"],
        "response": ["AI is...", "ML is..."]
    })
    
    data = DataLoader.load_from_dataframe(
        id_column=None  # Auto-generate IDs
        name="my_evaluation",
        data=df,
    )
    ```

## Required Data Structure

Your data must match the columns specified in your EvalTask:

```python
# If your EvalTask has:
task = EvalTask(
    prompt_columns=["user_input"],
    response_columns=["llm_response"], 
    # ...
)

# Your CSV must contain these columns:
```

```csv
user_input,llm_response
"What is 2+2?","The answer is 4"
"Hello there","Hi! How can I help you?"
```

## Arguments

### ID Column (`id_column`)

The `id_column` uniquely identifies each example:

```python
# Option 1: Use existing ID column
data = DataLoader.load_csv(
    id_column="example_id"  # Must exist in your CSV
    name="eval",
    file_path="data.csv",
)

# Option 2: Auto-generate IDs (recommended for simple datasets)
data = DataLoader.load_csv(
    id_column=None  # Creates "id" column: ["id-1", "id-2", "id-3", ...]
    name="eval",
    file_path="data.csv", 
)
```

### Data Name (`name`)

The `name` parameter provides a human-readable identifier for your dataset:

```python
data = DataLoader.load_csv(
    name="customer_feedback_evaluation",  # Descriptive name
    file_path="data.csv"
)

# Name is used in:
# - Project serialization and loading
# - Logging and error messages  
# - Result tracking and organization
```

### File Path (`file_path`)

The `file_path` specifies the location of your data file:

```python
# Relative paths (relative to current working directory)
data = DataLoader.load_csv(
    name="eval",
    file_path="data/samples.csv"  # Relative path
)

# Absolute paths
data = DataLoader.load_csv(
    name="eval", 
    file_path="/full/path/to/data.csv"  # Absolute path
)
```


## Stratified Sampling

Create a representative sample that preserves data distribution. 

```python
# Sample 20% while preserving topic distribution
sample_data = data.stratified_sample_by_columns(
    columns=["topic", "difficulty"], 
    sample_percentage=0.2,
    seed=42
)

# Add to MetaEvaluator
evaluator = MetaEvaluator(project_dir="quickstart_project", load=False)
evaluator.add_data(sample_data)
```
