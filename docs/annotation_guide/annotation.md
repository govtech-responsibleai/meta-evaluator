# Setting up the Annotation Platform

Launch the Streamlit-based annotation interface to collect human evaluations for your dataset.

## Quick Setup

=== "Basic Annotation"

    ```python
    from meta_evaluator import MetaEvaluator
    
    # Load existing project with data and task
    evaluator = MetaEvaluator(project_dir="my_project", load=True) 
    
    # Launch annotation interface
    evaluator.launch_annotator(port=8501) 
    ```

=== "With ngrok (Remote Access)"

    ```python
    from meta_evaluator import MetaEvaluator
    
    # Load existing project with data and task
    evaluator = MetaEvaluator(project_dir="my_project", load=True) 
    
    # Enable internet access via ngrok tunnel
    evaluator.launch_annotator(
        port=8501,
        use_ngrok=True
    )
    ```

=== "Complete Setup Example"

    ```python
    from meta_evaluator import MetaEvaluator
    from meta_evaluator.data import DataLoader
    from meta_evaluator.eval_task import EvalTask
    
    # Initialize new project
    evaluator = MetaEvaluator(project_dir="annotation_project", load=False)
    
    # Load your data
    data = DataLoader.load_csv(
        name="evaluation_data",
        file_path="data/samples.csv"
    )
    
    # Define annotation task
    task = EvalTask(
        task_schemas={
            "quality": ["excellent", "good", "fair", "poor"],
            "relevance": ["relevant", "partially_relevant", "irrelevant"],
            "explanation": None  # Free-form text field
        },
        prompt_columns=["prompt"],
        response_columns=["response"],
        answering_method="structured",
        annotation_prompt="Please evaluate the AI response quality and relevance."
    )
    
    # Add to evaluator
    evaluator.add_data(data)
    evaluator.add_eval_task(task)
    
    # Save state and launch annotator
    evaluator.save_state()
    evaluator.launch_annotator(port=8501)
    ```

!!! warning "Dataset choice"
    - Use a small dataset (recommended: 20-50 rows of data)
    - Current implementation doesn't support saving progress midway, so users must complete all annotations in one session.

## Annotation Interface Features

The interface automatically presents all tasks from your `EvalTask` configuration in the UI:

```python
task = EvalTask(
    task_schemas={
        "quality": ["excellent", "good", "fair", "poor"],
        "relevance": ["relevant", "partially_relevant", "irrelevant"],
        "explanation": None,  # Free-form text field
    },
    prompt_columns=["prompt"],
    response_columns=["response"],
    answering_method="structured",
    annotation_prompt="Please evaluate the AI response quality and relevance.",
)
```

**Interface displays:**

- Required input for annotator name  
- Radio buttons for classification tasks (helpfulness, accuracy, safety)  
- Text area for free-form fields (explanation)  
- Progress tracking and navigation controls 

**Example of the platform:**
<figure markdown="span">
  ![Simple Annotation Interface](../assets/simple_annotation_interface.png)
</figure>

!!! note
    The annotation platform is slated for usability, functionality, and visual updates in the near future!


## Launch Configuration

### Port Settings

```python
# Default port (8501)
evaluator.launch_annotator()

# Custom port
evaluator.launch_annotator(port=8080)

# System will find available port if specified port is occupied
```

### Remote Access with ngrok

```python
# Basic ngrok tunnel
evaluator.launch_annotator(
    port=8501,
    use_ngrok=True
)

# Advanced ngrok with traffic policy
evaluator.launch_annotator(
    port=8501,
    use_ngrok=True,
    traffic_policy_file="ngrok_policy.yaml"
)
```

!!! note "ngrok Setup"
    - Requires ngrok installed and authenticated
    - Creates secure tunnel to your local annotation interface
    - Useful for remote annotators or team collaboration
    - **Traffic policy files** allow advanced configuration including login authentication, IP restrictions, custom headers, and more.   
        For examples and detailed configuration, see: [ngrok Traffic Policy Documentation](https://ngrok.com/docs/traffic-policy/)

## Using the Interface

### Starting Annotation

1. **Launch the interface**: Load your data and task, and run `evaluator.launch_annotator()`
2. **Open browser**: Navigate to `http://localhost:8501` 
3. **Enter annotator ID**: Provide your name or identifier
4. **Begin annotation**: Start evaluating samples


### Best Practices

!!! tip "Annotation Quality"
    - **Documentation**: Provide clear `annotation_prompt` in your EvalTask for the interface
    - **Multiple annotators**: Have 2-3 people annotate the same data for reliability
    - **Regular breaks**: Avoid annotation fatigue with scheduled breaks

!!! warning "Session Management"
    - Each annotator should use a unique ID for tracking
    - Administrator should coordinate annotation assignments to avoid duplication
    - Don't close browser tabs abruptly to avoid losing progress
    - Internet connection required for ngrok remote access

## Results and Output

### File Structure

After annotation, results are saved to your project directory:

```
my_project/
└── annotations/
    ├── annotation_run_20250715_171040_f54e00c6_person_1_Person 1_data.json
    ├── annotation_run_20250715_171040_f54e00c6_person_1_Person 1_metadata.json
    ├── annotation_run_20250715_171156_617ba2da_person_2_Person 2_data.json
    └── annotation_run_20250715_171156_617ba2da_person_2_Person 2_metadata.json
```

**File naming pattern:**
```
annotation_run_{timestamp}_{run_hash}_{annotator_id}_{display_name}_{data|metadata}.json
```

### Annotation Data Format

Each annotation run creates two files: a data file containing the actual annotations and a metadata file with run information.

**Data file example (`*_data.json`):**
```json
[
  {
    "sample_example_id": "sample_1",
    "original_id": "lg_auto_rt_successful_fn_en_60",
    "run_id": "annotation_run_20250715_163632_09d6cc37",
    "status": "success",
    "error_message": null,
    "error_details_json": null,
    "annotator_id": "h1",
    "annotation_timestamp": "2025-07-15 16:37:52.163517",
    "hateful": "FALSE",
    "insults": "FALSE",
    "sexual": "FALSE",
    "physical_violence": "FALSE",
    "self_harm": "self_harm",
    "all_other_misconduct": "all_other_misconduct"
  }
]
```

**Metadata file example (`*_metadata.json`):**
```json
{
  "run_id": "annotation_run_20250715_163632_09d6cc37",
  "task_schemas": {
    "hateful": ["FALSE", "hateful"],
    "insults": ["FALSE", "insults"],
    "sexual": ["FALSE", "sexual"],
    "physical_violence": ["FALSE", "physical_violence"],
    "self_harm": ["FALSE", "self_harm"],
    "all_other_misconduct": ["FALSE", "all_other_misconduct"]
  },
  "timestamp_local": "2025-07-15T16:49:46.684126",
  "total_count": 50,
  "succeeded_count": 50,
  "is_sampled_run": false,
  "data_file": "annotation_run_20250715_163632_09d6cc37_h1_H1_data.json",
  "data_format": "json",
  "annotator_id": "h1",
  "error_count": 0
}
```

### Loading Annotation Results

After collection, load annotations for analysis:

```python
# Load all human annotation results
human_results = evaluator.load_all_human_results()

print(f"Loaded {len(human_results)} human annotation result sets")

# Access individual annotator results
for result in human_results:
    print(f"Annotator: {result.annotator_id}")
    print(f"Samples annotated: {len(result.results_data)}")
```

## Troubleshooting

### Common Issues

=== "Port Already in Use"
    ```python
    # Let system find available port
    evaluator.launch_annotator()  # Will auto-detect free port
    
    # Or specify different port
    evaluator.launch_annotator(port=8080)
    ```

=== "Browser Doesn't Open"
    ```
    Manually navigate to: http://localhost:8501
    Check console output for exact URL and port
    ```

=== "ngrok Issues"
    ```bash
    # Ensure ngrok is installed and authenticated
    ngrok config check
    ngrok authtoken YOUR_TOKEN
    
    # Check ngrok logs in console output
    ```

=== "Lost Annotations"
    ```python
    # Check annotations directory
    evaluator.paths.annotations
    
    # Verify annotation files are present
    # Files are auto-saved on each submission
    ```
