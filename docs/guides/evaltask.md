# Defining the Evaluation Task

The `EvalTask` is the central configuration that defines **what to evaluate** and **how to parse responses**. It's the most important component to configure correctly as it determines the structure of your entire evaluation.

## Overview

EvalTask supports two main evaluation scenarios:

1. **Judge LLM Responses**: Evaluate responses from another LLM (prompt + response evaluation)
2. **Judge Text Content**: Evaluate arbitrary text content (response-only evaluation)

## Quick Setup

=== "Scenario 1: Judge LLM Responses (Prompt + Response)"

    ```python linenums="1" hl_lines="10 11"
    from meta_evaluator.eval_task import EvalTask

    # Evaluate chatbot responses for safety and helpfulness
    task = EvalTask(
        task_schemas={
            "safety": ["safe", "unsafe"],           # Classification task
            "helpfulness": ["helpful", "not_helpful"], # Classification task  
            "explanation": None                     # Free-form text
        },
        prompt_columns=["user_prompt"],     # Original prompt to the LLM
        response_columns=["chatbot_response"], # LLM response to evaluate
        answering_method="structured",      # Use JSON output parsing
        structured_outputs_fallback=True   # Fallback to XML if needed
    )
    ```

=== "Scenario 2: Judge Text Content (Response Only)"

    ```python linenums="1" hl_lines="8 9"
    # Evaluate text summaries for quality
    task = EvalTask(
        task_schemas={
            "accuracy": ["accurate", "inaccurate"],
            "coherence": ["coherent", "incoherent"],
            "summary": None  # Free-form numeric or text score
        },
        prompt_columns=None,               # No prompt context needed
        response_columns=["summary_text"], # Just evaluate the summary
        answering_method="structured"
    )
    ```

## Arguments


### Define columns (`prompt_columns` and `response_columns`)

Control which columns judges see during evaluation:

```python linenums="1"
# Scenario 1: Judge sees both prompt and response
prompt_columns=["user_input", "system_instruction"]  # Context
response_columns=["llm_output"]                      # What to evaluate

# Scenario 2: Judge sees only the content to evaluate
prompt_columns=None                    # No context
response_columns=["text_to_evaluate"]  # Direct evaluation
```

**Template Variable System:**

MetaEvaluator uses a template-based system where your prompt.md files can include placeholders like `{column_name}` that get automatically replaced with actual data. The available variables correspond to your `prompt_columns` and `response_columns`.

=== "Scenario 1: Prompt + Response Evaluation"

    **CSV Data:**
    ```csv
    user_input,system_instruction,llm_output
    "What is 2+2?","Be helpful","The answer is 4"
    ```

    **Configuration:**
    ```python linenums="1"
    prompt_columns=["user_input", "system_instruction"]
    response_columns=["llm_output"]
    ```

    **Example prompt.md:**
    ```markdown
    ## Instructions:
    Evaluate the LLM response for helpfulness.

    ## Context:
    User Input: {user_input}
    System Instruction: {system_instruction}

    ## Response to Evaluate:
    {llm_output}
    ```

    **Formatted prompt given to Judge:**
    ```
    ## Instructions:
    Evaluate the LLM response for helpfulness.

    ## Context:
    User Input: What is 2+2?
    System Instruction: Be helpful

    ## Response to Evaluate:
    The answer is 4
    ```

=== "Scenario 2: Response-Only Evaluation"

    **CSV Data:**
    ```csv
    text_to_evaluate
    "This is a summary of the research paper findings."
    ```

    **Configuration:**
    ```python linenums="1"
    prompt_columns=None
    response_columns=["text_to_evaluate"]
    ```

    **Example prompt.md:**
    ```markdown
    ## Instructions:
    Evaluate the quality of this summary.

    ## Text to Evaluate:
    {text_to_evaluate}
    ```

    **Formatted prompt given to Judge:**
    ```
    ## Instructions:
    Evaluate the quality of this summary.

    ## Text to Evaluate:
    This is a summary of the research paper findings.
    ```


### Task Schemas (`task_schemas`)

The `task_schemas` dictionary maps task names to their allowed outcomes:

```python linenums="1"
task_schemas = {
    # Classification tasks (predefined options)
    "toxicity": ["toxic", "non_toxic"],
    "relevance": ["relevant", "irrelevant", "partially_relevant"],
    
    # Free-form tasks (open-ended responses)
    "explanation": None,
    "how_confident": None
}
```

**Classification Tasks**: Use a list of strings for predefined outcomes  

- Minimum 2 outcomes required  
- Judges must choose from these exact options  
- Examples: sentiment, safety, relevance ratings  

**Free-form Tasks**: Use `None` for open-ended responses    

- No restrictions on judge responses  
- Examples: explanations, detailed feedback  


!!! tip "Add Free-form Tasks for Context and Explainability"
    Include explanation fields to understand judge reasoning:
    ```python linenums="1"
    task_schemas = {
        "is_helpful": ["helpful", "not_helpful"],
        "explanation": None  # Why this classification?
    }
    ```

### Required Tasks (`required_tasks`)

The `required_tasks` parameter controls which tasks must be completed for a valid annotation or judge response.

**Default Behavior** (when `required_tasks` is not specified):

- All **classification tasks** (non-`None` schemas) are **required**
- All **free-form tasks** (`None` schemas) are **not required**

=== "Default Behavior"

    ```python linenums="1"hl_lines="9"
    # Default behavior example
    task = EvalTask(
        task_schemas={
            "safety": ["safe", "unsafe"],        # Required by default
            "helpfulness": ["helpful", "not_helpful"],  # Required by default
            "explanation": None,                 # NOT required by default (free-form)
            "notes": None                        # NOT required by default (free-form)
        },
        # required_tasks not specified - uses default behavior
        prompt_columns=["user_prompt"],
        response_columns=["chatbot_response"],
        answering_method="structured"
    )
    ```

=== "Custom Required Tasks"

    ```python linenums="1" hl_lines="8"
    # Custom behavior example
    task = EvalTask(
        task_schemas={
            "safety": ["safe", "unsafe"],
            "helpfulness": ["helpful", "not_helpful"],
            "explanation": None,  # Free-form
        },
        required_tasks=["safety"],  # Only `safety` required
        prompt_columns=["user_prompt"],
        response_columns=["chatbot_response"],
        answering_method="structured"
    )
    ```

**Impact on Annotation Interface:**

In the Streamlit annotation interface, required fields are marked with a red asterisk (*) and must be filled before the annotation is auto-saved.

**Impact on Judge Results:**

For judge evaluations, only the required tasks need to be successfully parsed for a result to be marked as successful. 


### Answer Parsing Methods (`answering_method`)

Three parsing methods with different trade-offs:

=== "Structured (Recommended)"

    ```python linenums="1"
    answering_method="structured"
    structured_outputs_fallback=True  # Fallback to other methods if unsupported
    ```

    **Pros**: Most reliable, cleanest parsing, best model support  
    **Cons**: Newer feature, not supported by all models  
    **Best for**: Production use with modern models
    
    **Fallback sequence** (when `structured_outputs_fallback=True`): structured → instructor → xml
 
=== "Instructor"

    ```python linenums="1"
    answering_method="instructor"
    structured_outputs_fallback=True  # Fallback to other methods if unsupported
    ```

    **Pros**: Good compatibility, structured validation  
    **Cons**: Additional dependency, model-specific implementation  
    **Best for**: When you need structured outputs with older models
    
    **Fallback sequence** (when `structured_outputs_fallback=True`): instructor → structured → xml

=== "XML"

    ```python linenums="1"
    answering_method="xml"
    structured_outputs_fallback=True  # Fallback to other methods if unsupported
    ```

    **Pros**: Universal compatibility, works with any model  
    **Cons**: More prone to parsing errors, verbose output  
    **Best for**: Maximum compatibility, legacy models
    
    **Fallback sequence**: None

!!! tip "Enable Fallback for Production"
    Always enable fallback to maximise Judge completion.


### Skip Function to Filter Data Rows (`skip_function`)

```python linenums="1"
def skip_empty_responses(row):
    return len(row.get("llm_response", "").strip()) == 0

task = EvalTask(
    # ... other config ...
    skip_function=skip_empty_responses
)
```

!!! warning "Skip Function Serialization"
    Currently, **skip functions are not saved** when EvalTask is serialized/deserialized. When loading a saved project:
    
    ```python linenums="1"
    # Load existing project
    evaluator = MetaEvaluator(project_dir="my_project", load=True)
    
    # Skip function resets to default. You must reassign it after loading:
    evaluator.eval_task.skip_function = skip_empty_responses
    ```

### Annotation Prompt for Human Interface (`annotation_prompt`)

Customize the prompt shown to human annotators:

```python linenums="1"
task = EvalTask(
    # ... other config ...
    annotation_prompt="Please evaluate this response for toxicity and helpfulness. Consider both content and tone."
)
```

## Real-World Examples

===  "Content Moderation Pipeline"

    ```python linenums="1"
    moderation_task = EvalTask(
        task_schemas={
            "toxicity": ["toxic", "borderline", "safe"],
            "harassment": ["harassment", "no_harassment"], 
            "violence": ["violent", "non_violent"],
            "explanation": None
        },
        prompt_columns=["user_message"],      # Original user input
        response_columns=["content_to_check"], # Content that might violate policy
        answering_method="structured",
        structured_outputs_fallback=True,
        annotation_prompt="Evaluate this content for policy violations."
    )
    ```

=== "Multi-turn Conversation Evaluation"

    ```python linenums="1"
    conversation_task = EvalTask(
        task_schemas={
            "coherence": ["coherent", "somewhat_coherent", "incoherent"],
            "helpfulness": ["very_helpful", "helpful", "not_helpful"],
            "factuality": ["factual", "mostly_factual", "inaccurate"],
            "improvement_suggestions": None
        },
        prompt_columns=["conversation_history", "user_query"],
        response_columns=["assistant_response"],
        answering_method="structured",
        structured_outputs_fallback=True
    )
    ```

=== "Research Paper Evaluation"

    ```python linenums="1"
    research_task = EvalTask(
        task_schemas={
            "methodology_quality": ["excellent", "good", "fair", "poor"],
            "novelty": ["highly_novel", "somewhat_novel", "incremental", "not_novel"],
            "clarity": ["very_clear", "clear", "unclear", "very_unclear"],
            "detailed_feedback": None,
        },
        prompt_columns=None,  # No prompt needed
        response_columns=["paper_abstract", "methodology_section"],
        answering_method="structured",
        structured_outputs_fallback=True,
        annotation_prompt="Please evaluate this research paper on methodology, novelty, and clarity. Provide detailed feedback."
    )
    ```
