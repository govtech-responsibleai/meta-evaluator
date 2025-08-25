# Defining the Evaluation Task

The `EvalTask` is the central configuration that defines **what to evaluate** and **how to parse responses**. It's the most important component to configure correctly as it determines the structure of your entire evaluation.

## Overview

EvalTask supports two main evaluation scenarios:

1. **Judge LLM Responses**: Evaluate responses from another LLM (prompt + response evaluation)
2. **Judge Text Content**: Evaluate arbitrary text content (response-only evaluation)

## Quick Setup

=== "Scenario 1: Judge LLM Responses (Prompt + Response)"

    ```python
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

    ```python
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

```python
# Scenario 1: Judge sees both prompt and response
prompt_columns=["user_input", "system_instruction"]  # Context
response_columns=["llm_output"]                      # What to evaluate

# Scenario 2: Judge sees only the content to evaluate
prompt_columns=None                    # No context
response_columns=["text_to_evaluate"]  # Direct evaluation
```

**Examples with formatted prompts:**

=== "Scenario 1: Prompt + Response Evaluation"

    CSV Data:
    ```csv
    user_input,system_instruction,llm_output
    "What is 2+2?","Be helpful","The answer is 4"
    ```

    Configuration:
    ```python
    prompt_columns=["user_input", "system_instruction"]
    response_columns=["llm_output"]
    ```

    Formatted prompt given to Judge:
    ```
    The prompts to be evaluated are user_input, system_instruction.
    user_input: What is 2+2?
    system_instruction: Be helpful

    The responses to be evaluated are llm_output.
    llm_output: The answer is 4
    ```

=== "Scenario 2: Response-Only Evaluation"

    **CSV Data:**
    ```csv
    text_to_evaluate
    "This is a summary of the research paper findings."
    ```

    **Configuration:**
    ```python
    prompt_columns=None
    response_columns=["text_to_evaluate"]
    ```

    **Formatted prompt given to Judge:**
    ```
    The texts to be evaluated are text_to_evaluate.
    text_to_evaluate: This is a summary of the research paper findings.
    ```


### Task Schemas (`task_schemas`)

The `task_schemas` dictionary maps task names to their allowed outcomes:

```python
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
    ```python
    task_schemas = {
        "is_helpful": ["helpful", "not_helpful"],
        "explanation": None  # Why this classification?
    }
    ```

### Answer Parsing Methods (`answering_method`)

Three parsing methods with different trade-offs:

=== "Structured (Recommended)"

    ```python
    answering_method="structured"
    structured_outputs_fallback=True  # Fallback to other methods if unsupported
    ```

    **Pros**: Most reliable, cleanest parsing, best model support  
    **Cons**: Newer feature, not supported by all models  
    **Best for**: Production use with modern models
    
    **Fallback sequence** (when `structured_outputs_fallback=True`): structured → instructor → xml
 
=== "Instructor"

    ```python
    answering_method="instructor"
    structured_outputs_fallback=True  # Fallback to other methods if unsupported
    ```

    **Pros**: Good compatibility, structured validation  
    **Cons**: Additional dependency, model-specific implementation  
    **Best for**: When you need structured outputs with older models
    
    **Fallback sequence** (when `structured_outputs_fallback=True`): instructor → structured → xml

=== "XML"

    ```python
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

```python
def skip_empty_responses(row):
    return len(row.get("llm_response", "").strip()) == 0

task = EvalTask(
    # ... other config ...
    skip_function=skip_empty_responses
)
```

!!! warning "Skip Function Serialization"
    Currently, **skip functions are not saved** when EvalTask is serialized/deserialized. When loading a saved project:
    
    ```python
    # Load existing project
    evaluator = MetaEvaluator(project_dir="my_project", load=True)
    
    # Skip function resets to default. You must reassign it after loading:
    evaluator.eval_task.skip_function = skip_empty_responses
    ```

### Annotation Prompt for Human Interface (`annotation_prompt`)

Customize the prompt shown to human annotators:

```python
task = EvalTask(
    # ... other config ...
    annotation_prompt="Please evaluate this response for toxicity and helpfulness. Consider both content and tone."
)
```

## Real-World Examples

===  "Content Moderation Pipeline"

    ```python
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

    ```python
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

    ```python
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
