# Judge Configuration

Configure LLM judges to evaluate your data using YAML files and prompt templates.

## Quick Setup

### YAML Configuration

Create `judges.yaml`:
```yaml
judges:
  - id: gpt_4_judge
    llm_client: openai
    model: gpt-4o-mini
    prompt_file: ./prompt.md
```

### Prompt File

Create `prompt.md`:
```markdown
Evaluate whether the response is helpful or not helpful.

For each evaluation, provide:
1. **helpfulness**: "helpful" or "not helpful"
2. **explanation**: Brief reasoning for your classification

A response is "helpful" if it:
- Directly addresses the user's question
- Provides accurate and relevant information
- Is clear and easy to understand
```

### Load Judges

```python
from meta_evaluator import MetaEvaluator

evaluator = MetaEvaluator(project_dir="my_project")

# Load judges from YAML
evaluator.load_judges_from_yaml(
    yaml_file="judges.yaml",
    on_duplicate="skip",  # or "overwrite", "error"
    async_mode=True       # Enable async evaluation
)
```

## YAML Structure

Each judge requires these fields:

```yaml
judges:
  - id: unique_judge_identifier        # Required: alphanumeric + underscores only
    llm_client: provider_name          # Required: openai, anthropic, azure, etc.
    model: model_name                  # Required: specific model name
    prompt_file: ./path/to/prompt.md   # Required: relative to YAML file location, or absolute path
```


## Environment Variables

Set API keys for your providers:

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic 
export ANTHROPIC_API_KEY="your-key"

# Azure
export AZURE_API_KEY="your-key"
export AZURE_API_BASE="your-endpoint"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```


## Supported Providers

!!! warning
    Currently, only LLMs covered by LiteLLM are supported. Custom Judges (ability to add other model types) will be implemented in the future.

Via LiteLLM integration, supports 100+ providers. Check the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for complete provider list and model naming conventions.
Some examples: 

=== "OpenAI"

    ```yaml
    - id: openai_judge
      llm_client: openai
      model: gpt-4o-mini
      prompt_file: ./prompt.md
    ```

=== "Anthropic"

    ```yaml
    - id: anthropic_judge
      llm_client: anthropic
      model: claude-3-5-haiku-latest
      prompt_file: ./prompt.md
    ```

=== "Azure OpenAI"

    ```yaml
    - id: azure_judge
      llm_client: azure
      model: gpt-4o-mini-2024-07-18
      prompt_file: ./prompt.md
    ```

=== "AWS Bedrock"

    ```yaml
    - id: bedrock_judge
      llm_client: bedrock
      model: anthropic.claude-3-haiku-20240307-v1:0
      prompt_file: ./prompt.md
    ```

=== "Google Vertex AI"

    ```yaml
    - id: vertex_judge
      llm_client: vertex_ai
      model: gemini-2.5-flash-lite
      prompt_file: ./prompt.md
    ```
    
    **Setup**: Authenticate with Google Cloud:
    ```bash
    gcloud auth application-default login --no-launch-browser
    ```

=== "HuggingFace"

    ```yaml
    - id: together_judge
      llm_client: huggingface/together # Specify Inference Provider
      model: openai/gpt-oss-20b
      prompt_file: ./prompt.md
    ```

=== "OpenRouter"

    ```yaml
    - id: openrouter_judge
      llm_client: openrouter
      model: qwen/qwen-2.5-72b-instruct
      prompt_file: ./prompt.md
    ```

=== "xAI"

    ```yaml
    - id: xai_judge
      llm_client: xai
      model: grok-3-mini
      prompt_file: ./prompt.md
    ```

=== "Groq"

    ```yaml
    - id: groq_judge
      llm_client: groq
      model: llama-3.1-8b-instant
      prompt_file: ./prompt.md
    ```

=== "Fireworks AI"

    ```yaml
    - id: fireworks_judge
      llm_client: fireworks_ai
      model: kimi-k2-instruct
      prompt_file: ./prompt.md
    ```

## Writing Effective Prompts

Your prompt file should match your EvalTask schema:

```python
# If your EvalTask has:
task_schemas = {
    "toxicity": ["toxic", "non_toxic"],
    "explanation": None
}
```

Ideally, your prompt should specify the outputs you want:

```markdown
Evaluate the content for toxicity.

You must provide:
1. **toxicity**: Either "toxic" or "non_toxic" 
2. **explanation**: Brief reasoning for your classification

Guidelines:
- "toxic" if content contains harmful, offensive, or inappropriate material
- "non_toxic" if content is safe and appropriate
```

!!! tip "Match Task Schema"
    The prompt will be prepended with the task schema when processing. See [Defining the evaluation task](evaltask.md#define-columns-prompt_columns-and-response_columns) for more information.  
    Nonetheless, try to ensure your prompt specifies the exact task names from your EvalTask configuration.
    

## Arguments

Control how judges are loaded and handle duplicates:

```python
evaluator.load_judges_from_yaml(
    yaml_file="judges.yaml",      # Path to YAML configuration file
    on_duplicate="skip",          # How to handle duplicate judge IDs
    async_mode=True               # Enable async evaluation capabilities
)
```

### Control how judges handle duplicates (`on_duplicate`)

=== "skip (Recommended)"

    ```python
    on_duplicate="skip"
    ```
    
    - **Skip** loading judges with IDs that already exist
    - **Behavior**: Existing judges remain unchanged, only new judges added

=== "overwrite"

    ```python
    on_duplicate="overwrite"
    ```
    
    - **Replace** existing judges with same ID
    - **Behavior**: Completely replaces judge configuration (model, prompt, etc.)

### Control whether to run judges asynchronously (`async_mode`)

```python
# Enable async evaluation (recommended)
async_mode=True   # Allows concurrent judge evaluation for faster processing

# Disable async (synchronous only)  
async_mode=False  # Sequential evaluation, slower but simpler debugging
```
