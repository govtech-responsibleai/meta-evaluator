# LLM Client Documentation

## Overview

Provider-agnostic LLM client with robust response parsing. Designed for the meta-evaluator project where reliable data extraction from variable LLM outputs is critical.

## Quick Start

```python
from meta_evaluator.LLMClient import OpenAIClient, OpenAIConfig, Message, RoleEnum

# Setup
config = OpenAIConfig(api_key="...", default_model="gpt-4")
client = OpenAIClient(config)

# Basic usage
messages = [Message(role=RoleEnum.USER, content="Hello")]
response = client.prompt(messages)
print(response.content)
```

## Core Architecture

**Abstract Base Classes**:
- `LLMClient`: Provider-agnostic interface
- `LLMClientConfig`: Provider configuration base

**Provider Implementations**: Extend these bases for each LLM service (OpenAI, Anthropic, etc.)

## Adding New Providers

This is the most common extension. Here's the pattern:

```python
class MyProviderClient(LLMClient):
    @property
    def enum_value(self) -> LLMClientEnum:
        return LLMClientEnum.MY_PROVIDER
    
    def _prompt(self, model: str, messages: list[Message], get_logprobs: bool) -> tuple[str, LLMUsage]:
        # 1. Convert our Message objects to provider format
        provider_messages = self._convert_messages(messages)
        
        # 2. Make API call
        response = my_provider_api.chat(
            model=model,
            messages=provider_messages,
            logprobs=get_logprobs
        )
        
        # 3. Extract text and usage stats
        text = response.content
        usage = LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens, 
            total_tokens=response.usage.total_tokens
        )
        
        return text, usage
    
    def _prompt_with_structured_response(
        self, messages: list[Message], response_model: Type[BaseModel], model: str
    ) -> tuple[BaseModel, LLMUsage]:
        # Use instructor internally for better reliability
        import instructor
        client = instructor.patch(self.provider_client)
        
        response = client.chat.completions.create(
            model=model,
            messages=self._convert_messages(messages),
            response_model=response_model
        )
        
        usage = LLMUsage(...)  # Extract from response
        return response, usage

class MyProviderConfig(LLMClientConfig):
    api_key: str
    default_model: str = "my-provider-default"
    supports_structured_output: bool = True  # If using instructor
    supports_logprobs: bool = True  # If provider supports logprobs
    # ... other provider-specific config
    
    def _prevent_instantiation(self) -> None:
        pass
```

**Key Points**:
- `_prompt()` does the actual API call - everything else is handled by the base class
- Always return `(text: str, usage: LLMUsage)` 
- For structured outputs, use [Instructor](https://github.com/567-labs/instructor) internally - it's more reliable than native API structured output
- Handle provider-specific errors and let them bubble up as exceptions (base class wraps them)

## Token Tracking

Token usage is automatically tracked and logged for every request:

```python
response = client.prompt(messages)

# Usage stats are always available
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")  
print(f"Total tokens: {response.usage.total_tokens}")

# Usage is also logged automatically at INFO level
# "Output usage: LLMUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40)"
```

**Cost Tracking**: Use these stats to calculate costs per provider's pricing model. All LLM interactions return usage data consistently across providers.

## Log Probs

Get token-level probability information when supported by the provider:

```python
# Check if provider supports logprobs
if client.config.supports_logprobs:
    response = client.prompt(messages, get_logprobs=True)
    
    if response.logprobs:
        for token_info in response.logprobs.content:
            print(f"Token: {token_info.token}, Probability: {token_info.logprob}")
            
            # See alternative tokens considered
            if token_info.top_logprobs:
                for alt in token_info.top_logprobs:
                    print(f"  Alternative: {alt.token} ({alt.logprob})")
```

**Use Cases**: Model confidence analysis, debugging unexpected outputs, research on model behavior. Not all providers support logprobs - check `config.supports_logprobs` first.

## XML Parsing Philosophy

**The Problem**: LLMs are inconsistent. They'll give you:
```xml
<!-- Asked for one status -->
<status>active</status><status>pending</status>

<!-- Typos in values -->
<priority>urgent</priority> <!-- when you only allow ["low", "medium", "high"] -->

<!-- Missing tags entirely -->
<task_id>123</task_id> <!-- where's the status? -->
```

**Our Solution**: Parse what works, report what doesn't, avoid retries on validation errors.

```python
# Configure what you expect
configs = [
    TagConfig(
        name="status", 
        allowed_values=["active", "inactive"], 
        cardinality="one"
    ),
    TagConfig(
        name="tags",
        cardinality="many"  # Accept multiple values
    )
]

# Parse LLM response
parse_result, llm_response = client.prompt_with_xml_tags(messages, configs)

if parse_result.success:
    # All tags parsed successfully
    status = parse_result.data["status"]
    tags = parse_result.data["tags"]
elif parse_result.partial_success:
    # Some worked, some didn't - use what you can
    for tag_name, value in parse_result.data.items():
        process_valid_data(tag_name, value)
    
    # Log what failed for debugging, but don't retry
    for error in parse_result.errors:
        log.warning(f"XML parsing error: {error}")
else:
    # Complete failure - this is rare
    log.error("No valid XML tags found")
```

**Why This Approach**:
- **Avoid wasted retries**: Validation errors aren't worth retrying
- **Graceful degradation**: Use partial data when possible  
- **Clear error tracking**: Know exactly what failed and why
- **Production resilience**: System keeps working with partial data

## Structured Outputs

Structured outputs are handled transparently by the client implementation. Provider implementations should use [Instructor](https://github.com/567-labs/instructor) internally for better reliability than native API structured output.

```python
# Clean interface for users
structured_response, llm_response = client.prompt_with_structured_response(
    messages, 
    response_model=MyPydanticModel
)

# Instructor complexity is hidden inside the provider implementation
```

Users don't need to know about Instructor - it's an implementation detail that makes structured outputs more reliable.

## Error Handling

```python
try:
    response = client.prompt(messages)
except LLMValidationError as e:
    # Client-side validation failed (empty messages, etc.)
    log.error(f"Invalid request: {e}")
except LLMAPIError as e:
    # Provider API failed (network, auth, etc.)
    log.error(f"API error: {e}")
    log.debug(f"Original error: {e.original_error}")
```

All errors include provider context and detailed messages.

## Testing Patterns

```python
def test_my_feature(mocker):
    # Mock the internal _prompt method
    mock_response = "Test LLM response"
    mock_usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    
    mocker.patch.object(client, "_prompt", return_value=(mock_response, mock_usage))
    
    # Test your logic
    response = client.prompt(messages)
    assert response.content == mock_response
```

Mock `_prompt()` rather than external APIs - cleaner and more reliable.

## Common Issues

**XML Parsing Configuration**: 
- `cardinality="one"` expects exactly one tag, but has strategies for handling multiples
- `allowed_values=None` means any string is valid (freeform text)
- Parsing errors don't raise exceptions - check `parse_result.errors`

**Provider Implementation**:
- Must implement both `_prompt()` and `enum_value` property
- For structured outputs, implement `_prompt_with_structured_response()` using Instructor
- `_prompt()` gets called by all public methods (`prompt()`, `prompt_with_xml_tags()`, etc.)
- Base class handles all logging, error wrapping, and response construction

**Beartype**: Runtime type checking is enabled package-wide. All function parameters are validated automatically.

