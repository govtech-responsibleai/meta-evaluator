# MetaEvaluator

A comprehensive Python framework for evaluating LLM-as-a-Judge systems by comparing judge outputs with human annotations and calculating alignment metrics.

## What is MetaEvaluator?

MetaEvaluator helps you answer the critical question: **"How well do LLM judges align with human judgment?"**

Given an evaluation task and dataset, MetaEvaluator: 

- Runs multiple LLM judges across different providers (OpenAI, Anthropic, Google, AWS, etc.)
- Collects human annotations through a built-in web interface
- Calculates alignment metrics (Accuracy, Cohen's Kappa, Alt-Test, Text Similarity)
- Provides detailed comparison and analysis

## When to Use MetaEvaluator?

- When evaluating the quality of your LLM-as-a-judge
- Research on LLM evaluation capabilities, to compare performance across various LLMs and system prompts.

## Key Features

### 1. Easy LLM Judge processing
- LiteLLM Integration: Support for 100+ LLM providers with unified API
- Supports Structured Outputs/Instructor/XML parsing for automatic JSON parsing
- Load multiple judges through simplified YAML Configurations.

### 2. Built-in Human Annotation Platform
- **Streamlit Interface**: Clean, intuitive annotation workflow
- **Multi-annotator Support**: Separate sessions with progress tracking
- **Resume Capability**: Continue annotation sessions across multiple visits
- **Export Options**: JSON format for analysis and sharing

### 3. Comprehensive Alignment Metrics
- **Classification Metrics**: Accuracy, Cohen's Kappa for agreement analysis
- **Statistical Testing**: Alt-Test for advantage comparison
- **Text Similarity**: Semantic similarity for free-form responses
- **Custom Metrics**: Extensible framework for your own evaluation methods



Ready to start evaluating your LLM judges? Head to the [Tutorial](tutorial.md)!