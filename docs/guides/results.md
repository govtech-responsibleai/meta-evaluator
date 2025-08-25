# Loading Results

Access judge evaluation results and human annotations.

## Load Results

```python
# Load both judge and human results  
judge_results = evaluator.load_all_judge_results()
human_results = evaluator.load_all_human_results()

print(f"Judges: {len(judge_results)}, Humans: {len(human_results)}")
```

## View Data Format

```python
# Check judge results data format
for judge_id, judge_result in judge_results.items():
    print(f"Judge: {judge_id}")
    print(judge_result.data)  # Shows the Polars DataFrame
    break

# Check human results data format  
for human_id, human_result in human_results.items():
    print(f"Human: {human_id}")
    print(human_result.data)  # Shows the Polars DataFrame
    break
```

## Result Files

Results are stored in your project directory:

```
my_project/
├── main_state.json             # Project configuration
├── data/
│   └── main_state_data.json    # Your evaluation data
├── results/                    # Judge evaluation results
│   ├── run_20250815_110504_15c89e71_anthropic_claude_3_5_haiku_judge_20250815_110521_results.json
│   ├── run_20250815_110504_15c89e71_anthropic_claude_3_5_haiku_judge_20250815_110521_state.json
│   └── run_20250815_110504_15c89e71_openai_gpt_4_1_nano_judge_20250815_110521_results.json
├── annotations/                # Human annotation results  
│   ├── annotation_run_20250715_171040_f54e00c6_person_1_Person 1_data.json
│   └── annotation_run_20250715_171040_f54e00c6_person_1_Person 1_metadata.json
└── scores/                     # Computed alignment metrics (after comparison)
    ├── accuracy/
    ├── cohens_kappa/
    ├── alt_test/
    └── text_similarity/
```