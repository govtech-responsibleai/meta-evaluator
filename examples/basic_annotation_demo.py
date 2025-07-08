"""Basic Human Annotation Demo.

This script demonstrates how to use the human annotation system for evaluating
text responses. Run this script to see the annotation interface in action.

Usage:
    python examples/basic_annotation_demo.py

This will:
1. Create sample evaluation data
2. Define annotation tasks
3. Launch the Streamlit annotation interface
4. Open your browser to http://localhost:8501

Try annotating a few samples and see the results saved to ./examples/demo_annotations/.
"""

import polars as pl
from pathlib import Path

from meta_evaluator.annotator.launcher import StreamlitLauncher
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.data import EvalData


def create_sample_data():
    """Create sample evaluation data for demonstration.

    Returns:
        pl.DataFrame: A sample evaluation data with:
            - id: Unique identifier for each sample
            - question: The question to evaluate
            - response: The AI response to evaluate
    """
    df = pl.DataFrame(
        {
            "id": [f"sample_{i}" for i in range(1, 6)],
            "question": [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Explain the difference between AI and ML",
                "What are the benefits of automation?",
                "How will AI impact jobs in the future?",
            ],
            "response": [
                "AI is a technology that enables machines to simulate human intelligence and perform tasks like learning, reasoning, and problem-solving.",
                "Machine learning uses algorithms to analyze data, learn patterns, and make predictions or decisions without explicit programming.",
                "AI is the broader concept of machines being able to carry out tasks in a smart way, while ML is a subset of AI that learns from data.",
                "Automation increases efficiency, reduces human error, cuts costs, and allows humans to focus on more creative and strategic work.",
                "AI will likely automate some jobs but also create new opportunities, requiring workers to adapt and learn new skills.",
            ],
        }
    )
    return df


def create_evaluation_task():
    """Define the annotation tasks for evaluators.

    Returns:
        EvalTask: A configured evaluation task with:
            - Structured annotation for accuracy, clarity, and helpfulness
            - Free-form text field for comments
    """
    return EvalTask(
        task_schemas={
            "accuracy": ["accurate", "partially_accurate", "inaccurate"],
            "clarity": ["very_clear", "clear", "unclear", "very_unclear"],
            "helpfulness": [
                "very_helpful",
                "helpful",
                "somewhat_helpful",
                "not_helpful",
            ],
            "comments": None,  # Free-form text field
        },
        prompt_columns=["question"],
        response_columns=["response"],
        answering_method="structured",
        annotation_prompt="""
        Please evaluate the quality of the AI response to each question.
        
        Rate the response on the following criteria:
        • **Accuracy**: Is the information factually correct?
        • **Clarity**: Is the response easy to understand?
        • **Helpfulness**: Does it effectively answer the question?
        
        Feel free to add specific comments or suggestions for improvement.
        """,
    )


def main():
    """Run the basic annotation demo."""
    print("🚀 Starting Human Annotation Demo")
    print("=" * 50)

    # Create sample data
    print("📊 Creating sample evaluation data...")
    df = create_sample_data()
    eval_data = EvalData(name="ai_qa_demo", data=df, id_column="id")
    print(f"   Created dataset with {len(df)} samples")

    # Define annotation tasks
    print("📝 Setting up annotation tasks...")
    eval_task = create_evaluation_task()
    task_names = list(eval_task.task_schemas.keys())
    print(f"   Tasks: {', '.join(task_names)}")

    # Setup annotations directory
    annotations_dir = Path(__file__).parent / "demo_annotations"
    annotations_dir.mkdir(exist_ok=True)
    print(f"💾 Annotations will be saved to: {annotations_dir.absolute()}")

    # Launch the annotation interface
    print("\n🌐 Launching Streamlit annotation interface...")
    print("   Your browser should open automatically")
    print("   If not, navigate to: http://localhost:8501")
    print("\n📋 Instructions:")
    print("   1. Enter your name when prompted")
    print("   2. Read each question and response pair")
    print("   3. Rate the accuracy, clarity, and helpfulness")
    print("   4. Add any comments in the text field")
    print("   5. Click 'Submit Annotation' for each sample")
    print("   6. Use 'Export Annotations' when finished")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        launcher = StreamlitLauncher(
            eval_data=eval_data,
            eval_task=eval_task,
            annotations_dir=str(annotations_dir),
            port=8501,
        )
        launcher.launch()

    except KeyboardInterrupt:
        print(
            "\n\n👋 Demo stopped. Check the examples/demo_annotations/ folder for your results!"
        )
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure you have all dependencies installed:")
        print("   pip install streamlit polars")


if __name__ == "__main__":
    main()
