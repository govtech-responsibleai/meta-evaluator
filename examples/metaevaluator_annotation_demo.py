"""MetaEvaluator Annotation Demo.

This script demonstrates how to use the MetaEvaluator class to launch the Streamlit
annotation interface. This shows the integration between the MetaEvaluator and the
StreamlitLauncher.

Usage:
    python examples/metaevaluator_annotation_demo.py

This will:
1. Create a MetaEvaluator instance
2. Load sample evaluation data
3. Configure an evaluation task
4. Launch the Streamlit annotation interface
5. Open your browser to http://localhost:8501

Try annotating a few samples and see the results saved to the project's annotations directory.
"""

import polars as pl
from pathlib import Path

from meta_evaluator.meta_evaluator import MetaEvaluator
from meta_evaluator.eval_task import EvalTask
from meta_evaluator.data import EvalData


def create_sample_data():
    """Create sample evaluation data for demonstration.

    Returns:
        EvalData: A sample evaluation data with:
            - id: Unique identifier for each sample
            - question: The question to evaluate
            - response: The AI response to evaluate
    """
    data = pl.DataFrame(
        {
            "id": [f"sample_{i}" for i in range(1, 6)],
            "question": [
                "What is the capital of France?",
                "How do you make a cake?",
                "What is machine learning?",
                "Explain quantum physics",
                "What are the benefits of exercise?",
            ],
            "response": [
                "The capital of France is Paris.",
                "To make a cake, you need flour, eggs, sugar, and butter. Mix them together and bake.",
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                "Quantum physics is a fundamental theory in physics that describes nature at the smallest scales.",
                "Exercise has many benefits including improved cardiovascular health and mental well-being.",
            ],
        }
    )

    return EvalData(
        id_column="id",
        data=data,
        name="sample_qa_data",
    )


def create_eval_task():
    """Create a sample evaluation task for demonstration.

    Returns:
        EvalTask: A sample evaluation task for assessing response quality.
    """
    return EvalTask(
        task_schemas={
            "accuracy": ["correct", "incorrect", "partially_correct"],
            "helpfulness": [
                "very_helpful",
                "helpful",
                "somewhat_helpful",
                "not_helpful",
            ],
            "clarity": ["very_clear", "clear", "somewhat_clear", "unclear"],
        },
        prompt_columns=["question"],
        response_columns=["response"],
        answering_method="structured",
        annotation_prompt="Please evaluate the AI response based on the given criteria.",
    )


def main():
    """Run the MetaEvaluator annotation demo."""
    print("🚀 Starting MetaEvaluator Annotation Demo")
    print("=" * 50)

    # Create project directory
    project_dir = Path("./examples/metaevaluator_demo_project")
    print(f"📁 Project directory: {project_dir.absolute()}")

    # Initialize MetaEvaluator
    evaluator = MetaEvaluator(str(project_dir))
    print("✅ MetaEvaluator initialized")

    # Create and add sample data
    eval_data = create_sample_data()
    evaluator.add_data(eval_data)
    print(f"📊 Added evaluation data: {eval_data.name} ({len(eval_data.data)} samples)")

    # Create and add evaluation task
    eval_task = create_eval_task()
    evaluator.add_eval_task(eval_task)
    task_names = list(eval_task.task_schemas.keys())
    print(
        f"🎯 Added evaluation task with {len(task_names)} criteria: {', '.join(task_names)}"
    )

    print("\n" + "=" * 50)
    print("\n🌐 Launching Streamlit annotation interface...")
    print("   Your browser should open automatically")
    print("   If not, navigate to: http://localhost:8501")
    print("   Annotations will be saved to the project's annotations directory.")
    print("   Press Ctrl+C to stop the demo.")
    print("=" * 50)

    try:
        # Launch the annotation interface
        evaluator.launch_annotator(port=8501)

    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user.")
        print(
            f"📁 Check the annotations directory for your results: {evaluator.paths.annotations}"
        )
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure you have all dependencies installed:")
        print("   pip install streamlit polars")


if __name__ == "__main__":
    main()
