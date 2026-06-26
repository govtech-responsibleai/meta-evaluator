"""FastAPI application factory for annotation interface."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from meta_evaluator.annotator.api.routes import (
    annotations,
    export,
    samples,
    session,
    task,
)
from meta_evaluator.annotator.api.state import SessionStore
from meta_evaluator.data import EvalData
from meta_evaluator.eval_task import EvalTask


def create_app(
    eval_task: EvalTask,
    eval_data: EvalData,
    annotations_dir: str,
    static_dir: str | None = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        eval_task: Task configuration.
        eval_data: Evaluation data.
        annotations_dir: Directory for saving annotations.
        static_dir: Optional path to frontend static build directory.

    Returns:
        FastAPI: Configured application instance.
    """
    app = FastAPI(title="Meta-Evaluator Annotation Interface")

    store = SessionStore(
        eval_task=eval_task,
        eval_data=eval_data,
        annotations_dir=annotations_dir,
    )
    app.state.store = store

    app.include_router(task.router, prefix="/api")
    app.include_router(session.router, prefix="/api")
    app.include_router(samples.router, prefix="/api")
    app.include_router(annotations.router, prefix="/api")
    app.include_router(export.router, prefix="/api")

    if static_dir:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app
