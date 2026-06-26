"""Annotation submission and progress routes."""

from fastapi import APIRouter, HTTPException, Request

from meta_evaluator.annotator.api.schemas import (
    ProgressResponse,
    SubmitAnnotationRequest,
    SubmitAnnotationResponse,
)

router = APIRouter()


@router.post("/annotations", response_model=SubmitAnnotationResponse)
def submit_annotation(
    body: SubmitAnnotationRequest, request: Request
) -> SubmitAnnotationResponse:
    """Submit annotation for a sample.

    Args:
        body: Request with run_id, sample_index, outcomes.
        request: FastAPI request with app state.

    Returns:
        SubmitAnnotationResponse: Submission result.

    Raises:
        HTTPException: 404 if session or sample not found.
    """
    store = request.app.state.store
    try:
        return store.submit_annotation(body.run_id, body.sample_index, body.outcomes)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except IndexError:
        raise HTTPException(status_code=404, detail="Sample index out of bounds")


@router.get("/progress", response_model=ProgressResponse)
def get_progress(run_id: str, request: Request) -> ProgressResponse:
    """Get annotation progress.

    Args:
        run_id: Session run ID (query parameter).
        request: FastAPI request with app state.

    Returns:
        ProgressResponse: Current progress.

    Raises:
        HTTPException: 404 if session not found.
    """
    store = request.app.state.store
    try:
        return store.get_progress(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
