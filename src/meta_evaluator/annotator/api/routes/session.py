"""Session management routes."""

from fastapi import APIRouter, HTTPException, Request

from meta_evaluator.annotator.api.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
)

router = APIRouter()


@router.post("/session", response_model=CreateSessionResponse)
def create_session(
    body: CreateSessionRequest, request: Request
) -> CreateSessionResponse:
    """Create or resume annotation session.

    Args:
        body: Request containing annotator name.
        request: FastAPI request with app state.

    Returns:
        CreateSessionResponse: Session info.

    Raises:
        HTTPException: 422 if name is invalid.
    """
    store = request.app.state.store
    try:
        return store.create_session(body.annotator_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/session/{run_id}", response_model=CreateSessionResponse)
def get_session(run_id: str, request: Request) -> CreateSessionResponse:
    """Get existing session info.

    Args:
        run_id: Session run ID.
        request: FastAPI request with app state.

    Returns:
        CreateSessionResponse: Session info.

    Raises:
        HTTPException: 404 if session not found.
    """
    store = request.app.state.store
    result = store.get_session(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result
