"""Sample data routes."""

from fastapi import APIRouter, HTTPException, Request

from meta_evaluator.annotator.api.schemas import SampleResponse

router = APIRouter()


@router.get("/samples/{index}", response_model=SampleResponse)
def get_sample(index: int, run_id: str, request: Request) -> SampleResponse:
    """Get sample data at index.

    Args:
        index: Sample index.
        run_id: Session run ID (query parameter).
        request: FastAPI request with app state.

    Returns:
        SampleResponse: Sample data.

    Raises:
        HTTPException: 404 if session or sample not found.
    """
    store = request.app.state.store
    try:
        return store.get_sample(run_id, index)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except IndexError:
        raise HTTPException(status_code=404, detail="Sample index out of bounds")
