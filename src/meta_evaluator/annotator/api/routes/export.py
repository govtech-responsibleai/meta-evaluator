"""Export routes."""

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from meta_evaluator.annotator.api.schemas import ExportRequest, ExportResponse

router = APIRouter()


@router.post("/export", response_model=ExportResponse)
def export_annotations(body: ExportRequest, request: Request) -> ExportResponse:
    """Export completed annotations.

    Args:
        body: Request with run_id.
        request: FastAPI request with app state.

    Returns:
        ExportResponse: Export result with file info.

    Raises:
        HTTPException: 404 if session not found.
    """
    store = request.app.state.store
    session = store._sessions.get(body.run_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return store.export(body.run_id, session.annotator_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/export/download/{filename}")
def download_file(filename: str, request: Request) -> FileResponse:
    """Download exported file.

    Args:
        filename: Name of file to download.
        request: FastAPI request with app state.

    Returns:
        FileResponse: File download response.

    Raises:
        HTTPException: 404 if file not found.
    """
    store = request.app.state.store
    filepath = os.path.join(store._annotations_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, filename=filename)
