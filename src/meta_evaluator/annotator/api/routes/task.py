"""Task configuration route."""

from fastapi import APIRouter, Request

from meta_evaluator.annotator.api.schemas import TaskConfigResponse

router = APIRouter()


@router.get("/task", response_model=TaskConfigResponse)
def get_task_config(request: Request) -> TaskConfigResponse:
    """Return task configuration.

    Args:
        request: FastAPI request with app state.

    Returns:
        TaskConfigResponse: Task configuration.
    """
    store = request.app.state.store
    eval_task = store._eval_task
    return TaskConfigResponse(
        task_schemas=eval_task.task_schemas,
        prompt_columns=eval_task.prompt_columns,
        response_columns=eval_task.response_columns,
        annotation_prompt=eval_task.annotation_prompt,
        required_tasks=eval_task.get_required_tasks(),
    )
