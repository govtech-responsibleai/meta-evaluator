# Claude Development Notes

## Linting and Code Quality

After every task, you must run both:
```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
```

Always fix all errors from the ruff check

## Testing

To run tests, always use:
```bash
uv run pytest
```

You can use regular pytest options after `uv run pytest`, for example:
```bash
uv run pytest tests/specific_test.py
uv run pytest -v
uv run pytest -k "test_name"
```

Note: After every task, run tests with skip integration:
```bash
uv run pytest --skip-integration
```

## Type Checking

To run type checking:
```bash
uv run pyright
```

Note: After every task, run type checking:
```bash
uv run pyright
```

## Task Completion Workflow

At the end of every task, ensure that you run ruff check and ruff format, and ensure your last command is always ruff format

After every task, run in order:
1. `uv run pyright`
2. `uv run pytest --skip-integration`

## Additional Reminders

- you have to fix all warnings from ruff check everytime