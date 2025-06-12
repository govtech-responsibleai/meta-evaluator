# Claude Development Notes

## Linting and Code Quality

After every task, you must run both:
```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
```

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

## Type Checking

To run type checking:
```bash
uv run pyright
```