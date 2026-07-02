# Claude Development Notes

## App Development Guide

This is a Python application to develop a MetaEvaluator. 
Given an evaluation task and dataset, the MetaEvaluator gathers results from LLM Judges and Human annotators, and calculates alignment metrics to measure the performance of different LLM-as-a-Judge.

## Logging 

For each main class, initialize a logger object with the main class name, and utilize the same logger through the class methods.

## Error Handling 

Always use custom exceptions. Define an exceptions file in each main functionality group, and define all custom exceptions within the file. 
Before implementing specific error messaging within each exception class, refer to the common utility (meta_evaluator.common.error_constants.py) and utilize existing error messages where applicable.

## Linting and Code Quality

After every task, you must run both:
```bash
uv tool run ruff check --preview --fix
uv tool run ruff format .
```

Always fix all errors from the ruff check. If there are unfixable errors, document them and ask for guidance.

## Type Checking

Streamlit is an optional dependency (`ui` extra). To ensure pyright can resolve all imports, sync with extras before type checking:
```bash
uv sync --extra all
```

After every task, run type checking:
```bash
uv run pyright
```

If there are type errors that cannot be resolved, document them and ask for guidance before proceeding.

## Testing

Always look for existing pytest fixtures in the corresponding conftest.py files before implementing new fixtures. 
All pytest fixtures should be defined in the corresponding conftest.py files with clear documentation. 

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

After every task, run tests excluding integration tests (integration tests require external services and are slower):
```bash
uv run pytest -m "not integration"
```

## Task Completion Workflow

At the end of every task, run in order:

1. **Linting and formatting:**
   ```bash
   uv tool run ruff check --preview --fix
   uv tool run ruff format .
   ```

2. **Type checking:**
   ```bash
   uv run pyright
   ```

3. **Testing:**
   ```bash
   uv run pytest -m "not integration"
   ```

Ensure your last command is always `uv tool run ruff format .`

## Packaging

- The package version is defined in `src/meta_evaluator/__init__.py` (`__version__`) and read by hatchling via `[tool.hatch.version]`.
- Optional dependency groups: `docs` (mkdocs), `ui` (streamlit), `all` (everything). Core deps include matplotlib.
- Publishing is automated via `.github/workflows/publish.yml` on GitHub release creation using trusted publishing (OIDC).

## Annotator Frontend

The annotator UI is a React + Vite + Tailwind app living in `src/meta_evaluator/annotator/frontend`. It is served to annotators as a pre-built bundle embedded in the Python package (the launcher serves `frontend/dist`).

All frontend commands are run from inside `src/meta_evaluator/annotator/frontend`:
```bash
npm run lint    # oxlint
npm run test    # vitest run
npm run build   # tsc -b && vite build — regenerates frontend/dist
npm run dev      # local dev server for manual verification
```

After any change to frontend source:
1. Run `npm run lint` and `npm run test`; fix issues in the files you touched.
2. Run `npm run build` to confirm the bundle compiles. **Do NOT commit `frontend/dist` — it is gitignored.** The bundle is rebuilt from source by CI (`ci.yml`) and the publish workflow (`publish.yml`) via `npm run build`, and hatchling includes it as a build artifact (`[tool.hatch.build.targets.*].artifacts`). So a UI change ships as source; the embedded bundle is regenerated at build/publish time.

## Additional Reminders (VERY IMPORTANT!)

- If you have any questions or doubts about the instructions, ask me before you begin. 
- Do NOT make any assumptions about what the user is asking. ALWAYS CLARIFY.