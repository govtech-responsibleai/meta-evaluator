# Annotation UI Migration: Streamlit → React + FastAPI

## Summary

Replace the Streamlit-based annotation interface with a React (shadcn/ui) frontend served by a FastAPI backend. Goal: better UX, mobile support, reliable deployment, full customization.

## Constraints

- Keep `launch_annotator()` API signature unchanged
- Existing data models (EvalTask, EvalData, HumanAnnotationResults) stay untouched
- Single Docker image, single port
- No auth required (name-based identification, multiple concurrent annotators supported)
- Mobile-friendly

## Architecture

Monorepo approach: FastAPI + React in single package.

```
src/meta_evaluator/annotator/
├── api/
│   ├── __init__.py
│   ├── app.py              ← FastAPI app factory
│   ├── routes/
│   │   ├── session.py      ← POST /session, GET /session/:id
│   │   ├── annotations.py  ← GET /samples/:idx, POST /annotations, GET /progress
│   │   └── export.py       ← POST /export, GET /export/download/:filename
│   ├── state.py            ← In-memory session store (wraps HumanAnnotationResultsBuilder)
│   └── schemas.py          ← Pydantic request/response models
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── NameEntry.tsx
│   │   │   ├── AnnotationView.tsx
│   │   │   ├── SampleDisplay.tsx
│   │   │   ├── TaskPanel.tsx
│   │   │   ├── Navigation.tsx
│   │   │   └── ExportDialog.tsx
│   │   ├── hooks/
│   │   │   └── useAnnotation.ts
│   │   └── lib/
│   │       └── api.ts
│   └── components/ui/      ← shadcn/ui components
├── launcher/
│   ├── __init__.py
│   └── launcher.py         ← starts uvicorn, serves frontend/dist as static
├── exceptions.py
└── __init__.py
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/task | Task config (schemas, columns, annotation prompt) |
| POST | /api/session | Create session → returns run_id, total samples |
| GET | /api/session/{run_id} | Resume session → returns progress, current index |
| GET | /api/samples/{index} | Sample data (prompt + response columns) |
| POST | /api/annotations | Submit annotation → triggers auto-save |
| GET | /api/progress | Annotation count, total, incomplete indices |
| POST | /api/export | Finalize → saves parquet + metadata |
| GET | /api/export/download/{filename} | Download exported file |

## Data Flow

1. `launch_annotator()` → serializes EvalTask + EvalData → starts uvicorn
2. FastAPI loads task/data into memory, serves React SPA at `/`
3. User enters name → `POST /api/session` → creates `HumanAnnotationResultsBuilder`
4. User annotates samples → `POST /api/annotations` → auto-save to parquet
5. User exports → `POST /api/export` → finalizes `HumanAnnotationResults`

**State ownership:**
- Backend: persisted annotations, auto-save, export, session identity
- Frontend: current sample index, form field values (pre-submit), UI state

**Session resume:** `POST /api/session` with same name detects existing auto-save file and returns previous progress.

## UI Design

### Desktop (≥768px)

```
┌─────────────────────────────────────────────────────┐
│  Header: task name + progress bar (3/50)            │
├────────────────────────┬────────────────────────────┤
│  Sample Display        │  Annotation Panel (sticky) │
│  ─────────────────     │  ─────────────────         │
│  Prompt columns        │  Instruction text          │
│  (collapsible)         │                            │
│                        │  Task 1: ○ Good            │
│  Response columns      │          ○ Bad             │
│  (main content)        │          ○ Neutral         │
│                        │                            │
│                        │  Task 2: [text area]       │
│                        │                            │
│                        │  [Submit]                  │
├────────────────────────┴────────────────────────────┤
│  ← Previous                              Next →     │
└─────────────────────────────────────────────────────┘
```

### Mobile (<768px)

```
┌───────────────────────┐
│ Progress bar (3/50)   │
├───────────────────────┤
│ Sample Display        │
│ (scrollable)          │
├───────────────────────┤
│ Annotation Panel      │
│ ○ Good  ○ Bad         │
│ [text area]           │
├───────────────────────┤
│ [← Prev] [Submit] [Next →] ← sticky bottom │
└───────────────────────┘
```

### shadcn/ui Components Used

Card, RadioGroup, Textarea, Button, Progress, Input, Dialog, Badge, ScrollArea

### UX Improvements Over Streamlit

- No full-page rerun on interaction
- Keyboard shortcuts (Enter to submit, arrow keys for navigation)
- Smooth transitions between samples
- Inline validation (highlight missing fields, no modal alerts)
- Instant feedback on submit
- Touch-friendly (min 44px tap targets)
- Swipe gestures for prev/next (progressive enhancement)

## Error Handling

| Scenario | Backend | Frontend |
|----------|---------|----------|
| Invalid annotator name | 422 | Inline error below input |
| Missing required fields | 422 | Highlight unfilled fields, prevent submit |
| Auto-save write failure | 500, log, keep in memory | Toast "auto-save failed, data safe in memory" |
| Export failure | 500 with details | Dialog with retry button |
| Sample index OOB | 404 | Disable nav buttons at boundaries |
| Session not found | 404 | Redirect to name entry |

- Client-side validation first (reduce round-trips)
- Backend validates independently (never trust client)
- Existing `AnnotationError` exception hierarchy reused in API error handlers

## Deployment

### Docker

```dockerfile
FROM node:20-alpine AS frontend
WORKDIR /app/frontend
COPY src/meta_evaluator/annotator/frontend/ .
RUN npm ci && npm run build

FROM python:3.11-slim
WORKDIR /app
COPY --from=frontend /app/frontend/dist /app/static
COPY . .
RUN pip install -e ".[ui]"
EXPOSE 8000
CMD ["python", "-m", "meta_evaluator.annotator.launcher"]
```

### Package Distribution

- `frontend/dist/` is NOT committed to git (gitignored)
- Hatchling custom build hook runs `npm ci && npm run build` during `hatch build`
- CI/publish workflow already has Node available for this step
- `pip install meta-evaluator[ui]` works without Node at runtime (dist included in wheel)
- `ui` extra: swap `streamlit` for `fastapi`, `uvicorn`
- For local dev: run `npm run dev` in frontend/ for hot-reload against FastAPI backend

### ngrok

Same support as today — point at uvicorn port instead of Streamlit port.

## Testing Strategy

### Existing Tests to Remove

All Streamlit-coupled tests are deleted entirely (not incrementally refactored):

- `tests/annotator/interface/test_streamlit_app.py` (1,555 lines)
- `tests/annotator/interface/test_streamlit_session_manager.py` (602 lines)
- `tests/annotator/test_annotator_integration.py` (315 lines)

Keep unchanged:
- `tests/annotator/test_human_results.py` (666 lines) — tests data models, not UI
- `tests/annotator/conftest.py` — review and keep fixtures for EvalTask/EvalData setup; remove Streamlit-specific fixtures

### New Backend Tests (pytest + httpx AsyncClient)

**`tests/annotator/api/test_session.py`**
- Create session with valid name → 200, returns run_id + total samples
- Create session with empty name → 422
- Create session with existing auto-save → 200, returns previous progress
- Get session by run_id → 200 with progress
- Get session with invalid run_id → 404

**`tests/annotator/api/test_annotations.py`**
- Submit annotation with all required fields → 200, auto-save triggered
- Submit annotation missing required fields → 422, lists missing tasks
- Submit for invalid session → 404
- Submit for out-of-bounds sample index → 404
- Submit idempotent (re-submit same sample) → 200, overwrites previous

**`tests/annotator/api/test_samples.py`**
- Get sample at valid index → 200, returns prompt + response column data
- Get sample at negative index → 404
- Get sample beyond total → 404
- Sample data matches EvalData content

**`tests/annotator/api/test_export.py`**
- Export with all samples annotated → 200, files written to disk
- Export with incomplete samples → 200, includes warning in response
- Export creates valid parquet + metadata JSON
- Download endpoint returns correct file

**`tests/annotator/api/test_progress.py`**
- Progress after 0 annotations → count=0, total=N
- Progress after K annotations → count=K, correct incomplete indices

**`tests/annotator/api/test_task.py`**
- Returns task schemas, prompt columns, response columns, annotation prompt
- Matches EvalTask configuration

**`tests/annotator/test_state.py`**
- State manager wraps HumanAnnotationResultsBuilder correctly
- Auto-save writes parquet on each annotation
- Resume loads from auto-save file
- Multiple concurrent sessions (different annotator names) don't conflict

**`tests/annotator/test_launcher.py`**
- Launcher starts uvicorn on specified port
- Launcher detects port occupied → raises PortOccupiedError
- Launcher serializes EvalTask + EvalData to temp dir

### New Frontend Tests (Vitest + React Testing Library)

**`frontend/src/__tests__/NameEntry.test.tsx`**
- Renders name input and submit button
- Empty name → shows validation error, no API call
- Valid name → calls POST /api/session
- On success → navigates to annotation view

**`frontend/src/__tests__/AnnotationView.test.tsx`**
- Loads task config and first sample on mount
- Displays prompt columns (collapsible) and response columns
- Shows progress bar with correct count

**`frontend/src/__tests__/TaskPanel.test.tsx`**
- Renders radio group for multi-choice tasks
- Renders textarea for free-form tasks
- Required fields highlighted when submit attempted empty
- Submit disabled until all required fields filled (client-side)

**`frontend/src/__tests__/Navigation.test.tsx`**
- Previous disabled on first sample
- Next disabled on last sample
- Clicking next/prev fetches correct sample
- Keyboard arrows trigger navigation

**`frontend/src/__tests__/ExportDialog.test.tsx`**
- Shows completion summary (counts, timestamps)
- Download link points to correct endpoint
- Retry button visible on export failure

**`frontend/src/__tests__/useAnnotation.test.ts`**
- Hook manages current index, form state, submission
- On submit success: advances to next sample, clears form
- On submit error: preserves form state, shows error

### TDD Order (write tests first, then implement)

1. Backend: `test_task.py` → implement `GET /api/task`
2. Backend: `test_session.py` → implement session routes + state manager
3. Backend: `test_samples.py` → implement sample route
4. Backend: `test_annotations.py` → implement annotation submission + auto-save
5. Backend: `test_progress.py` → implement progress route
6. Backend: `test_export.py` → implement export route + download
7. Backend: `test_launcher.py` → implement launcher
8. Frontend: `NameEntry.test.tsx` → implement NameEntry component
9. Frontend: `TaskPanel.test.tsx` → implement TaskPanel
10. Frontend: `AnnotationView.test.tsx` → implement AnnotationView + SampleDisplay
11. Frontend: `Navigation.test.tsx` → implement Navigation
12. Frontend: `ExportDialog.test.tsx` → implement ExportDialog
13. Frontend: `useAnnotation.test.ts` → implement hook (wires everything together)
14. Integration: manual end-to-end run in Docker

## Files Changed Outside annotator/

| File | Change |
|------|--------|
| `src/meta_evaluator/meta_evaluator/base.py` | `launch_annotator()` internals → new launcher |
| `pyproject.toml` | `ui` extra deps: fastapi, uvicorn (remove streamlit) |
| `tests/annotator/` | Full test rewrite (API tests + React component tests) |
| `examples/` | Minor tweaks if launcher signature changes (goal: keep same) |

## Effort Estimate

| Layer | Effort |
|-------|--------|
| FastAPI backend | Low |
| React annotation UI | Medium |
| Session/auto-save | Medium |
| Launcher rewrite | Low |
| Docker packaging | Low |
| Test rewrite | Medium-High |

Total: ~2-3 weeks solo, ~1 week with incremental test writing.
