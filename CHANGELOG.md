# Changelog

## 0.3.2

- Fix the annotator being unusable on mobile / narrow viewports. The annotation
  view was a fixed-height two-column layout (`h-screen` + `overflow-hidden`)
  whose columns fought over the viewport height when stacked, squeezing the
  prompt/response to an unreadable sliver behind the questions. Below the `md`
  breakpoint the view now flows as a single scrolling document (prompt/response,
  then questions), with the fixed-height and per-pane scroll behavior scoped to
  desktop. The navigation header stays sticky on mobile. Desktop layout —
  side-by-side panes, independent scroll, draggable rail resize — is unchanged.

## 0.2.2

- Fix blank annotator page when the bundle is embedded under a path prefix
  (e.g. served at `/annotate-ui/`). The Vite build now uses a relative `base`
  (`./`), so `index.html` references `./assets/...` instead of absolute
  `/assets/...`. Absolute paths resolved against the host root and 404'd (or
  collided with the embedding app's own `/assets`), leaving a white screen.
  Works for both standalone (served at `/`) and embedded deployments.

## 0.2.1

- Annotator frontend is now embeddable: `api.ts` reads optional `slug`/`token`
  URL query params (base `/api/annotate/{slug}`, bearer token). Fully backward
  compatible — no params means the previous `/api`, no-auth behavior.
- The published wheel now ships a prebuilt `annotator/frontend/dist/`, so no
  Node build is required by consumers (standalone or embedding).
- `meta_evaluator.annotator.frontend` is now an importable package.
- Optional (non-required) annotation tasks no longer block completion/scoring:
  a success row that omits an optional task in `task_schemas` is padded with a
  null column, so the run validates and can be scored. Previously any task in
  the schema effectively had to be answered or completion failed with a
  "missing required column" error.
