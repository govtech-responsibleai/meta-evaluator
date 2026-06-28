# Changelog

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
