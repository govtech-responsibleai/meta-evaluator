## Why

On narrow (mobile) viewports the annotation view is unusable: the annotation rail (questions) and the prompt/response column fight over a fixed viewport height, so the questions dominate the screen and the prompt/response collapse to an unreadable sliver at the top. The layout was designed as a locked-height two-column desktop view and never degraded correctly when stacked vertically.

## What Changes

- On viewports below the `md` breakpoint, the annotation view SHALL flow as a single scrolling document (prompt/response, then questions, then the submit action) instead of two independently-scrolling panes locked to the viewport height.
- The outer frame's fixed viewport height and `overflow-hidden` clipping SHALL apply only at `md` and above; below `md` the page height grows with content.
- Neither stacked pane SHALL own an independent scroll region or a height that starves the other pane on mobile.
- The navigation header SHALL remain sticky at the top of the single mobile scroll.
- The existing desktop two-column layout — side-by-side panes, independent scroll regions, draggable resize handle — SHALL be unchanged.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `annotator-layout`: Add requirement that below the `md` breakpoint the view is a single natural-scroll document with a sticky header, and that the fixed-height/independent-scroll behavior is scoped to desktop only.

## Impact

- `src/meta_evaluator/annotator/frontend/src/components/AnnotationView.tsx` — outer frame height/overflow and the two pane wrappers become breakpoint-gated.
- `src/meta_evaluator/annotator/frontend/src/components/SampleDisplay.tsx` — the inner `ScrollArea h-full` height constraint is scoped to `md` so mobile content flows into the page scroll.
- No API, backend, or data-model changes. Desktop behavior (resize handle, `railWidth`/`isDesktop` logic) is untouched.
