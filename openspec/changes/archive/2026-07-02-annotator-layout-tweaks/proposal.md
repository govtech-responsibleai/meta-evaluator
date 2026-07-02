## Why

In the annotator UI, the prompt/response card is left-aligned in the left column, leaving a large empty gap on its right and making the layout feel unbalanced. Separately, the annotation rail is a fixed width, so annotators working with long option labels or wanting more room to read the sample cannot adjust the split. Both are small layout ergonomics issues that make daily annotation work less comfortable.

## What Changes

- Center the prompt/response card horizontally within the left column (currently pinned to the left edge).
- Make the annotation rail (right column) resizable by dragging the vertical divider between the two columns, with a visible `col-resize` handle.
- Clamp the rail width between a sensible minimum and maximum so neither column can collapse or dominate the viewport.
- Resize is desktop-only (the layout already stacks vertically on mobile); width resets to the default on each page load (no persistence).

## Capabilities

### New Capabilities
- `annotator-layout`: Horizontal layout behavior of the annotation view — card centering within the left column and drag-to-resize of the annotation rail, including min/max clamping and desktop-only applicability.

### Modified Capabilities
<!-- None: no existing specs define annotation view layout behavior. -->

## Impact

- **Code**: `src/meta_evaluator/annotator/frontend/src/components/AnnotationView.tsx` (layout markup, resize state, drag handlers, resize handle element).
- **Tests**: Frontend component tests under `src/meta_evaluator/annotator/frontend/src/__tests__/` may add coverage for the resize handle presence/behavior.
- **Build artifact**: The embedded frontend bundle (`dist/`) is regenerated and re-embedded for the Python package.
- **Dependencies**: None — resize is hand-rolled with React state and mouse listeners; no new npm packages.
- **No changes** to Python API, data schemas, or annotation persistence.
