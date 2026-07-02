## Context

The annotation view (`AnnotationView.tsx`) is a two-column flex layout inside a full-height flex-column shell:

```
h-screen flex flex-col
├── Navigation bar (sticky)
├── "Save & Next" hint bar
└── flex-1 flex md:flex-row          ← the two-column row
    ├── LEFT: flex-1, overflow-y-auto
    │     └── div.max-w-4xl          ← card wrapper (SampleDisplay)
    └── RIGHT: md:w-[26rem] shrink-0  ← annotation rail (TaskPanel)
          bg var(--annotation-rail), md:border-l
```

Two ergonomics issues:
1. The card wrapper caps width at `max-w-4xl` but has no horizontal auto-margin, so it hugs the left edge and leaves dead space on its right.
2. The rail is a hardcoded `md:w-[26rem]`, so annotators cannot widen it for long option labels or narrow it to read more of the sample.

The layout stacks vertically below the `md` breakpoint (`flex-col md:flex-row`), where a horizontal drag handle makes no sense.

## Goals / Non-Goals

**Goals:**
- Center the capped-width card within the left column.
- Allow drag-to-resize of the rail via the vertical divider, with a visible `col-resize` handle.
- Clamp rail width to a sane min/max so neither column collapses.
- Keep resize desktop-only; keep the mobile stacked layout unchanged.

**Non-Goals:**
- Persisting the chosen width across reloads (explicitly out of scope; width resets to default).
- Introducing a resizable-panels library or any new dependency.
- Centering the card relative to the whole viewport (centering is within the left column only).
- Making the left column independently resizable — it always fills remaining space.

## Decisions

### Decision 1: Center the card with `mx-auto`

Add `mx-auto` to the `max-w-4xl` wrapper on line 47. `max-w-4xl` already caps the width; `mx-auto` supplies the equal left/right margins that center it within the `flex-1` column.

- **Why not `justify-center` on the column?** The column is `overflow-y-auto` and also positions other absolutely/relatively placed content; `mx-auto` on the block is the least invasive and the idiomatic Tailwind approach already used elsewhere in the codebase.

### Decision 2: Hand-rolled drag resize with React state

Replace the static `md:w-[26rem]` with a state-driven width applied via inline style on the rail, plus a thin drag handle on the divider.

- Introduce `const [railWidth, setRailWidth] = useState(DEFAULT_RAIL_WIDTH)` where `DEFAULT_RAIL_WIDTH = 416` (26rem in px).
- Apply the width with an inline `style={{ width: railWidth }}` on the rail (guarded to desktop — see Decision 4), keeping `shrink-0` so the left column yields space.
- Add a handle element positioned on the shared border (`cursor-col-resize`, a few px wide, full height). On `mouseDown`, attach `mousemove`/`mouseup` listeners to `window`; on each `mousemove`, compute the new width from the viewport right edge minus `event.clientX` and call `setRailWidth(clamp(...))`; on `mouseup`, detach.
- Use a `useRef`/`useEffect` cleanup so listeners are always removed (including on unmount mid-drag) and add `user-select: none` on the body during drag to prevent text selection.

- **Why hand-rolled over `react-resizable-panels`?** This is a single split; the library would add a dependency and force both columns into its `Panel`/`PanelResizeHandle` wrappers. Hand-rolled is ~30–50 lines, matches the existing bespoke component style, and keeps the bundle lean.

### Decision 3: Clamp width to min/max

Clamp `railWidth` between `MIN_RAIL_WIDTH` (e.g. 320px — enough for the widest option labels) and `MAX_RAIL_WIDTH` (e.g. `min(640, window.innerWidth * 0.6)` — leaves the left column usable). Exact constants are an implementation detail; the requirement is only that both columns stay usable.

### Decision 4: Desktop-only via breakpoint guard

The layout is `flex-col md:flex-row`. The inline width and the handle must only apply on the `md:` (horizontal) layout so the mobile stacked layout keeps full-width columns.

- Render the handle only in the horizontal layout (e.g. an `md:block hidden` handle), and apply the inline width in a way that does not override the mobile full-width (`w-full`) behavior — e.g. gate the inline style on a matchMedia/`md` check, or scope it so mobile `w-full` wins. Tailwind's responsive classes cannot read a JS number, so the inline style is applied only when the desktop layout is active.

## Risks / Trade-offs

- **Inline width vs. Tailwind responsive classes** → Inline styles always win over classes, which could break the mobile `w-full` layout if applied unconditionally. Mitigation: only apply the inline width when the desktop (`md`) layout is active (matchMedia or equivalent guard), verified against the stacked mobile view.
- **Drag listeners leaking / text selection during drag** → Mitigation: attach listeners on `mouseDown`, always detach on `mouseUp` and on unmount via `useEffect` cleanup; toggle `user-select: none` during drag.
- **No persistence surprises the user** → Accepted trade-off per scope decision; width resets to 26rem each load. Revisit later if annotators ask for it (would be a small `localStorage` follow-up).
- **Fast drags outrunning `mousemove`** → Using `window`-level listeners (not the handle element) keeps tracking even when the cursor leaves the thin handle, avoiding stalls.
