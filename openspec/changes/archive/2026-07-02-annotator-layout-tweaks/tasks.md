## 1. Center the prompt/response card

- [x] 1.1 In `AnnotationView.tsx`, add `mx-auto` to the `max-w-4xl` card wrapper (the div around `SampleDisplay`)
- [x] 1.2 Verify in the dev server that the card centers within the left column at desktop width and still respects its max width

## 2. Resizable rail state and handlers

- [x] 2.1 In `AnnotationView.tsx`, add width constants (`DEFAULT_RAIL_WIDTH = 416`, `MIN_RAIL_WIDTH`, `MAX_RAIL_WIDTH`) and `const [railWidth, setRailWidth] = useState(DEFAULT_RAIL_WIDTH)`
- [x] 2.2 Add a `clamp` helper (or inline clamp) that bounds the new width between min and max (max may be relative to `window.innerWidth`)
- [x] 2.3 Implement drag handlers: `onMouseDown` on the handle attaches `mousemove`/`mouseup` listeners on `window`; `mousemove` computes width from `window.innerWidth - event.clientX` and calls `setRailWidth(clamp(...))`; `mouseup` detaches
- [x] 2.4 Toggle `user-select: none` on the document body during an active drag and restore it on `mouseup`
- [x] 2.5 Add a `useEffect` cleanup that removes any attached listeners on unmount (guard against mid-drag unmount)

## 3. Resize handle and desktop-only layout wiring

- [x] 3.1 Add a thin vertical resize handle element on the divider between the left column and the rail, with `cursor-col-resize` and a subtle visible affordance; render it only on the horizontal layout (`hidden md:block` or equivalent)
- [x] 3.2 Apply `railWidth` to the rail via inline `style={{ width }}` only when the desktop (`md`) layout is active, so the mobile `w-full` stacked layout is not overridden (use a matchMedia/`md` guard)
- [x] 3.3 Confirm the rail keeps `shrink-0` and the left column keeps `flex-1` so the left column absorbs the width change

## 4. Verification

- [x] 4.1 Manually verify in dev: drag widens/narrows the rail, clamps at min and max, text does not get selected during drag, and fast drags off the handle still track
- [x] 4.2 Verify mobile/stacked layout (below `md`) shows no handle and columns remain full-width
- [x] 4.3 Verify width resets to default after a page reload (no persistence)
- [x] 4.4 Add/adjust a component test under `src/__tests__/` covering the handle's presence on desktop layout (and absence on stacked layout if feasible)
- [x] 4.5 Run frontend lint (`npm run lint`) and tests (`npm run test`) in the frontend dir; fix any issues
- [x] 4.6 Rebuild the embedded bundle (`npm run build`) so `frontend/dist` reflects the changes
