## 1. Frame and pane layout (AnnotationView.tsx)

- [x] 1.1 Change the outer frame from `h-screen … overflow-hidden` to `min-h-screen md:h-screen … md:overflow-hidden` so the page grows with content on mobile and stays locked on desktop
- [x] 1.2 Scope the pane container's `overflow-hidden` to `md:overflow-hidden` (keep `flex-col md:flex-row`)
- [x] 1.3 Gate the left prompt/response pane's height/scroll rules to desktop: `flex-1 min-h-0 … overflow-y-auto` become `md:flex-1 md:min-h-0 md:overflow-y-auto`
- [x] 1.4 Gate the annotation rail's `overflow-y-auto` to `md:overflow-y-auto`, keeping `w-full md:w-[26rem]` and `md:shrink-0` sizing
- [x] 1.5 Confirm the navigation header keeps `sticky top-0 z-20` at all breakpoints (no change needed, verify)
- [x] 1.6 Confirm the desktop resize handle (`hidden md:block`) and `railWidth`/`isDesktop` logic are untouched

## 2. Inner scroll region (SampleDisplay.tsx)

- [x] 2.1 Scope the `ScrollArea` height constraint to desktop (`h-full` → `md:h-full`) so mobile content flows into the page scroll instead of an inner scroll region

## 3. Verification

- [x] 3.1 Manually verify on a narrow viewport: whole page scrolls as one document, prompt/response fully readable, no independent pane scroll, header stays pinned — confirmed by user
- [x] 3.2 Manually verify desktop (`md`+) is unchanged: two-column layout, independent scroll, draggable resize still works — confirmed by user
- [x] 3.3 Run `npm run lint` and `npm run test` in `src/meta_evaluator/annotator/frontend`; fix issues in touched files (lint clean; 2 pre-existing Navigation.test failures unrelated to this change, confirmed present on clean tree)
- [x] 3.4 Run `npm run build` to confirm the bundle compiles
