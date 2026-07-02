## Context

`AnnotationView.tsx` frames the annotation view as a locked-height, two-column desktop layout: `h-screen flex flex-col overflow-hidden`, containing a left prompt/response pane (`flex-1 min-h-0 overflow-y-auto`) and a right question rail (`shrink-0 overflow-y-auto`) separated by a draggable resize handle. Each pane owns an independent scroll region so the two columns scroll separately within a fixed viewport.

This model breaks when the panes stack (`flex-col`) below the `md` breakpoint. With the container locked to `h-screen overflow-hidden`, the `shrink-0` rail keeps its full (tall) content height and wins the height fight, while the `flex-1 min-h-0` prompt/response pane collapses to a sliver. The page cannot scroll as a whole because of `overflow-hidden`, so the annotator cannot reach or read the prompt and response.

The desktop-only resize logic (`railWidth`, `isDesktop`, the `hidden md:block` handle) is already correctly gated and is not in scope to change.

## Goals / Non-Goals

**Goals:**
- Below `md`, the view flows as a single natural-scroll document: prompt/response, then questions, then submit — one page scrollbar, top to bottom.
- The navigation header stays sticky at the top of the mobile scroll.
- Desktop (`md` and up) layout is byte-for-byte unchanged: side-by-side panes, independent scroll, draggable resize.

**Non-Goals:**
- No change to the desktop two-column behavior or the resize handle.
- No tabs, segmented toggles, or column-level collapse (options C/D were considered and rejected).
- No change to field-level collapsibles, task logic, keyboard handling, or any backend/API.

## Decisions

**Decision: Breakpoint-gate the frame rather than restructure the DOM.**
Keep a single component tree and scope the desktop-only CSS (fixed height, overflow clipping, per-pane scroll, pane sizing) behind `md:` utilities. Below `md` those rules are absent, so the browser's default document flow takes over and the stacked panes scroll as one.
- *Rationale*: The DOM already stacks correctly (`flex-col md:flex-row`); the only thing forcing the mobile breakage is desktop-oriented height/overflow rules applied at all widths. Removing them at mobile is subtractive and low-risk.
- *Alternatives considered*: (A) Separate mobile/desktop component trees — more code, duplicated task rendering, higher divergence risk. (B) Column-level collapse / tab toggle — hides context while answering; user explicitly chose single-scroll (option A).

**Decision: `min-h-screen` on mobile, `md:h-screen md:overflow-hidden` on desktop.**
The outer frame grows with content on mobile and locks to the viewport on desktop.
- *Rationale*: `min-h-screen` still fills the viewport when content is short, but allows growth when content is tall.

**Decision: Scope pane scroll/height to `md`.**
The left pane's `flex-1 min-h-0 overflow-y-auto` and the rail's `overflow-y-auto` become `md:`-prefixed; `SampleDisplay`'s inner `ScrollArea h-full` becomes `md:h-full` so it does not impose a fixed height on mobile.
- *Rationale*: Independent per-pane scroll is a two-column concept. On a single mobile scroll it produces nested/competing scroll regions and re-introduces the squish.

**Decision: Keep the header sticky on mobile.**
The existing `sticky top-0 z-20` stays as-is at all breakpoints (confirmed with user).
- *Rationale*: Keeps the sample counter and progress visible while scrolling into the questions.

## Risks / Trade-offs

- **Nested scroll leftovers** → If any inner `h-full`/`overflow` constraint is missed, the mobile squish partially returns. Mitigation: audit `SampleDisplay`'s `ScrollArea` and both pane wrappers; verify by scrolling a long form on a narrow viewport.
- **Sticky header eats vertical space on small screens** → Accepted per user decision; header is compact.
- **Desktop regression risk** → Every desktop rule must remain gated so nothing changes at `md`+. Mitigation: only add `md:` prefixes / mobile fallbacks; do not alter existing `md:` values. Manually verify desktop resize still works.
