# annotator-layout Specification

## Purpose

Define the layout behavior of the annotation view, including how the prompt/response card is positioned, how the annotation rail can be resized, and how these behaviors respond to viewport size.
## Requirements
### Requirement: Prompt/response card centering

The annotation view SHALL horizontally center the prompt/response card within the left column, while retaining the existing maximum width constraint on the card.

#### Scenario: Card centered when left column is wider than the card

- **WHEN** the annotation view is rendered on a viewport where the left column is wider than the card's maximum width
- **THEN** the card SHALL be horizontally centered within the left column with equal left and right margins

#### Scenario: Card respects maximum width

- **WHEN** the left column is wider than the card's maximum width
- **THEN** the card SHALL NOT exceed its maximum width and SHALL remain centered rather than stretching to fill the column

### Requirement: Resizable annotation rail

The annotation view SHALL let the user resize the annotation rail (right column) by dragging the vertical divider between the left column and the rail. A visible resize handle SHALL be presented on the divider with a `col-resize` cursor affordance.

#### Scenario: Dragging the handle changes the rail width

- **WHEN** the user presses on the resize handle and drags horizontally
- **THEN** the annotation rail width SHALL update to follow the horizontal cursor position and the left column SHALL adjust to fill the remaining space

#### Scenario: Rail width is clamped within bounds

- **WHEN** the user drags the handle beyond the allowed range
- **THEN** the rail width SHALL be clamped to a minimum and maximum bound so that neither the rail nor the left column can collapse or dominate the viewport

#### Scenario: Width resets on reload

- **WHEN** the user reloads the page after resizing the rail
- **THEN** the rail width SHALL return to its default width (no persistence across reloads)

### Requirement: Desktop-only resize

The resize handle SHALL only be active on the horizontal (desktop) layout. On narrow viewports where the layout stacks vertically, the resize handle SHALL NOT be presented and the columns SHALL retain their stacked full-width behavior.

#### Scenario: Handle hidden on stacked layout

- **WHEN** the viewport is narrow enough that the left column and annotation rail stack vertically
- **THEN** the resize handle SHALL NOT be shown and dragging SHALL have no effect on layout

### Requirement: Single-scroll mobile layout

On viewports below the `md` breakpoint (the stacked layout), the annotation view SHALL flow as a single scrolling document containing, in order, the prompt/response column and then the annotation rail (questions and submit action). The page height SHALL grow with content rather than being locked to the viewport height, and neither stacked section SHALL own an independent scroll region or a fixed height that starves the other.

#### Scenario: Whole page scrolls as one document

- **WHEN** the view is rendered on a narrow viewport where the columns stack vertically and the combined content is taller than the viewport
- **THEN** the entire page SHALL scroll as a single document from the prompt/response through the questions to the submit action, using one scrollbar

#### Scenario: Prompt/response is fully readable on mobile

- **WHEN** the view is rendered on a narrow viewport
- **THEN** the prompt/response section SHALL render at its natural content height and SHALL NOT be collapsed to a sliver by the annotation rail

#### Scenario: No independent pane scrolling on mobile

- **WHEN** the columns are stacked on a narrow viewport
- **THEN** neither the prompt/response section nor the annotation rail SHALL present its own inner scroll region; scrolling SHALL move the whole page

### Requirement: Fixed-height layout scoped to desktop

The locked viewport height and content clipping of the annotation view SHALL apply only on the horizontal (desktop) layout at the `md` breakpoint and above. Below `md`, the outer frame SHALL NOT clip content or lock to the viewport height.

#### Scenario: Desktop retains locked-height independent scroll

- **WHEN** the view is rendered at the `md` breakpoint or wider
- **THEN** the layout SHALL remain a fixed-height two-column view with each column scrolling independently within the viewport

#### Scenario: Mobile does not clip content

- **WHEN** the view is rendered below the `md` breakpoint
- **THEN** the outer frame SHALL NOT apply `overflow-hidden` clipping or a fixed viewport height that hides content

### Requirement: Sticky navigation header on mobile

On the single-scroll mobile layout, the navigation header (sample counter and progress) SHALL remain sticky at the top of the viewport as the page scrolls.

#### Scenario: Header stays pinned while scrolling questions

- **WHEN** the annotator scrolls down into the questions on a narrow viewport
- **THEN** the navigation header SHALL remain visible pinned to the top of the viewport

