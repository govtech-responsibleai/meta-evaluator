## ADDED Requirements

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
