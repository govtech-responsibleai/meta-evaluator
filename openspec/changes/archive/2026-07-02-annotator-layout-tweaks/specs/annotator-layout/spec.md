## ADDED Requirements

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
