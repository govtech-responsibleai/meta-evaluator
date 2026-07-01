# Task-card keyboard focus design

## Context

`TaskPanel` currently routes number-key shortcuts to the first selectable task that is
not answered. That heuristic does not work for native multi-label tasks: their full
all-`FALSE` vector is valid immediately, so they are considered answered before the
annotator has reviewed them. As a result, the UI has no completion transition with
which to advance keyboard routing, and tasks after a multi-label task can become
unreachable by keyboard.

Answer validity and keyboard focus are separate concerns. An all-`FALSE` vector must
remain a valid multi-label answer, while keyboard routing should follow the task the
annotator is currently operating.

## Goals

- Give each task exactly one stop in the normal `Tab` order.
- Route number-key shortcuts to the task that currently has focus.
- Let `Tab` and `Shift+Tab` move forward and backward between tasks using native
  browser behavior.
- Preserve mouse operation, existing submission behavior, and the validity of an
  untouched all-`FALSE` multi-label vector.
- Prevent number keys from changing task answers when focus is outside the task area.

## Non-goals

- Adding a separate "done" or "reviewed" state.
- Changing backend schemas, multi-label vector encoding, or submission validation.
- Intercepting or reimplementing browser `Tab` behavior.
- Changing what the answered counter means.

## Interaction model

Each task contributes one focus target:

- A single-select or multi-label task uses its task card as the focus target.
- A free-form task uses its textarea as the card's single focus target, preserving
  normal text entry.
- Child radio buttons and checkboxes remain visible, accessible, and mouse-clickable,
  but are removed from the sequential tab order.

Focus determines the active task. While a single-select or multi-label task is active,
number keys select or toggle its corresponding options. `Tab` advances to the next
task, and `Shift+Tab` returns to the previous task. Numeric input in a free-form
textarea remains text and never triggers task shortcuts.

Clicking within a task activates that task for shortcuts. Leaving the task card clears
its active state. Consequently, number keys do nothing while focus is on instructions,
navigation, Save, or any other control outside a task.

Moving away from a multi-label task does not modify its vector or mark it complete.
The all-`FALSE` vector remains a valid value whether or not the task received focus.

## Component design

`TaskPanel` owns an `activeTaskName: string | null` state. Task focus-entry events set
the task name, and focus-exit events clear it when focus moves outside that task's
card. Navigating to another sample also clears the active task.

The number-key handler no longer derives a target from `isAnswered`. It resolves the
active task by name and applies the existing behavior:

- single-select: replace the selected value;
- multi-label: toggle the selected slot;
- free-form or no active task: do nothing.

The current active-border treatment and option-number hints are driven by
`activeTaskName`. Selectable task cards expose a focus indicator, an accessible group
label, and keyboard instructions. Focus entering a child control through a mouse click
still activates the containing task. Focus leaving the card for any external control
clears the task before further number-key input can be handled.

## Data flow

1. The browser moves focus to a task's single focus target.
2. The task updates `activeTaskName`.
3. A number-key event looks up the active task schema.
4. For a single-select task, the matching option replaces the scalar outcome.
5. For a multi-label task, the matching slot toggles between its outcome name and
   `"FALSE"`.
6. Native `Tab` movement transfers focus to the next task, which updates the active
   task name without changing either task's answer.
7. Submission uses the existing outcome normalization and required-task validation.

## Accessibility

The implementation preserves native forward and reverse tab navigation and does not
cancel `Tab`. Each selectable card is a keyboard-focusable composite group whose label
identifies the task and whose description explains the number-key shortcuts. Its child
checkboxes or radios remain in the accessibility tree and retain their checked state,
labels, and mouse behavior even though they are not separate sequential tab stops.

Free-form textareas remain native text controls and are the only focus target for their
cards. Focus styling must be visible on the active card, including when focus is within
its textarea or a child control.

## Error and reset behavior

No new application error state is required. If an active task disappears because the
sample or configuration changes, active state is cleared and number keys become a
no-op until another task receives focus. Unsupported number keys are also a no-op.

## Tests

Component tests will verify that:

- tabbing from a multi-label card activates the following single-select task;
- consecutive multi-label cards receive number-key input independently;
- `Shift+Tab` routes input back to the previous task;
- clicking a card or one of its child controls activates that task;
- number keys do nothing after focus leaves the task area;
- a free-form textarea is its card's single tab stop and accepts numeric text normally;
- child checkbox and radio controls remain mouse-clickable outside the tab sequence;
- navigating to another sample clears active keyboard state; and
- submission still accepts and emits untouched all-`FALSE` vectors.

Existing single-select, multi-label submission, and frontend build tests remain part of
the regression suite.
