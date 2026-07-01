# Initial task autofocus design

## Context

The task-card keyboard model routes shortcuts according to browser focus, but the
annotation page initially leaves focus outside the tasks. This makes the first task
look inactive and hides its number-key affordances until the annotator presses `Tab`
past the Instructions control. The keyboard feature is therefore not discoverable
when a sample opens.

## Goals

- Make the first task visibly active and ready for keyboard input as soon as a sample
  opens.
- Apply the same behavior after every sample transition, including Save & Next.
- Focus the task's real input target so browser focus, active styling, and shortcut
  routing cannot disagree.
- Explain native `Tab` navigation on every task.

## Non-goals

- Reordering or removing the Instructions control from the document tab order.
- Intercepting `Tab` or implementing custom focus traversal.
- Changing answer state, submission, validation, or multi-label encoding.

## Interaction design

When `TaskPanel` first renders a sample, it moves browser focus to the first task's
single focus target. A single-select or multi-label task focuses its task card. A
free-form task focuses its textarea. The same reset occurs whenever the sample
changes, so a new sample always starts at its first task.

Native focus events remain the only source of active-task state. Programmatic focus
therefore produces the same active border, option-number hints, and number-key routing
as mouse or keyboard focus. Instructions remain reachable by pressing `Shift+Tab`
from the first task.

Each task includes a compact navigation hint. Every task except the last displays
`Press [Tab] to go to next task`, where `[Tab]` is rendered as a keyboard-key badge.
The final task displays `Press [Tab] to continue`, because its next focus target is
not another task. These hints are visible rather than conditional on active state so
the navigation model is discoverable on every card.

## Component design

`TaskPanel` keeps a reference to the first task's actual focus target. After sample
state is seeded and the task list is rendered, an effect focuses that target. The
first-task reference points to the card for selectable tasks and directly to the
textarea for free-form tasks.

Task cards handle focus entry and exit for every task type. Selectable cards remain
focusable themselves. Free-form cards do not gain an additional tab stop; focus on
their textarea bubbles to the card handler and activates the task.

The existing global number-key handler remains unchanged in principle: it acts only
on the focused active selectable task, and textarea input remains exempt.

## Edge cases

- With no tasks, no autofocus is attempted.
- If the task configuration changes with the sample, focus targets are resolved from
  the newly rendered task list rather than retained from the previous sample.
- If the first task is free-form, autofocus may place the caret in the textarea; this
  is intentional and makes it ready for immediate typing.
- Autofocus does not select, toggle, clear, or otherwise mutate an answer.

## Accessibility

The implementation uses real DOM focus and preserves visible focus indication. It
does not synthesize a separate active state or override native `Tab`/`Shift+Tab`
behavior. The hint uses semantic `<kbd>` presentation and readable text, while task
labels continue to provide each focus target's accessible name.

## Tests

Component tests will verify that:

- the first selectable task card receives focus when the page opens;
- the first free-form textarea receives focus when the page opens;
- changing to a new sample returns focus to its first task;
- initial focus immediately enables the first selectable task's number shortcuts;
- `Tab` advances from the focused first task to the next task using native order;
- non-final and final tasks render the correct navigation hints; and
- autofocus does not change seeded or empty outcomes.

