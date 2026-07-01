import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Textarea } from "@/components/ui/textarea";
import type {
  MultiLabelSchema,
  OutcomeValue,
  Sample,
  TaskConfig,
  TaskSchema,
} from "@/lib/api";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { FocusEvent } from "react";
import { useCallback, useEffect, useState } from "react";

interface Props {
  taskConfig: TaskConfig;
  sample: Sample;
  onSubmit: (outcomes: Record<string, OutcomeValue>) => void;
}

/** The not-selected sentinel for a multi-label slot (mirrors the backend). */
const FALSE = "FALSE";

function truncate(text: string, maxLen = 60): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
}

/** A multi-label schema is the `{ outcomes: [...] }` object form. */
function isMultiLabel(schema: TaskSchema): schema is MultiLabelSchema {
  return (
    schema !== null && !Array.isArray(schema) && Array.isArray(schema.outcomes)
  );
}

/** A single-select schema is a bare `string[]`. */
function isSingleSelect(schema: TaskSchema): schema is string[] {
  return Array.isArray(schema);
}

/** Build the all-FALSE vector for a multi-label task of the given length. */
function emptyVector(length: number): string[] {
  return Array.from({ length }, () => FALSE);
}

/**
 * Normalise a stored multi-label outcome into a full ordered vector.
 * A previous annotation is already a full vector; anything else (absent /
 * wrong length) resets to the all-FALSE vector so slot alignment holds.
 */
function toVector(value: OutcomeValue | undefined, outcomes: string[]): string[] {
  if (Array.isArray(value) && value.length === outcomes.length) {
    return [...value];
  }
  return emptyVector(outcomes.length);
}

/** Whether slot i of a multi-label vector is selected (holds the outcome name). */
function isSlotSelected(vector: string[], outcomes: string[], i: number): boolean {
  return vector[i] === outcomes[i];
}

export function TaskPanel({ taskConfig, sample, onSubmit }: Props) {
  const [outcomes, setOutcomes] = useState<Record<string, OutcomeValue>>({});
  const [attempted, setAttempted] = useState(false);
  const [promptOpen, setPromptOpen] = useState(false);
  const [activeTaskName, setActiveTaskName] = useState<string | null>(null);

  const taskEntries = Object.entries(taskConfig.task_schemas);

  useEffect(() => {
    // Seed state from any previous annotation, normalising multi-label tasks to
    // a full ordered vector so checkbox state and slot toggles stay aligned.
    const seeded: Record<string, OutcomeValue> = {};
    for (const [name, schema] of taskEntries) {
      const prev = sample.previous_annotation?.[name];
      if (isMultiLabel(schema)) {
        seeded[name] = toVector(prev, schema.outcomes);
      } else if (prev !== undefined) {
        seeded[name] = prev;
      }
    }
    setOutcomes(seeded);
    setAttempted(false);
    setActiveTaskName(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sample.index, sample.previous_annotation]);

  // A task is "answered" if: single-select has a value, free-form has text, or
  // multi-label has been rendered (its vector is always full-length once seeded).
  const isAnswered = useCallback(
    (name: string, schema: TaskSchema): boolean => {
      const value = outcomes[name];
      if (isMultiLabel(schema)) {
        return Array.isArray(value) && value.length === schema.outcomes.length;
      }
      return typeof value === "string" && value.trim().length > 0;
    },
    [outcomes],
  );

  const handleSubmit = useCallback(() => {
    setAttempted(true);
    // Ensure every multi-label task submits a full vector even if untouched
    // ("nothing applies" = all-FALSE), and validate required non-multi-label tasks.
    const finalOutcomes: Record<string, OutcomeValue> = { ...outcomes };
    for (const [name, schema] of taskEntries) {
      if (isMultiLabel(schema)) {
        finalOutcomes[name] = toVector(outcomes[name], schema.outcomes);
      }
    }

    const missing = taskConfig.required_tasks.filter((t) => {
      const schema = taskConfig.task_schemas[t];
      // Multi-label is always fully defined (all-FALSE is valid); only
      // single-select / free-form can be "missing".
      if (isMultiLabel(schema)) return false;
      const value = finalOutcomes[t];
      return typeof value !== "string" || !value.trim();
    });
    if (missing.length > 0) return;
    onSubmit(finalOutcomes);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskConfig.required_tasks, taskConfig.task_schemas, outcomes, onSubmit]);

  const isFieldMissing = (task: string) => {
    if (!attempted || !taskConfig.required_tasks.includes(task)) return false;
    const schema = taskConfig.task_schemas[task];
    if (isMultiLabel(schema)) return false;
    const value = outcomes[task];
    return typeof value !== "string" || !value.trim();
  };

  const answeredCount = taskEntries.filter(([name, schema]) =>
    isAnswered(name, schema),
  ).length;
  const totalTasks = taskEntries.length;

  const toggleSlot = useCallback(
    (taskName: string, outcomeList: string[], slot: number) => {
      setOutcomes((prev) => {
        const vector = toVector(prev[taskName], outcomeList);
        vector[slot] =
          vector[slot] === outcomeList[slot] ? FALSE : outcomeList[slot];
        return { ...prev, [taskName]: vector };
      });
    },
    [],
  );

  const handleTaskBlur = useCallback(
    (taskName: string, e: FocusEvent<HTMLDivElement>) => {
      if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
        setActiveTaskName((current) =>
          current === taskName ? null : current,
        );
      }
    },
    [],
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        handleSubmit();
        return;
      }

      if (e.target instanceof HTMLTextAreaElement) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const num = parseInt(e.key, 10);
      if (num < 1 || num > 9 || activeTaskName === null) return;

      const schema = taskConfig.task_schemas[activeTaskName];
      if (isSingleSelect(schema) && num <= schema.length) {
        // Single-select: number replaces the current selection.
        e.preventDefault();
        setOutcomes((prev) => ({
          ...prev,
          [activeTaskName]: schema[num - 1],
        }));
      } else if (isMultiLabel(schema) && num <= schema.outcomes.length) {
        // Multi-label: number toggles that slot's membership.
        e.preventDefault();
        toggleSlot(activeTaskName, schema.outcomes, num - 1);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeTaskName, taskConfig.task_schemas, handleSubmit, toggleSlot]);

  return (
    <div className="space-y-6">
      <Collapsible
        open={promptOpen}
        onOpenChange={setPromptOpen}
        className="rounded-lg border border-[var(--annotation-rail-border)] bg-background/45 px-3 py-2"
      >
        <CollapsibleTrigger className="flex items-center gap-1.5 w-full text-left rounded-md py-1 transition-colors hover:text-foreground">
          {promptOpen ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
          )}
          <span className="text-xs font-medium text-muted-foreground">
            Instructions
          </span>
          {!promptOpen && (
            <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground/70">
              {truncate(taskConfig.annotation_prompt, 96)}
            </span>
          )}
        </CollapsibleTrigger>
        <CollapsibleContent>
          <p className="mt-2 max-h-40 overflow-y-auto border-t border-[var(--annotation-rail-border)] pt-2 text-sm leading-relaxed text-muted-foreground whitespace-pre-wrap md:max-h-56">
            {taskConfig.annotation_prompt}
          </p>
        </CollapsibleContent>
      </Collapsible>

      <p className="text-xs text-muted-foreground">
        {answeredCount}/{totalTasks} answered
      </p>

      {taskEntries.map(([taskName, schema], taskIdx) => {
        const isSelectable = isSingleSelect(schema) || isMultiLabel(schema);
        const isActive = activeTaskName === taskName;
        const labelId = `task-${taskIdx}-label`;
        const instructionsId = `task-${taskIdx}-instructions`;

        return (
          <div
            key={taskName}
            role={isSelectable ? "group" : undefined}
            aria-labelledby={isSelectable ? labelId : undefined}
            aria-describedby={isSelectable ? instructionsId : undefined}
            tabIndex={isSelectable ? 0 : undefined}
            onFocus={isSelectable ? () => setActiveTaskName(taskName) : undefined}
            onBlur={isSelectable ? (e) => handleTaskBlur(taskName, e) : undefined}
            className={`rounded-lg bg-card border px-4 py-3.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 ${isActive ? "border-primary/40 shadow-sm" : "border-border/50"}`}
          >
            <div className="flex items-center gap-2 mb-2.5">
              <span className="text-xs font-medium text-muted-foreground/60 tabular-nums">
                {String(taskIdx + 1).padStart(2, "0")}
              </span>
              <Label
                id={labelId}
                className="text-[13px] font-semibold leading-5 text-foreground flex-1"
              >
                {taskName}
                {taskConfig.required_tasks.includes(taskName) && (
                  <span aria-hidden="true" className="text-red-500 ml-0.5">
                    *
                  </span>
                )}
                {isMultiLabel(schema) && (
                  <span
                    aria-hidden="true"
                    className="ml-1.5 text-[10px] font-normal text-muted-foreground/60"
                  >
                    (select all that apply)
                  </span>
                )}
              </Label>
            </div>

            {isMultiLabel(schema) ? (
              <div className="grid w-full gap-2">
                {schema.outcomes.map((option, optIndex) => {
                  const vector = toVector(outcomes[taskName], schema.outcomes);
                  const checked = isSlotSelected(
                    vector,
                    schema.outcomes,
                    optIndex,
                  );
                  return (
                    <div
                      key={option}
                      className={`flex min-h-[34px] items-center space-x-2.5 rounded-md px-2.5 py-1 transition-colors ${checked ? "bg-primary/8" : "hover:bg-accent/40"}`}
                    >
                      <Checkbox
                        checked={checked}
                        onCheckedChange={() =>
                          toggleSlot(taskName, schema.outcomes, optIndex)
                        }
                        id={`${taskName}-${option}`}
                        tabIndex={-1}
                      />
                      <Label
                        htmlFor={`${taskName}-${option}`}
                        className="flex-1 cursor-pointer text-sm font-normal leading-5 text-foreground/85"
                      >
                        {option}
                      </Label>
                      {isActive && (
                        <kbd className="inline-flex size-5 items-center justify-center rounded border border-border/60 bg-muted/50 text-[10px] font-medium text-muted-foreground/70 tabular-nums">
                          {optIndex + 1}
                        </kbd>
                      )}
                    </div>
                  );
                })}
                <p
                  id={instructionsId}
                  className={`mt-2 text-[11px] text-muted-foreground/60 text-right ${isActive ? "" : "sr-only"}`}
                >
                  Press number keys to toggle options
                </p>
              </div>
            ) : isSingleSelect(schema) ? (
              <>
                <RadioGroup
                  value={
                    typeof outcomes[taskName] === "string"
                      ? (outcomes[taskName] as string)
                      : ""
                  }
                  onValueChange={(v) =>
                    setOutcomes((prev) => ({ ...prev, [taskName]: v }))
                  }
                  className={
                    isFieldMissing(taskName)
                      ? "rounded-md border border-destructive/50 bg-destructive/5 p-2"
                      : ""
                  }
                >
                  {schema.map((option, optIndex) => (
                    <div
                      key={option}
                      className={`flex min-h-[34px] items-center space-x-2.5 rounded-md px-2.5 py-1 transition-colors ${outcomes[taskName] === option ? "bg-primary/8" : "hover:bg-accent/40"}`}
                    >
                      <RadioGroupItem
                        value={option}
                        id={`${taskName}-${option}`}
                        tabIndex={-1}
                      />
                      <Label
                        htmlFor={`${taskName}-${option}`}
                        className="flex-1 cursor-pointer text-sm font-normal leading-5 text-foreground/85"
                      >
                        {option}
                      </Label>
                      {isActive && (
                        <kbd className="inline-flex size-5 items-center justify-center rounded border border-border/60 bg-muted/50 text-[10px] font-medium text-muted-foreground/70 tabular-nums">
                          {optIndex + 1}
                        </kbd>
                      )}
                    </div>
                  ))}
                </RadioGroup>
                <p
                  id={instructionsId}
                  className={`mt-2 text-[11px] text-muted-foreground/60 text-right ${isActive ? "" : "sr-only"}`}
                >
                  Press number keys to select an option
                </p>
              </>
            ) : (
              <Textarea
                value={
                  typeof outcomes[taskName] === "string"
                    ? (outcomes[taskName] as string)
                    : ""
                }
                onChange={(e) =>
                  setOutcomes((prev) => ({
                    ...prev,
                    [taskName]: e.target.value,
                  }))
                }
                placeholder="Enter your response..."
                className={
                  isFieldMissing(taskName)
                    ? "border-destructive/60"
                    : "bg-background/50 border-border/40"
                }
              />
            )}
          </div>
        );
      })}

      <div className="sticky bottom-0 bg-[var(--annotation-rail)] pt-4 pb-2 -mx-5 px-5 border-t border-[var(--annotation-rail-border)]">
        <Button onClick={handleSubmit} className="w-full h-10 text-sm font-semibold">
          Save & Next
        </Button>
      </div>
    </div>
  );
}
