import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Textarea } from "@/components/ui/textarea";
import type { Sample, TaskConfig } from "@/lib/api";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

interface Props {
  taskConfig: TaskConfig;
  sample: Sample;
  onSubmit: (outcomes: Record<string, string>) => void;
}

function truncate(text: string, maxLen = 60): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
}

export function TaskPanel({ taskConfig, sample, onSubmit }: Props) {
  const [outcomes, setOutcomes] = useState<Record<string, string>>({});
  const [attempted, setAttempted] = useState(false);
  const [promptOpen, setPromptOpen] = useState(false);

  const taskEntries = Object.entries(taskConfig.task_schemas);
  const radioTasks = taskEntries.filter(([, options]) => options !== null);

  useEffect(() => {
    setOutcomes(sample.previous_annotation || {});
    setAttempted(false);
  }, [sample.index, sample.previous_annotation]);

  const handleSubmit = useCallback(() => {
    setAttempted(true);
    const missing = taskConfig.required_tasks.filter(
      (t) => !outcomes[t]?.trim(),
    );
    if (missing.length > 0) return;
    onSubmit(outcomes);
  }, [taskConfig.required_tasks, outcomes, onSubmit]);

  const isFieldMissing = (task: string) =>
    attempted &&
    taskConfig.required_tasks.includes(task) &&
    !outcomes[task]?.trim();

  const answeredCount = taskEntries.filter(
    ([name]) => outcomes[name]?.trim(),
  ).length;
  const totalTasks = taskEntries.length;

  const activeTaskIndex = radioTasks.findIndex(
    ([name]) => !outcomes[name]?.trim(),
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
      if (num >= 1 && num <= 9 && activeTaskIndex >= 0) {
        const [taskName, options] = radioTasks[activeTaskIndex];
        if (options && num <= options.length) {
          e.preventDefault();
          setOutcomes((prev) => ({ ...prev, [taskName]: options[num - 1] }));
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeTaskIndex, radioTasks, handleSubmit]);

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

      {taskEntries.map(([taskName, options], taskIdx) => {
        const isActive =
          radioTasks[activeTaskIndex]?.[0] === taskName;

        return (
          <div
            key={taskName}
            className={`rounded-lg bg-card border px-4 py-3.5 transition-colors ${isActive ? "border-primary/40 shadow-sm" : "border-border/50"}`}
          >
            <div className="flex items-center gap-2 mb-2.5">
              <span className="text-xs font-medium text-muted-foreground/60 tabular-nums">
                {String(taskIdx + 1).padStart(2, "0")}
              </span>
              <Label className="text-[13px] font-semibold leading-5 text-foreground flex-1">
                {taskName}
                {taskConfig.required_tasks.includes(taskName) && (
                  <span className="text-red-500 ml-0.5">*</span>
                )}
              </Label>
            </div>

            {options ? (
              <>
                <RadioGroup
                  value={outcomes[taskName] || ""}
                  onValueChange={(v) =>
                    setOutcomes((prev) => ({ ...prev, [taskName]: v }))
                  }
                  className={
                    isFieldMissing(taskName)
                      ? "rounded-md border border-destructive/50 bg-destructive/5 p-2"
                      : ""
                  }
                >
                  {options.map((option, optIndex) => (
                    <div
                      key={option}
                      className={`flex min-h-[34px] items-center space-x-2.5 rounded-md px-2.5 py-1 transition-colors ${outcomes[taskName] === option ? "bg-primary/8" : "hover:bg-accent/40"}`}
                    >
                      <RadioGroupItem
                        value={option}
                        id={`${taskName}-${option}`}
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
                {isActive && (
                  <p className="mt-2 text-[11px] text-muted-foreground/60 text-right">
                    Press key to select
                  </p>
                )}
              </>
            ) : (
              <Textarea
                value={outcomes[taskName] || ""}
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
