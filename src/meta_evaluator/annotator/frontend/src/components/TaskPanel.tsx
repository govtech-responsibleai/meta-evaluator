import { Badge } from "@/components/ui/badge";
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
import { useEffect, useState } from "react";

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
  const [promptOpen, setPromptOpen] = useState(true);

  useEffect(() => {
    setOutcomes(sample.previous_annotation || {});
    setAttempted(false);
  }, [sample.index, sample.previous_annotation]);

  const handleSubmit = () => {
    setAttempted(true);
    const missing = taskConfig.required_tasks.filter(
      (t) => !outcomes[t]?.trim(),
    );
    if (missing.length > 0) return;
    onSubmit(outcomes);
  };

  const isFieldMissing = (task: string) =>
    attempted &&
    taskConfig.required_tasks.includes(task) &&
    !outcomes[task]?.trim();

  return (
    <div className="space-y-6">
      <Collapsible open={promptOpen} onOpenChange={setPromptOpen}>
        <CollapsibleTrigger className="flex items-center gap-1 w-full text-left">
          {promptOpen ? (
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
          )}
          <span className="text-xs font-medium text-muted-foreground">
            Instructions
          </span>
          {!promptOpen && (
            <span className="text-xs text-muted-foreground/60 ml-2 truncate">
              {truncate(taskConfig.annotation_prompt)}
            </span>
          )}
        </CollapsibleTrigger>
        <CollapsibleContent>
          <p className="text-sm text-muted-foreground mt-1 pl-4 whitespace-pre-wrap">
            {taskConfig.annotation_prompt}
          </p>
        </CollapsibleContent>
      </Collapsible>

      {Object.entries(taskConfig.task_schemas).map(([taskName, options]) => (
        <div key={taskName} className="space-y-2">
          <div className="flex items-center gap-2">
            <Label className="font-medium">
              {taskName}
              {taskConfig.required_tasks.includes(taskName) && (
                <span className="text-red-500 ml-0.5">*</span>
              )}
            </Label>
            {outcomes[taskName] && (
              <Badge variant="secondary" className="text-xs">
                done
              </Badge>
            )}
          </div>

          {options ? (
            <RadioGroup
              value={outcomes[taskName] || ""}
              onValueChange={(v) =>
                setOutcomes((prev) => ({ ...prev, [taskName]: v }))
              }
              className={
                isFieldMissing(taskName)
                  ? "border border-red-300 rounded p-2"
                  : ""
              }
            >
              {options.map((option) => (
                <div
                  key={option}
                  className="flex items-center space-x-2 min-h-[44px]"
                >
                  <RadioGroupItem
                    value={option}
                    id={`${taskName}-${option}`}
                  />
                  <Label
                    htmlFor={`${taskName}-${option}`}
                    className="cursor-pointer"
                  >
                    {option}
                  </Label>
                </div>
              ))}
            </RadioGroup>
          ) : (
            <Textarea
              value={outcomes[taskName] || ""}
              onChange={(e) =>
                setOutcomes((prev) => ({ ...prev, [taskName]: e.target.value }))
              }
              placeholder="Enter your response..."
              className={isFieldMissing(taskName) ? "border-red-300" : ""}
            />
          )}
        </div>
      ))}

      <div className="space-y-2">
        <Button onClick={handleSubmit} className="w-full">
          Submit &amp; Save
        </Button>
        <p className="text-xs text-muted-foreground text-center">
          You must click Submit to log your labels for this sample. Your
          progress is auto-saved after each submission.
        </p>
      </div>
    </div>
  );
}
