import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Sample, TaskConfig } from "@/lib/api";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

interface Props {
  sample: Sample;
  taskConfig: TaskConfig;
}

function CollapsibleField({
  label,
  value,
  defaultOpen = true,
}: {
  label: string;
  value: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="group flex w-full items-center gap-1.5 text-left py-1 transition-colors hover:text-foreground">
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground group-hover:text-foreground transition-colors" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground group-hover:text-foreground transition-colors" />
        )}
        <span className="text-[13px] font-semibold leading-5 text-foreground">
          {label}
        </span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-3 space-y-3 pl-5">
          {value.split(/\n\n+/).map((block, i) => (
            <p
              key={i}
              className="text-[15px] leading-7 text-foreground/90 whitespace-pre-line"
            >
              {block}
            </p>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function SampleDisplay({ sample, taskConfig }: Props) {
  return (
    <ScrollArea className="md:h-full">
      <div className="space-y-6">
        {sample.prompt_data && taskConfig.prompt_columns && (
          <div className="rounded-xl bg-card border border-border/50 px-4 py-5 md:p-6 shadow-sm">
            <div className="flex items-center gap-2 mb-4">
              <span className="size-2.5 rounded-sm bg-primary" />
              <h2 className="text-sm font-semibold uppercase tracking-[0.06em] text-muted-foreground">
                Prompt
              </h2>
              <span className="ml-auto text-xs text-muted-foreground/60">
                user
              </span>
            </div>
            <div className="space-y-5">
              {Object.entries(sample.prompt_data).map(([col, value]) => (
                <CollapsibleField key={col} label={col} value={value} />
              ))}
            </div>
          </div>
        )}

        <div className="rounded-xl bg-card border border-border/50 px-4 py-5 md:p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <span className="size-2.5 rounded-sm bg-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-[0.06em] text-muted-foreground">
              Response
            </h2>
            <span className="ml-auto text-xs text-muted-foreground/60">
              assistant
            </span>
          </div>
          <div className="space-y-5">
            {Object.entries(sample.response_data).map(([col, value]) => (
              <CollapsibleField key={col} label={col} value={value} />
            ))}
          </div>
        </div>
      </div>
    </ScrollArea>
  );
}
