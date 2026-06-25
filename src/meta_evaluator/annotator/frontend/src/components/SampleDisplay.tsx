import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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

function truncate(text: string, maxLen = 80): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
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
    <Collapsible open={open} onOpenChange={setOpen} className="mb-3">
      <CollapsibleTrigger className="flex items-center gap-1 w-full text-left group">
        {open ? (
          <ChevronDown className="h-3 w-3 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 text-muted-foreground" />
        )}
        <span className="text-xs font-medium text-muted-foreground">
          {label}
        </span>
        {!open && (
          <span className="text-xs text-muted-foreground/60 ml-2 truncate">
            {truncate(value)}
          </span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <p className="text-sm whitespace-pre-wrap mt-1 pl-4">{value}</p>
      </CollapsibleContent>
    </Collapsible>
  );
}

export function SampleDisplay({ sample, taskConfig }: Props) {
  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 pr-4">
        {sample.prompt_data && taskConfig.prompt_columns && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Prompt
              </CardTitle>
            </CardHeader>
            <CardContent>
              {Object.entries(sample.prompt_data).map(([col, value]) => (
                <CollapsibleField key={col} label={col} value={value} />
              ))}
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Response
            </CardTitle>
          </CardHeader>
          <CardContent>
            {Object.entries(sample.response_data).map(([col, value]) => (
              <CollapsibleField key={col} label={col} value={value} />
            ))}
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
}
