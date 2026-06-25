import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { Sample, TaskConfig } from "@/lib/api";

interface Props {
  sample: Sample;
  taskConfig: TaskConfig;
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
                <div key={col} className="mb-3">
                  <p className="text-xs font-medium text-muted-foreground mb-1">
                    {col}
                  </p>
                  <p className="text-sm whitespace-pre-wrap">{value}</p>
                </div>
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
              <div key={col} className="mb-3">
                <p className="text-xs font-medium text-muted-foreground mb-1">
                  {col}
                </p>
                <p className="text-sm whitespace-pre-wrap">{value}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  );
}
