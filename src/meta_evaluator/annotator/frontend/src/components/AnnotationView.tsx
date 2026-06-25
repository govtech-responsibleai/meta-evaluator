import { Button } from "@/components/ui/button";
import type { Progress, Sample, TaskConfig } from "@/lib/api";
import { Navigation } from "./Navigation";
import { SampleDisplay } from "./SampleDisplay";
import { TaskPanel } from "./TaskPanel";

interface Props {
  taskConfig: TaskConfig;
  sample: Sample;
  progress: Progress | null;
  onSubmit: (outcomes: Record<string, string>) => void;
  onNavigate: (index: number) => void;
  onExport: () => void;
}

export function AnnotationView({
  taskConfig,
  sample,
  progress,
  onSubmit,
  onNavigate,
  onExport,
}: Props) {
  const canExport =
    progress && progress.annotated_count === progress.total_samples;

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <div className="border-b border-border/60 px-6 py-4 sticky top-0 z-20 bg-background/95 backdrop-blur-sm">
        <Navigation
          sample={sample}
          progress={progress}
          onPrevious={() => onNavigate(sample.index - 1)}
          onNext={() => onNavigate(sample.index + 1)}
        />
      </div>

      <div className="flex-1 flex min-h-0 flex-col md:flex-row overflow-hidden">
        <div className="order-2 md:order-1 flex-1 min-h-0 px-5 py-6 md:px-8 md:py-7 overflow-y-auto">
          <div className="max-w-4xl mx-auto">
            <SampleDisplay sample={sample} taskConfig={taskConfig} />
          </div>
        </div>

        <div className="order-1 md:order-2 w-full md:w-[26rem] shrink-0 max-h-[48vh] md:max-h-none px-5 py-6 md:py-7 overflow-y-auto bg-[var(--annotation-rail)] border-l border-[var(--annotation-rail-border)]">
          <TaskPanel
            taskConfig={taskConfig}
            sample={sample}
            onSubmit={onSubmit}
          />
          {canExport && (
            <Button onClick={onExport} className="mt-4 w-full">
              Export Results
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
