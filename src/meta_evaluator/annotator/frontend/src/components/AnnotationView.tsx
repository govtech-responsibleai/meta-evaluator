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
    <div className="min-h-screen flex flex-col">
      <div className="border-b p-4 sticky top-0 z-10 bg-background">
        <Navigation
          sample={sample}
          progress={progress}
          onPrevious={() => onNavigate(sample.index - 1)}
          onNext={() => onNavigate(sample.index + 1)}
        />
      </div>

      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        <div className="flex-1 p-4 overflow-y-auto md:border-r">
          <SampleDisplay sample={sample} taskConfig={taskConfig} />
        </div>

        <div className="w-full md:w-96 p-4 overflow-y-auto">
          <TaskPanel
            taskConfig={taskConfig}
            sample={sample}
            onSubmit={onSubmit}
          />
          {canExport && (
            <button
              onClick={onExport}
              className="mt-4 w-full bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700"
            >
              Export Results
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
