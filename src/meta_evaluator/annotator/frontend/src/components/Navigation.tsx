import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { Progress as ProgressType, Sample } from "@/lib/api";
import { ArrowLeft, ArrowRight } from "lucide-react";

interface Props {
  sample: Sample;
  progress: ProgressType | null;
  onPrevious: () => void;
  onNext: () => void;
}

export function Navigation({ sample, progress, onPrevious, onNext }: Props) {
  const percent = progress
    ? (progress.annotated_count / progress.total_samples) * 100
    : 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center">
        <div className="flex-1" />
        <div className="flex items-center gap-3">
          <Button
            variant="default"
            size="icon"
            onClick={onPrevious}
            disabled={sample.index === 0}
          >
            <ArrowLeft className="size-4" />
          </Button>
          <span className="text-sm text-foreground/60 tabular-nums">
            Sample {sample.index + 1} of {sample.total}
          </span>
          <Button
            variant="default"
            size="icon"
            onClick={onNext}
            disabled={sample.index >= sample.total - 1}
          >
            <ArrowRight className="size-4" />
          </Button>
        </div>
        <div className="flex-1 flex justify-end">
          <div className="flex items-center gap-2 text-[13px] text-green-800/60">
            <span className="size-2.5 rounded-full bg-green-800/50 ring-[3px] ring-green-800/10" />
            <span>Autosaved</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <Progress value={percent} className="flex-1" />
        <span className="text-sm font-medium text-foreground/70 whitespace-nowrap tabular-nums">
          {progress?.annotated_count ?? 0} / {sample.total} completed
        </span>
      </div>
    </div>
  );
}
