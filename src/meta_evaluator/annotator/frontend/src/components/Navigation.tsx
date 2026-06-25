import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { Progress as ProgressType, Sample } from "@/lib/api";

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
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <Progress value={percent} className="flex-1" />
        <span className="text-sm text-muted-foreground whitespace-nowrap">
          {progress?.annotated_count ?? 0} / {sample.total}
        </span>
      </div>
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={onPrevious}
          disabled={sample.index === 0}
        >
          Previous
        </Button>
        <span className="text-sm text-muted-foreground self-center">
          Sample {sample.index + 1} of {sample.total}
        </span>
        <Button
          variant="outline"
          onClick={onNext}
          disabled={sample.index >= sample.total - 1}
        >
          Next
        </Button>
      </div>
    </div>
  );
}
