import { Button } from "@/components/ui/button";
import type { OutcomeValue, Progress, Sample, TaskConfig } from "@/lib/api";
import { useCallback, useEffect, useRef, useState } from "react";
import { Navigation } from "./Navigation";
import { SampleDisplay } from "./SampleDisplay";
import { TaskPanel } from "./TaskPanel";

interface Props {
  taskConfig: TaskConfig;
  sample: Sample;
  progress: Progress | null;
  onSubmit: (outcomes: Record<string, OutcomeValue>) => void;
  onNavigate: (index: number) => void;
  onExport: () => void;
}

/** Default annotation rail width in px (26rem). */
const DEFAULT_RAIL_WIDTH = 416;
/** Narrowest the rail may be dragged — keeps long option labels readable. */
const MIN_RAIL_WIDTH = 320;
/** Widest the rail may be dragged, absolutely and relative to the viewport. */
const MAX_RAIL_WIDTH = 640;
/** Tailwind `md` breakpoint — below this the layout stacks vertically. */
const DESKTOP_MEDIA_QUERY = "(min-width: 768px)";

function clampRailWidth(width: number): number {
  const viewportMax =
    typeof window !== "undefined"
      ? Math.min(MAX_RAIL_WIDTH, window.innerWidth * 0.6)
      : MAX_RAIL_WIDTH;
  const upper = Math.max(MIN_RAIL_WIDTH, viewportMax);
  return Math.min(Math.max(width, MIN_RAIL_WIDTH), upper);
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

  const [railWidth, setRailWidth] = useState(DEFAULT_RAIL_WIDTH);
  // Whether the horizontal (desktop) layout is active. The inline rail width
  // must only apply here so the mobile stacked `w-full` layout is preserved.
  const [isDesktop, setIsDesktop] = useState(false);
  const cleanupDragRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia(DESKTOP_MEDIA_QUERY);
    const update = () => setIsDesktop(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  // Detach any active drag listeners on unmount (guards mid-drag unmount).
  useEffect(() => {
    return () => {
      cleanupDragRef.current?.();
    };
  }, []);

  const handleResizeStart = useCallback((event: React.MouseEvent) => {
    event.preventDefault();

    const handleMouseMove = (moveEvent: MouseEvent) => {
      setRailWidth(clampRailWidth(window.innerWidth - moveEvent.clientX));
    };

    const stopDrag = () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", stopDrag);
      document.body.style.userSelect = "";
      cleanupDragRef.current = null;
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", stopDrag);
    document.body.style.userSelect = "none";
    cleanupDragRef.current = stopDrag;
  }, []);

  return (
    <div className="min-h-screen md:h-screen flex flex-col bg-background md:overflow-hidden">
      <div className="border-b border-border/60 px-6 py-4 sticky top-0 z-20 bg-background/95 backdrop-blur-sm">
        <Navigation
          sample={sample}
          progress={progress}
          onPrevious={() => onNavigate(sample.index - 1)}
          onNext={() => onNavigate(sample.index + 1)}
        />
      </div>

      <div className="border-b border-border/60 bg-muted/40 px-6 py-2.5">
        <p className="text-center text-xs leading-relaxed text-muted-foreground">
          Hit <span className="font-medium text-foreground">Save &amp; Next</span> to save each
          answer and pick up right where you left off whenever you return.
        </p>
      </div>

      <div className="relative flex-1 flex min-h-0 flex-col md:flex-row md:overflow-hidden">
        <div className="relative md:flex-1 md:min-h-0 px-4 py-5 md:px-8 md:py-7 md:overflow-y-auto scroll-slim">
          <div className="max-w-4xl mx-auto">
            <SampleDisplay sample={sample} taskConfig={taskConfig} />
          </div>
        </div>

        <div
          role="separator"
          aria-orientation="vertical"
          onMouseDown={handleResizeStart}
          className="hidden md:block w-1 shrink-0 cursor-col-resize bg-[var(--annotation-rail-border)] hover:bg-primary/40 transition-colors"
        />

        <div
          className="relative w-full md:w-[26rem] md:shrink-0 px-4 py-5 md:px-5 md:py-7 md:overflow-y-auto scroll-slim bg-[var(--annotation-rail)] border-t md:border-t-0 border-[var(--annotation-rail-border)]"
          style={isDesktop ? { width: railWidth } : undefined}
        >
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
