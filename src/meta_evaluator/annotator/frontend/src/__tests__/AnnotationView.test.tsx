import { AnnotationView } from "@/components/AnnotationView";
import type { Progress, Sample, TaskConfig } from "@/lib/api";
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const taskConfig: TaskConfig = {
  task_schemas: {
    sentiment: ["positive", "negative", "neutral"],
  },
  prompt_columns: ["text"],
  response_columns: ["response"],
  annotation_prompt: "Evaluate this:",
  required_tasks: ["sentiment"],
};

const sample: Sample = {
  index: 0,
  total: 3,
  sample_id: "s1",
  prompt_data: { text: "Hello" },
  response_data: { response: "Hi" },
  previous_annotation: null,
};

const progress: Progress = {
  run_id: "run_1",
  annotated_count: 0,
  total_samples: 3,
  incomplete_indices: [0, 1, 2],
};

/** Install a matchMedia mock that reports the given desktop state. */
function mockMatchMedia(matches: boolean) {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

function renderView() {
  return render(
    <AnnotationView
      taskConfig={taskConfig}
      sample={sample}
      progress={progress}
      onSubmit={vi.fn()}
      onNavigate={vi.fn()}
      onExport={vi.fn()}
    />,
  );
}

describe("AnnotationView layout", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders a vertical resize handle", () => {
    mockMatchMedia(true);
    renderView();
    const handle = screen.getByRole("separator");
    expect(handle).toHaveAttribute("aria-orientation", "vertical");
    expect(handle).toHaveClass("cursor-col-resize");
  });

  it("applies an inline rail width on the desktop layout", () => {
    mockMatchMedia(true);
    renderView();
    const rail = screen.getByRole("separator").nextElementSibling as HTMLElement;
    expect(rail.style.width).toBe("416px");
  });

  it("does not apply an inline rail width on the stacked layout", () => {
    mockMatchMedia(false);
    renderView();
    const rail = screen.getByRole("separator").nextElementSibling as HTMLElement;
    expect(rail.style.width).toBe("");
  });
});
