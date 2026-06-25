import { Navigation } from "@/components/Navigation";
import type { Progress, Sample } from "@/lib/api";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

const sample: Sample = {
  index: 1,
  total: 5,
  sample_id: "s2",
  prompt_data: null,
  response_data: { response: "test" },
  previous_annotation: null,
};

const progress: Progress = {
  run_id: "run_1",
  annotated_count: 2,
  total_samples: 5,
  incomplete_indices: [0, 2, 4],
};

describe("Navigation", () => {
  it("shows progress count", () => {
    render(
      <Navigation
        sample={sample}
        progress={progress}
        onPrevious={vi.fn()}
        onNext={vi.fn()}
      />,
    );
    expect(screen.getByText("2 / 5 completed")).toBeInTheDocument();
  });

  it("disables previous on first sample", () => {
    const firstSample = { ...sample, index: 0 };
    render(
      <Navigation
        sample={firstSample}
        progress={progress}
        onPrevious={vi.fn()}
        onNext={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /previous/i })).toBeDisabled();
  });

  it("disables next on last sample", () => {
    const lastSample = { ...sample, index: 4 };
    render(
      <Navigation
        sample={lastSample}
        progress={progress}
        onPrevious={vi.fn()}
        onNext={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /next/i })).toBeDisabled();
  });

  it("calls onPrevious and onNext", async () => {
    const user = userEvent.setup();
    const onPrevious = vi.fn();
    const onNext = vi.fn();
    render(
      <Navigation
        sample={sample}
        progress={progress}
        onPrevious={onPrevious}
        onNext={onNext}
      />,
    );
    await user.click(screen.getByRole("button", { name: /previous/i }));
    expect(onPrevious).toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: /next/i }));
    expect(onNext).toHaveBeenCalled();
  });
});
