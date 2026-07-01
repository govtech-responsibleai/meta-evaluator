import { TaskPanel } from "@/components/TaskPanel";
import type { Sample, TaskConfig } from "@/lib/api";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

const taskConfig: TaskConfig = {
  task_schemas: {
    sentiment: ["positive", "negative", "neutral"],
    comments: null,
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

describe("TaskPanel", () => {
  it("renders radio options for classification tasks", () => {
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={vi.fn()} />,
    );
    expect(screen.getByRole("radio", { name: "positive" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "negative" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "neutral" })).toBeInTheDocument();
  });

  it("renders textarea for free-form tasks", () => {
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={vi.fn()} />,
    );
    expect(
      screen.getByPlaceholderText("Enter your response..."),
    ).toBeInTheDocument();
  });

  it("keeps long instructions collapsed by default", () => {
    const longPrompt =
      "Read the full rubric carefully before judging each response. Check factuality, tone, completeness, refusal behavior, and whether the response follows every constraint in the prompt.";
    render(
      <TaskPanel
        taskConfig={{
          ...taskConfig,
          annotation_prompt: longPrompt,
        }}
        sample={sample}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.queryByText(longPrompt)).not.toBeInTheDocument();
    expect(
      screen.getByText(/Read the full rubric carefully/),
    ).toBeInTheDocument();
  });

  it("does not submit when required fields empty", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );
    await user.click(screen.getByRole("button", { name: /save & next/i }));
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits when required fields filled", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));
    expect(onSubmit).toHaveBeenCalledWith({ sentiment: "positive" });
  });

  it("pre-fills from previous annotation", () => {
    const sampleWithPrev = {
      ...sample,
      previous_annotation: { sentiment: "negative" },
    };
    render(
      <TaskPanel
        taskConfig={taskConfig}
        sample={sampleWithPrev}
        onSubmit={vi.fn()}
      />,
    );
    const radio = screen.getByRole("radio", { name: "negative" });
    expect(radio).toBeChecked();
  });

  it("uses the single-select task card as one tab stop", async () => {
    const user = userEvent.setup();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={vi.fn()} />,
    );

    await user.tab();
    expect(screen.getByRole("button", { name: /instructions/i })).toHaveFocus();

    await user.tab();
    expect(screen.getByRole("group", { name: "sentiment" })).toHaveFocus();
    expect(screen.getByRole("radio", { name: "positive" })).toHaveAttribute(
      "tabindex",
      "-1",
    );
  });

  it("keeps numeric input in the free-form textarea as text", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );

    await user.tab();
    expect(screen.getByRole("button", { name: /instructions/i })).toHaveFocus();

    await user.tab();
    expect(screen.getByRole("group", { name: "sentiment" })).toHaveFocus();

    await user.tab();

    const commentsTextarea = screen.getByPlaceholderText(
      "Enter your response...",
    );
    expect(commentsTextarea).toHaveFocus();

    await user.keyboard("123");
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(commentsTextarea).toHaveValue("123");
    expect(onSubmit).toHaveBeenCalledWith({
      sentiment: "positive",
      comments: "123",
    });
  });
});
