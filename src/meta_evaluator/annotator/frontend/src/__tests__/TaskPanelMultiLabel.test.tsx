import { TaskPanel } from "@/components/TaskPanel";
import type { Sample, TaskConfig } from "@/lib/api";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

const taskConfig: TaskConfig = {
  task_schemas: {
    harm: { outcomes: ["hateful", "insults", "sexual"] },
    sentiment: ["positive", "negative"],
  },
  prompt_columns: ["text"],
  response_columns: ["response"],
  annotation_prompt: "Evaluate this:",
  required_tasks: ["harm", "sentiment"],
};

const sample: Sample = {
  index: 0,
  total: 3,
  sample_id: "s1",
  prompt_data: { text: "Hello" },
  response_data: { response: "Hi" },
  previous_annotation: null,
};

describe("TaskPanel multi-label", () => {
  it("renders a checkbox for each multi-label outcome", () => {
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={vi.fn()} />,
    );
    expect(
      screen.getByRole("checkbox", { name: "hateful" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", { name: "insults" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "sexual" })).toBeInTheDocument();
  });

  it("submits the full ordered vector with selected slots as names", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );
    await user.click(screen.getByRole("checkbox", { name: "hateful" }));
    await user.click(screen.getByRole("checkbox", { name: "sexual" }));
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["hateful", "FALSE", "sexual"],
      sentiment: "positive",
    });
  });

  it("submits the all-FALSE vector when nothing is checked", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["FALSE", "FALSE", "FALSE"],
      sentiment: "positive",
    });
  });

  it("toggles a slot off when a checked box is clicked again", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );
    const hateful = screen.getByRole("checkbox", { name: "hateful" });
    await user.click(hateful);
    await user.click(hateful);
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["FALSE", "FALSE", "FALSE"],
      sentiment: "positive",
    });
  });

  it("pre-fills checkboxes from a previous vector annotation", () => {
    const sampleWithPrev: Sample = {
      ...sample,
      previous_annotation: { harm: ["hateful", "FALSE", "sexual"] },
    };
    render(
      <TaskPanel
        taskConfig={taskConfig}
        sample={sampleWithPrev}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByRole("checkbox", { name: "hateful" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "insults" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "sexual" })).toBeChecked();
  });

  it("does not block submission on an untouched multi-label required task", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    // Only the multi-label task is required here.
    const config: TaskConfig = {
      ...taskConfig,
      task_schemas: { harm: { outcomes: ["hateful", "insults", "sexual"] } },
      required_tasks: ["harm"],
    };
    render(<TaskPanel taskConfig={config} sample={sample} onSubmit={onSubmit} />);
    await user.click(screen.getByRole("button", { name: /save & next/i }));
    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["FALSE", "FALSE", "FALSE"],
    });
  });

  it("focuses the first multi-label task and enables shortcuts immediately", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const config: TaskConfig = {
      ...taskConfig,
      task_schemas: { harm: { outcomes: ["hateful", "insults", "sexual"] } },
      required_tasks: ["harm"],
    };
    render(<TaskPanel taskConfig={config} sample={sample} onSubmit={onSubmit} />);

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.keyboard("2");
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["FALSE", "insults", "FALSE"],
    });
  });

  it("tabs from a multi-label card to the following single-select task", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.keyboard("2");
    await user.tab();
    expect(screen.getByRole("group", { name: "sentiment" })).toHaveFocus();
    await user.keyboard("1");
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["FALSE", "insults", "FALSE"],
      sentiment: "positive",
    });
  });

  it("routes number keys independently across consecutive multi-label cards", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const config: TaskConfig = {
      ...taskConfig,
      task_schemas: {
        harm: { outcomes: ["hateful", "insults"] },
        quality: { outcomes: ["vague", "incorrect"] },
      },
      required_tasks: ["harm", "quality"],
    };

    render(<TaskPanel taskConfig={config} sample={sample} onSubmit={onSubmit} />);

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.keyboard("1");
    await user.tab();
    expect(screen.getByRole("group", { name: "quality" })).toHaveFocus();
    await user.keyboard("2");
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["hateful", "FALSE"],
      quality: ["FALSE", "incorrect"],
    });
  });

  it("routes shortcuts back to the previous task after shift-tab", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("group", { name: "sentiment" })).toHaveFocus();
    await user.keyboard("1");
    await user.tab({ shift: true });
    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.keyboard("3");
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["FALSE", "FALSE", "sexual"],
      sentiment: "positive",
    });
  });

  it("activates a task when clicking one of its child controls", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );

    const hateful = screen.getByRole("checkbox", { name: "hateful" });
    await user.click(hateful);
    expect(hateful).toHaveFocus();
    await user.keyboard("3");
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["hateful", "FALSE", "sexual"],
      sentiment: "positive",
    });
  });

  it("keeps checkbox and radio controls mouse-clickable outside the tab sequence", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );

    expect(screen.getByRole("checkbox", { name: "hateful" })).toHaveAttribute(
      "tabindex",
      "-1",
    );
    expect(screen.getByRole("radio", { name: "positive" })).toHaveAttribute(
      "tabindex",
      "-1",
    );

    await user.click(screen.getByRole("checkbox", { name: "hateful" }));
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["hateful", "FALSE", "FALSE"],
      sentiment: "positive",
    });
  });

  it("does not route number keys after focus leaves the task area", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("group", { name: "sentiment" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: /save & next/i })).toHaveFocus();
    await user.keyboard("2");
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["FALSE", "FALSE", "FALSE"],
      sentiment: "positive",
    });
  });

  it("returns focus and shortcut routing to the first task when the sample changes", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const { rerender } = render(
      <TaskPanel taskConfig={taskConfig} sample={sample} onSubmit={onSubmit} />,
    );

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("group", { name: "sentiment" })).toHaveFocus();

    rerender(
      <TaskPanel
        taskConfig={taskConfig}
        sample={{ ...sample, index: 1, sample_id: "s2" }}
        onSubmit={onSubmit}
      />,
    );

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.keyboard("1");
    await user.click(screen.getByRole("radio", { name: "positive" }));
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["hateful", "FALSE", "FALSE"],
      sentiment: "positive",
    });
  });

  it("reseeds state and refocuses the first task when the task config changes", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const initialConfig: TaskConfig = {
      ...taskConfig,
      task_schemas: { harm: { outcomes: ["hateful", "insults", "sexual"] } },
      required_tasks: ["harm"],
    };
    const nextConfig: TaskConfig = {
      ...taskConfig,
      task_schemas: { harm: { outcomes: ["privacy"] } },
      required_tasks: ["harm"],
    };
    const { rerender } = render(
      <TaskPanel
        taskConfig={initialConfig}
        sample={sample}
        onSubmit={onSubmit}
      />,
    );

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    await user.keyboard("2");

    rerender(
      <TaskPanel taskConfig={nextConfig} sample={sample} onSubmit={onSubmit} />,
    );

    expect(screen.getByRole("group", { name: "harm" })).toHaveFocus();
    expect(screen.getByRole("checkbox", { name: "privacy" })).not.toBeChecked();
    await user.keyboard("1");
    await user.click(screen.getByRole("button", { name: /save & next/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      harm: ["privacy"],
    });
  });
});
