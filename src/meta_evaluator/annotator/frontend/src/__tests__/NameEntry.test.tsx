import { NameEntry } from "@/components/NameEntry";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

describe("NameEntry", () => {
  it("renders input and submit button", () => {
    render(<NameEntry onSubmit={vi.fn()} loading={false} error={null} />);
    expect(screen.getByPlaceholderText("Enter your name")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /start annotating/i }),
    ).toBeInTheDocument();
  });

  it("disables button when name is empty", () => {
    render(<NameEntry onSubmit={vi.fn()} loading={false} error={null} />);
    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("calls onSubmit with trimmed name", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<NameEntry onSubmit={onSubmit} loading={false} error={null} />);
    await user.type(
      screen.getByPlaceholderText("Enter your name"),
      "  Alice  ",
    );
    await user.click(screen.getByRole("button"));
    expect(onSubmit).toHaveBeenCalledWith("Alice");
  });

  it("shows error message", () => {
    render(
      <NameEntry onSubmit={vi.fn()} loading={false} error="Name is required" />,
    );
    expect(screen.getByText("Name is required")).toBeInTheDocument();
  });

  it("shows loading state", () => {
    render(<NameEntry onSubmit={vi.fn()} loading={true} error={null} />);
    expect(screen.getByRole("button", { name: /starting/i })).toBeDisabled();
  });
});
