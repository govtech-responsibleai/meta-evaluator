const params = new URLSearchParams(window.location.search);
const slug = params.get("slug");
const token = params.get("token");

const BASE = slug ? `/api/annotate/${slug}` : "/api";
const authHeaders: Record<string, string> = token
  ? { Authorization: `Bearer ${token}` }
  : {};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...authHeaders },
    ...options,
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(detail.detail || resp.statusText);
  }
  return resp.json();
}

/**
 * A multi-label (pick-several) task schema. Serialized by the backend as
 * `{ "outcomes": [...] }`, distinct from a bare `string[]` (single-select) and
 * from `null` (free-form).
 */
export interface MultiLabelSchema {
  outcomes: string[];
}

/** A single task's schema: single-select list, multi-label wrapper, or free-form. */
export type TaskSchema = string[] | MultiLabelSchema | null;

/** An outcome value: a single label (single-select/free-form) or an ordered vector (multi-label). */
export type OutcomeValue = string | string[];

export interface TaskConfig {
  task_schemas: Record<string, TaskSchema>;
  prompt_columns: string[] | null;
  response_columns: string[];
  annotation_prompt: string;
  required_tasks: string[];
}

export interface SessionInfo {
  run_id: string;
  annotator_id: string;
  total_samples: number;
  resumed: boolean;
  annotated_count: number;
}

export interface Sample {
  index: number;
  total: number;
  sample_id: string;
  prompt_data: Record<string, string> | null;
  response_data: Record<string, string>;
  previous_annotation: Record<string, OutcomeValue> | null;
}

export interface SubmitResult {
  success: boolean;
  annotated_count: number;
  auto_saved: boolean;
}

export interface Progress {
  run_id: string;
  annotated_count: number;
  total_samples: number;
  incomplete_indices: number[];
}

export interface ExportResult {
  metadata_file: string;
  data_file: string;
  total_count: number;
  succeeded_count: number;
  error_count: number;
}

export const api = {
  getTask: () => request<TaskConfig>("/task"),

  createSession: (annotator_name: string) =>
    request<SessionInfo>("/session", {
      method: "POST",
      body: JSON.stringify({ annotator_name }),
    }),

  getSession: (run_id: string) => request<SessionInfo>(`/session/${run_id}`),

  getSample: (run_id: string, index: number) =>
    request<Sample>(`/samples/${index}?run_id=${run_id}`),

  submitAnnotation: (
    run_id: string,
    sample_index: number,
    outcomes: Record<string, OutcomeValue>,
  ) =>
    request<SubmitResult>("/annotations", {
      method: "POST",
      body: JSON.stringify({ run_id, sample_index, outcomes }),
    }),

  getProgress: (run_id: string) =>
    request<Progress>(`/progress?run_id=${run_id}`),

  exportAnnotations: (run_id: string) =>
    request<ExportResult>("/export", {
      method: "POST",
      body: JSON.stringify({ run_id }),
    }),

  getDownloadUrl: (filename: string) => `${BASE}/export/download/${filename}`,
};
