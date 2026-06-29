import { afterEach, describe, expect, it, vi } from "vitest";

// BASE is computed at module load from window.location.search, so each case
// sets the URL, resets the module registry, then dynamically imports api.ts.
async function loadApiWith(url: string) {
  window.history.pushState({}, "", url);
  vi.resetModules();
  return (await import("@/lib/api")).api;
}

afterEach(() => {
  vi.resetModules();
  vi.unstubAllGlobals();
  window.history.pushState({}, "", "/");
});

function mockFetchOnce() {
  const fetchMock = vi
    .fn()
    .mockResolvedValue({ ok: true, json: async () => ({}) });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("api base + token resolution", () => {
  it("no params -> base /api, no auth header", async () => {
    const fetchMock = mockFetchOnce();
    const api = await loadApiWith("/");
    await api.getTask();
    expect(fetchMock).toHaveBeenCalledWith("/api/task", expect.any(Object));
    const [, opts] = fetchMock.mock.calls[0];
    expect(opts.headers).toEqual({ "Content-Type": "application/json" });
  });

  it("slug param -> base /api/annotate/{slug}", async () => {
    const fetchMock = mockFetchOnce();
    const api = await loadApiWith("/?slug=abc");
    await api.getTask();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/annotate/abc/task",
      expect.any(Object),
    );
  });

  it("token param -> Authorization bearer header", async () => {
    const fetchMock = mockFetchOnce();
    const api = await loadApiWith("/?token=xyz");
    await api.getTask();
    const [, opts] = fetchMock.mock.calls[0];
    expect(opts.headers).toEqual({
      "Content-Type": "application/json",
      Authorization: "Bearer xyz",
    });
  });

  it("getDownloadUrl uses the resolved base", async () => {
    const api = await loadApiWith("/?slug=abc&token=xyz");
    expect(api.getDownloadUrl("f.json")).toBe(
      "/api/annotate/abc/export/download/f.json",
    );
  });
});
