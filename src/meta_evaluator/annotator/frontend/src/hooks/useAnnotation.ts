import { useCallback, useEffect, useState } from "react";
import {
  api,
  type ExportResult,
  type Progress,
  type Sample,
  type SessionInfo,
  type TaskConfig,
} from "@/lib/api";

export type AppState = "name_entry" | "annotating" | "exported";

export function useAnnotation() {
  const [appState, setAppState] = useState<AppState>("name_entry");
  const [taskConfig, setTaskConfig] = useState<TaskConfig | null>(null);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [currentSample, setCurrentSample] = useState<Sample | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getTask().then(setTaskConfig).catch((e: Error) => setError(e.message));
  }, []);

  const startSession = useCallback(async (name: string) => {
    setError(null);
    setLoading(true);
    try {
      const sess = await api.createSession(name);
      setSession(sess);
      const sample = await api.getSample(sess.run_id, 0);
      setCurrentSample(sample);
      const prog = await api.getProgress(sess.run_id);
      setProgress(prog);
      setAppState("annotating");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadSample = useCallback(
    async (index: number) => {
      if (!session) return;
      setLoading(true);
      try {
        const sample = await api.getSample(session.run_id, index);
        setCurrentSample(sample);
        setError(null);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    },
    [session],
  );

  const submitAnnotation = useCallback(
    async (outcomes: Record<string, string>) => {
      if (!session || !currentSample) return;
      setError(null);
      try {
        await api.submitAnnotation(
          session.run_id,
          currentSample.index,
          outcomes,
        );
        const prog = await api.getProgress(session.run_id);
        setProgress(prog);
        if (currentSample.index < currentSample.total - 1) {
          await loadSample(currentSample.index + 1);
        }
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    },
    [session, currentSample, loadSample],
  );

  const doExport = useCallback(async () => {
    if (!session) return;
    setError(null);
    setLoading(true);
    try {
      const result = await api.exportAnnotations(session.run_id);
      setExportResult(result);
      setAppState("exported");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [session]);

  return {
    appState,
    taskConfig,
    session,
    currentSample,
    progress,
    exportResult,
    error,
    loading,
    startSession,
    loadSample,
    submitAnnotation,
    doExport,
  };
}
