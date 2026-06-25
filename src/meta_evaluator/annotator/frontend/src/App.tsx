import { AnnotationView } from "@/components/AnnotationView";
import { ExportDialog } from "@/components/ExportDialog";
import { NameEntry } from "@/components/NameEntry";
import { useAnnotation } from "@/hooks/useAnnotation";

function App() {
  const {
    appState,
    taskConfig,
    currentSample,
    progress,
    exportResult,
    error,
    loading,
    startSession,
    loadSample,
    submitAnnotation,
    doExport,
  } = useAnnotation();

  if (appState === "name_entry") {
    return <NameEntry onSubmit={startSession} loading={loading} error={error} />;
  }

  if (appState === "exported" && exportResult) {
    return <ExportDialog result={exportResult} />;
  }

  if (appState === "annotating" && taskConfig && currentSample) {
    return (
      <AnnotationView
        taskConfig={taskConfig}
        sample={currentSample}
        progress={progress}
        onSubmit={submitAnnotation}
        onNavigate={loadSample}
        onExport={doExport}
      />
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center">
      <p className="text-muted-foreground">Loading...</p>
    </div>
  );
}

export default App;
