import { Button } from "@/components/ui/button";
import type { ExportResult } from "@/lib/api";
import { api } from "@/lib/api";

interface Props {
  result: ExportResult;
}

export function ExportDialog({ result }: Props) {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        <div className="h-1 w-12 bg-primary rounded-full mb-6" />
        <h1 className="text-2xl font-semibold tracking-tight mb-1">
          Export Complete
        </h1>
        <p className="text-sm text-muted-foreground mb-6">
          Your annotations have been exported successfully.
        </p>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm mb-6">
          <span className="text-muted-foreground">Total samples:</span>
          <span className="font-medium tabular-nums">
            {result.total_count}
          </span>
          <span className="text-muted-foreground">Succeeded:</span>
          <span className="font-medium tabular-nums">
            {result.succeeded_count}
          </span>
          <span className="text-muted-foreground">Errors:</span>
          <span className="font-medium tabular-nums">
            {result.error_count}
          </span>
        </div>
        <div className="space-y-2">
          <a
            href={api.getDownloadUrl(result.data_file)}
            download
            className="block"
          >
            <Button variant="outline" className="w-full">
              Download Data (.parquet)
            </Button>
          </a>
          <a
            href={api.getDownloadUrl(result.metadata_file)}
            download
            className="block"
          >
            <Button variant="outline" className="w-full">
              Download Metadata (.json)
            </Button>
          </a>
        </div>
      </div>
    </div>
  );
}
