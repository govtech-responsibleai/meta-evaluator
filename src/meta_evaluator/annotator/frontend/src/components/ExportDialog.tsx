import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ExportResult } from "@/lib/api";
import { api } from "@/lib/api";

interface Props {
  result: ExportResult;
}

export function ExportDialog({ result }: Props) {
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-lg">
        <CardHeader>
          <CardTitle>Export Complete</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <span className="text-muted-foreground">Total samples:</span>
            <span>{result.total_count}</span>
            <span className="text-muted-foreground">Succeeded:</span>
            <span>{result.succeeded_count}</span>
            <span className="text-muted-foreground">Errors:</span>
            <span>{result.error_count}</span>
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
        </CardContent>
      </Card>
    </div>
  );
}
