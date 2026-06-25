import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useState } from "react";

interface Props {
  onSubmit: (name: string) => void;
  loading: boolean;
  error: string | null;
}

export function NameEntry({ onSubmit, loading, error }: Props) {
  const [name, setName] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) onSubmit(name.trim());
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Annotation Session</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <Input
                placeholder="Enter your name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoFocus
              />
              <p className="text-xs text-muted-foreground mt-2">
                Use the same name you have been using. Case sensitive.
              </p>
              {error && <p className="text-sm text-red-500 mt-1">{error}</p>}
            </div>
            <Button
              type="submit"
              className="w-full"
              disabled={!name.trim() || loading}
            >
              {loading ? "Starting..." : "Start Annotating"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
