import { Button } from "@/components/ui/button";
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
      <div className="w-full max-w-sm">
        <div className="h-1 w-12 bg-primary rounded-full mb-6" />
        <h1 className="text-2xl font-semibold tracking-tight mb-1">
          Annotation Session
        </h1>
        <p className="text-sm text-muted-foreground mb-6">
          Enter your name to begin reviewing samples.
        </p>
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
      </div>
    </div>
  );
}
