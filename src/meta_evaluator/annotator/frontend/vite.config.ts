import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  // Relative base so the built bundle works both standalone (served at "/")
  // and embedded under a path prefix (e.g. the platform serves it at
  // "/annotate-ui/"). Absolute "/assets/..." refs would otherwise resolve to
  // the host root and collide with / 404 against the embedding app.
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/__tests__/setup.ts",
  },
});
