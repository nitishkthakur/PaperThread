import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The frontend talks to the backend over HTTP only — no local-only shortcuts a
    // hosted deployment could not support (REQUIREMENTS.md §11 D3).
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
