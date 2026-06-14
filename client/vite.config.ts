import adapter from "@sveltejs/adapter-auto";
import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
    plugins: [
        sveltekit({
            compilerOptions: {
                // Force runes mode for the project, except for libraries. Can be removed in svelte 6.
                runes: ({ filename }) =>
                    filename.split(/[/\\]/).includes("node_modules") ? undefined : true,
            },

            // adapter-auto only supports some environments, see https://svelte.dev/docs/kit/adapter-auto for a list.
            // If your environment is not supported, or you settled on a specific environment, switch out the adapter.
            // See https://svelte.dev/docs/kit/adapters for more information about adapters.
            adapter: adapter(),
        }),
        tailwindcss(),
    ],
    server: {
        // Proxy the API to the standalone harness server (npm run serve, port
        // 8787 by default) so the client calls `/api/*` same-origin: no CORS in
        // dev, and the frontend never hardcodes the backend's host. SSE works
        // through the proxy as a plain unbuffered HTTP pipe. Override the target
        // with VITE_API_TARGET if the server runs elsewhere.
        proxy: {
            "/api": {
                target: process.env.VITE_API_TARGET ?? "http://localhost:8787",
                changeOrigin: true,
            },
        },
    },
});
