# construct-harness task runner.
#
# The stack is two processes: the API server (port 8787) and the SvelteKit
# client (vite, port 5173, which proxies /api to the server). `just dev` runs
# both together; that's the one to use day to day, since the client alone now
# gets ECONNREFUSED on /api without the server up.
#
# Runtime split: the server runs on NODE, not bun, because the stores use
# node:sqlite, which bun does not implement (Bun 1.3.x). The package's npm
# scripts already invoke node with --env-file-if-exists, so `serve`/`test`/
# `check`/`fmt` go through them. The client is just vite, so it runs under bun.

# Default `just` with no args lists the recipes.
default:
    @just --list

# Run the whole stack: API server (node, :8787) + client (vite, :5173) together; one Ctrl-C stops both.
dev:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "starting api server (:8787, node) + client (:5173, vite) — ctrl-c stops both"
    npm run --silent serve &
    server=$!
    trap 'kill "$server" 2>/dev/null || true' EXIT INT TERM
    just client
    wait "$server"

# The API server alone, on node (loads .env). Override the port with PORT=9000 just serve.
serve:
    npm run serve

# The SvelteKit client alone (vite dev under bun, proxies /api to the server).
client:
    cd client && bun run dev

# Install dependencies for both the root harness and the client.
install:
    bun install
    cd client && bun install

# Backend test suite (node's test runner over test/**).
test:
    npm test

# Typecheck both halves: the harness (tsc) and the client (svelte-check).
check:
    npm run typecheck
    cd client && bun run check

# Format the backend with prettier (the client tree uses its own conventions).
fmt:
    npm run format
