/**
 * An HTTP surface over the harness: the backend the SvelteKit client talks to.
 *
 * This is the one tier the library deliberately doesn't ship: {@link Session},
 * {@link EventStore}, and {@link MemoryStore} are UI-agnostic, and this module
 * is the thin, dependency-free adapter that exposes them over `/api/*`. It owns
 * no conversation logic of its own; every endpoint is a scoped read over a store
 * or a stream off a {@link Session}, the same way the REPL is a terminal over a
 * Session.
 *
 * Process/session model: one live Session per server process (single-user, the
 * same shape as the REPL today), backed by a shared EventStore so every turn
 * persists to disk. Past conversations are replayed *read-only* out of the log;
 * only the one live Session accepts new turns. The client's `?session=<id>`
 * deep-link lands on that read-only replay.
 *
 * The five endpoints, each named by the frontend stub that consumes it:
 *  - `POST /api/chat`            — send a message; reply streams back as SSE.
 *  - `GET  /api/events?session=` — one conversation's transcript, oldest first.
 *  - `GET  /api/sessions`        — conversation list (id + preview + count).
 *  - `GET  /api/memories`        — the curated memory store.
 *  - `GET  /api/log`            — the raw event log, newest first.
 *
 * Run it with `npm run serve` (see package.json). It speaks only core types and
 * the stores' public surface, so it stays as provider-neutral as everything
 * under `src/`.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { AnthropicClient } from "./bridge/anthropic.ts";
import { HarnessError, type ErrorKind } from "./bridge/errors.ts";
import type { ModelClient } from "./bridge/types.ts";
import { MemoryStore } from "./memory.ts";
import { EventStore, type Event } from "./events.ts";
import { OpenAIEmbedder, EmbeddingError, type Embedder } from "./embeddings.ts";
import { Session } from "./session.ts";
import type { LoopEvent } from "./bridge/loop.ts";

const BASE_SYSTEM =
    "You are a helpful, concise assistant: a long-lived Construct that remembers " +
    "across conversations. Save durable facts and preferences with memory_save, and " +
    "recall them with memory_recall. Don't save transient chatter.";

/** Compaction threshold (estimated tokens), well below the model's real window
 *  so there's headroom for the next turn. Overridable via COMPACT_AT. */
const DEFAULT_COMPACT_AT = 120_000;

/** Default page size for the conversation and log reads, overridable per query. */
const DEFAULT_PAGE = 100;

/** What the server holds for its lifetime: the stores it reads, the one live
 *  Session it drives, and a close() that releases the database handles. */
interface ServerDeps {
    store: MemoryStore;
    events: EventStore;
    session: Session;
    close(): void;
}

/**
 * Construct the cloud embedder when an OpenAI key is configured, else undefined
 * so recall stays purely lexical. Mirrors index.ts: a bad key surfaces later as
 * a caught {@link EmbeddingError}, so construction only needs the key present.
 */
function makeEmbedder(): Embedder | undefined {
    if (!process.env.OPENAI_API_KEY && !process.env.OPENAI_ADMIN_KEY) return undefined;
    try {
        return new OpenAIEmbedder();
    } catch (err) {
        console.warn(`embeddings disabled: ${err instanceof Error ? err.message : String(err)}`);
        return undefined;
    }
}

/**
 * Wire up the stores and the single live Session this server drives. The
 * EventStore and MemoryStore share one database file (one schema, one migration
 * runner), so opening either brings the whole schema current; we open both and
 * hand them to a Session, which then logs every turn it runs into the events
 * table. That wiring is what makes chats persist to disk: without the EventStore
 * a Session is purely in-memory and `/api/sessions` would have nothing to read.
 */
function buildDeps(): ServerDeps {
    const client: ModelClient = new AnthropicClient({
        model: process.env.MODEL,
        retry: {
            onRetry: ({ attempt, delayMs, error }) =>
                console.error(
                    `retrying after ${error.kind} (attempt ${attempt}, waiting ${delayMs}ms)`,
                ),
        },
    });

    const dbPath = process.env.MEMORY_DB ?? "db.sqlite";
    const store = new MemoryStore(dbPath);
    const events = new EventStore(dbPath);
    const embedder = makeEmbedder();
    const compactAt = Number(process.env.COMPACT_AT) || DEFAULT_COMPACT_AT;

    const session = new Session({
        client,
        system: BASE_SYSTEM,
        store,
        events,
        embedder,
        compaction: { thresholdTokens: compactAt },
        providerOptions: { cacheSystem: true },
    });

    return {
        store,
        events,
        session,
        close() {
            // Close the events handle first, then memory: both point at the same
            // file, and closing checkpoints the WAL, so order only affects which
            // one truncates it. Either order is correct; this is deterministic.
            events.close();
            store.close();
        },
    };
}

// ── HTTP plumbing ───────────────────────────────────────────────────────────

/** Map a harness error kind to the closest HTTP status, so the client can tell
 *  a bad key (401) from a rate-limit (429) from a transient upstream blip (502)
 *  without parsing prose. Anything unclassified is a 500. */
function statusForKind(kind: ErrorKind): number {
    switch (kind) {
        case "auth":
            return 401;
        case "invalid_request":
            return 400;
        case "rate_limit":
            return 429;
        case "overloaded":
        case "server":
            return 502;
        case "network":
        case "timeout":
            return 504;
        case "canceled":
            return 499; // nginx's "client closed request"; fitting for an abort.
        default:
            return 500;
    }
}

/** Permissive CORS, so the client works whether it reaches us same-origin (via
 *  the Vite dev proxy) or cross-origin (a separately-served static build). The
 *  harness is a local, single-user tool; there's no credentialed session to
 *  protect with a strict origin allow-list. */
function cors(res: ServerResponse): void {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
    const text = JSON.stringify(body);
    res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
    res.end(text);
}

/** Read a request body to a string, bounded so a runaway upload can't exhaust
 *  memory. Rejects past the cap rather than truncating silently. */
function readBody(req: IncomingMessage, limit = 256 * 1024): Promise<string> {
    return new Promise((resolve, reject) => {
        let size = 0;
        const chunks: Buffer[] = [];
        req.on("data", (chunk: Buffer) => {
            size += chunk.length;
            if (size > limit) {
                reject(new Error("request body too large"));
                req.destroy();
                return;
            }
            chunks.push(chunk);
        });
        req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
        req.on("error", reject);
    });
}

// ── Read views over the log ─────────────────────────────────────────────────

/** Serialize an event for the wire: a stable JSON shape the client renders. The
 *  store's `Event` is already plain data; this just fixes the field set. */
function eventToJson(e: Event) {
    return {
        id: e.id,
        ts: e.ts,
        kind: e.kind,
        role: e.role ?? null,
        content: e.content,
        meta: e.meta ?? null,
        session: e.session ?? null,
        correlation: e.correlation ?? null,
    };
}

/**
 * Group the log into one row per conversation for the conversations applet.
 *
 * The log has no `sessions` table by design (a session is just a value in the
 * `session` column), so we derive the list: scan recent events, and for each
 * distinct session keep its newest timestamp, its event count, and a preview
 * line (the most recent user message, falling back to any content). Newest
 * conversation first, matching the stub's ordering.
 *
 * This reads a bounded window (`scan` events) rather than the whole log, so the
 * list reflects recent activity without an unbounded query; the count is the
 * count *within that window*, which is what a recent-activity list wants.
 */
function summarizeSessions(events: EventStore, scan: number) {
    const recent = events.recent({ limit: scan }); // newest first
    const order: string[] = [];
    const acc = new Map<
        string,
        { session: string; when: number; count: number; preview: string }
    >();

    for (const e of recent) {
        const id = e.session;
        if (!id) continue; // events with no session id aren't conversations.
        let row = acc.get(id);
        if (!row) {
            // First time we see this session, and because `recent` is newest
            // first, this is its latest event: the right timestamp to sort by.
            row = { session: id, when: e.ts, count: 0, preview: "" };
            acc.set(id, row);
            order.push(id);
        }
        row.count++;
        // Prefer the most recent user message as the preview (what the person
        // asked); since we walk newest-first, the first user message we meet for
        // a session is its latest, so only fill the slot once.
        if (!row.preview && e.kind === "message" && e.role === "user") {
            row.preview = e.content;
        }
    }

    // Any session whose user turns fell outside the scan window still deserves a
    // preview; fall back to its newest content line.
    for (const e of recent) {
        const id = e.session;
        if (!id) continue;
        const row = acc.get(id);
        if (row && !row.preview) row.preview = e.content;
    }

    return order.map((id) => acc.get(id)!);
}

// ── SSE for chat ────────────────────────────────────────────────────────────

/** Frame and write one SSE event. Each `LoopEvent` the Session yields becomes a
 *  named SSE event carrying a JSON payload, so the client can switch on the same
 *  `kind` vocabulary the REPL renders. */
function sse(res: ServerResponse, event: string, data: unknown): void {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
}

/**
 * Stream one chat turn over SSE.
 *
 * Drives {@link Session.send}, translating each {@link LoopEvent} into an SSE
 * frame the client consumes: `text` deltas as they arrive (the reply, typed
 * out), tool lifecycle as `tool` events, and a terminal `done` carrying the
 * turn's accounting. An error mid-stream (the model fails after the headers are
 * already sent, so we can't change the status) is reported as an `error` SSE
 * event and the stream is closed: the client distinguishes it from `done`.
 *
 * Crucially, the Session was built with an EventStore, so simply running this
 * turn persists the whole exchange (the user message, every tool call/result,
 * the reply) to disk as a side effect. Nothing here writes to the log directly.
 */
async function streamChat(session: Session, text: string, res: ServerResponse): Promise<void> {
    res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
    });
    // Announce the session id up front so the client can deep-link this
    // conversation (and so a brand-new session shows up addressable immediately).
    sse(res, "open", { session: session.id });

    try {
        const turn = session.send(text);
        let next = await turn.next();
        while (!next.done) {
            const event: LoopEvent = next.value;
            switch (event.kind) {
                case "text":
                    sse(res, "text", { text: event.text });
                    break;
                case "tool_start":
                    sse(res, "tool", { phase: "start", name: event.name, args: event.args });
                    break;
                case "tool_end":
                    sse(res, "tool", {
                        phase: "end",
                        name: event.name,
                        isError: event.isError,
                    });
                    break;
                case "compacted":
                    sse(res, "compacted", { turn: event.turn });
                    break;
                // thinking / tool_call_start / tool_call_args / turn_start /
                // loop_done are not surfaced to the client (loop_done's payload
                // is folded into the `done` frame below, from the TurnResult).
            }
            next = await turn.next();
        }
        const result = next.value;
        sse(res, "done", {
            text: result.text,
            modelTurns: result.modelTurns,
            stoppedAtMaxTurns: result.stoppedAtMaxTurns,
            compactions: result.compactions,
            usage: result.usage,
        });
    } catch (err) {
        // Headers are already sent, so we can't set a status; report the failure
        // in-band. Surface the neutral HarnessError kind when we have one so the
        // client can tell auth from rate-limit from a generic blip.
        const kind = err instanceof HarnessError ? err.kind : "unknown";
        const message = err instanceof Error ? err.message : String(err);
        sse(res, "error", { kind, message });
    } finally {
        res.end();
    }
}

// ── Router ──────────────────────────────────────────────────────────────────

/** Build the request handler over a set of deps. Pure routing: it owns no
 *  state beyond the deps, so it's straightforward to exercise in a test by
 *  pointing it at an in-memory store and a fake client. */
export function createHandler(deps: ServerDeps) {
    return async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
        cors(res);
        if (req.method === "OPTIONS") {
            res.writeHead(204);
            res.end();
            return;
        }

        const url = new URL(req.url ?? "/", "http://localhost");
        const path = url.pathname;

        try {
            if (req.method === "GET" && path === "/api/health") {
                sendJson(res, 200, { ok: true, session: deps.session.id });
                return;
            }

            if (req.method === "GET" && path === "/api/sessions") {
                const scan = clampPage(url.searchParams.get("scan"), 1000, 5000);
                const sessions = summarizeSessions(deps.events, scan).map((s) => ({
                    ...s,
                    // Mark the one session that's still live (accepts new turns);
                    // the rest are read-only replays.
                    live: s.session === deps.session.id,
                }));
                sendJson(res, 200, { sessions, live: deps.session.id });
                return;
            }

            if (req.method === "GET" && path === "/api/events") {
                const session = url.searchParams.get("session");
                if (!session) {
                    sendJson(res, 400, { error: "session query param is required" });
                    return;
                }
                const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
                // Oldest first: natural reading order for a transcript.
                const rows = deps.events.recent({ session, limit }).reverse().map(eventToJson);
                sendJson(res, 200, { session, live: session === deps.session.id, events: rows });
                return;
            }

            if (req.method === "GET" && path === "/api/memories") {
                const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
                const q = url.searchParams.get("q");
                const rows = (q ? deps.store.search(q, { limit }) : deps.store.all({ limit })).map(
                    (m) => ({
                        id: m.id,
                        content: m.content,
                        tags: m.tags,
                        importance: m.importance ?? null,
                        created: m.created,
                        updated: m.updated,
                    }),
                );
                sendJson(res, 200, { memories: rows, total: deps.store.count() });
                return;
            }

            if (req.method === "GET" && path === "/api/log") {
                const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
                const kind = url.searchParams.get("kind") ?? undefined;
                const rows = deps.events.recent({ limit, kind }).map(eventToJson);
                sendJson(res, 200, { events: rows, total: deps.events.count() });
                return;
            }

            if (req.method === "POST" && path === "/api/chat") {
                let text: string;
                try {
                    const body = await readBody(req);
                    const parsed = JSON.parse(body || "{}") as { message?: unknown };
                    if (typeof parsed.message !== "string" || parsed.message.trim() === "") {
                        sendJson(res, 400, { error: "message must be a non-empty string" });
                        return;
                    }
                    text = parsed.message;
                } catch {
                    sendJson(res, 400, { error: "invalid JSON body" });
                    return;
                }
                await streamChat(deps.session, text, res);
                return;
            }

            sendJson(res, 404, { error: `no route for ${req.method} ${path}` });
        } catch (err) {
            // A failure before we've sent any bytes: classify it to a status so
            // the client gets a meaningful code, not a bare 500, where we can.
            if (res.headersSent) {
                res.end();
                return;
            }
            const status = err instanceof HarnessError ? statusForKind(err.kind) : 500;
            const message = err instanceof Error ? err.message : String(err);
            sendJson(res, status, { error: message });
        }
    };
}

/** Parse a `limit`/`scan`-style query param, falling back to `def` and capping
 *  at `max`, so a client can't ask for an unbounded read. */
function clampPage(raw: string | null, def: number, max: number): number {
    const n = raw === null ? def : Number(raw);
    if (!Number.isFinite(n) || n <= 0) return def;
    return Math.min(Math.floor(n), max);
}

/** Boot the server: build deps, start listening, and tear the stores down
 *  cleanly on SIGINT/SIGTERM so the WAL is checkpointed on exit. */
function main(): void {
    if (!process.env.ANTHROPIC_API_KEY) {
        console.error("construct-harness server: ANTHROPIC_API_KEY is not set; chat will 401.");
    }

    const deps = buildDeps();
    const handler = createHandler(deps);
    const server = createServer((req, res) => {
        handler(req, res).catch((err) => {
            console.error("unhandled request error:", err);
            if (!res.headersSent) {
                res.writeHead(500, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ error: "internal error" }));
            } else {
                res.end();
            }
        });
    });

    const port = Number(process.env.PORT) || 8787;
    server.listen(port, () => {
        console.log(`construct-harness server listening on http://localhost:${port}`);
        console.log(`  live session: ${deps.session.id}`);
    });

    let closing = false;
    const shutdown = (signal: string) => {
        if (closing) return;
        closing = true;
        console.log(`\n${signal} received, shutting down…`);
        server.close(() => {
            deps.close();
            process.exit(0);
        });
        // If connections linger (an open SSE stream), don't hang forever.
        setTimeout(() => {
            deps.close();
            process.exit(0);
        }, 3000).unref();
    };
    process.on("SIGINT", () => shutdown("SIGINT"));
    process.on("SIGTERM", () => shutdown("SIGTERM"));
}

// Only boot when run as the entry point, so importing this module (e.g. from a
// test exercising createHandler) never starts a listener. import.meta.main is
// the Node entry-point check; the argv fallback covers runtimes without it.
const isEntry =
    (import.meta as unknown as { main?: boolean }).main ?? process.argv[1]?.endsWith("server.ts");
if (isEntry) main();
