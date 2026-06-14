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
 * Process/session model: a {@link SessionPool} of live Sessions, all sharing one
 * EventStore so every turn persists to disk. The pool starts with one fresh
 * conversation; chatting into a past conversation (a `session` id on `/api/chat`)
 * resumes it from the log into the pool, where it stays live and resumable. So
 * the client's `?session=<id>` deep-link isn't a read-only replay: sending a turn
 * there picks the conversation back up exactly where it left off. Reads
 * (`/api/sessions`, `/api/events`) report which conversations are currently live
 * (loaded in the pool) versus only on disk.
 *
 * The five endpoints, each named by the frontend stub that consumes it:
 *  - `POST /api/chat`            — send a message (optional `session` to resume
 *                                  a past conversation); reply streams as SSE.
 *  - `GET  /api/events?session=` — one conversation's transcript, oldest first.
 *  - `GET  /api/sessions`        — conversation list (id + preview + count).
 *  - `GET  /api/memories`        — the curated memory store.
 *  - `GET  /api/log`            — the raw event log, newest first.
 *  - `GET  /api/commands`        — the slash-command catalogue, for the client's
 *                                  `/` menu (mirrors {@link BUILTIN_COMMANDS}).
 *
 * Run it with `npm run serve` (see package.json). It speaks only core types and
 * the stores' public surface, so it stays as provider-neutral as everything
 * under `src/`.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { AnthropicClient, type ServerToolName } from "./bridge/anthropic.ts";
import { HarnessError, type ErrorKind } from "./bridge/errors.ts";
import type { ModelClient } from "./bridge/types.ts";
import { MemoryStore } from "./memory.ts";
import { EventStore, type Event } from "./events.ts";
import { backfillEventEmbeddings } from "./eventTools.ts";
import { GoalStore } from "./goals.ts";
import { OpenAIEmbedder, EmbeddingError, type Embedder } from "./embeddings.ts";
import { Session, type SessionConfig } from "./session.ts";
import type { LoopEvent } from "./bridge/loop.ts";
import { NotesStore, Note, NoteError, type NoteFrontmatter } from "./notes.ts";
import { NotesService } from "./notesService.ts";
import { noteTools } from "./noteTools.ts";
import { shellTools } from "./shellTools.ts";
import { BUILTIN_COMMANDS } from "./commands.ts";

const BASE_SYSTEM =
    "You are a helpful, concise assistant: a long-lived Construct that remembers " +
    "across conversations. Save durable facts and preferences with memory_save, and " +
    "recall them with memory_recall. Don't save transient chatter. When the human " +
    "gives you a task worth holding across turns, track it with goal_set and mark it " +
    "goal_update done when achieved; your active goals are shown to you each turn. To " +
    "look back over what actually happened earlier in this conversation (past " +
    "messages, whether a tool already ran, what was decided), search your transcript " +
    "with transcript_recall. For longer-form documentation the human also edits (runbooks, " +
    "references, design notes), use the knowledge base: note_save / note_update to " +
    "write, note_recall to read it when relevant, note_link to relate a note to a " +
    "memory or another note. You also have two ways to run code: the sandboxed " +
    "code-execution tool for disposable computation, and use__user__shell to run " +
    "commands on the user's real local machine (their files, tools, and working " +
    "directory) when the work has to touch this environment.";

/** Compaction threshold (estimated tokens), well below the model's real window
 *  so there's headroom for the next turn. Overridable via COMPACT_AT. */
const DEFAULT_COMPACT_AT = 120_000;

/** Default page size for the conversation and log reads, overridable per query. */
const DEFAULT_PAGE = 100;

/** What the server holds for its lifetime: the stores it reads, the pool of live
 *  Sessions it drives, the knowledge-base service, and a close() that releases the
 *  database handles. `notes` is optional so the server still runs without a KB
 *  (the routes 503 cleanly), but the default wiring always provides one. */
interface ServerDeps {
    store: MemoryStore;
    events: EventStore;
    goals: GoalStore;
    /** The live conversations this process holds in memory, keyed by session id.
     *  Each one accepts new turns; a conversation only in the log (not here) is a
     *  past one waiting to be resumed into the pool on its next turn. */
    sessions: SessionPool;
    notes?: NotesService;
    notesStore?: NotesStore;
    corsOrigin?: string;
    close(): void;
}

/** Builds the shared {@link SessionConfig} every live conversation runs under,
 *  carrying the `events` log so {@link Session.resume} has a transcript to
 *  rehydrate from. It deliberately leaves `sessionId` unset (a fresh
 *  conversation gets a fresh id); the {@link SessionPool} pins the id when
 *  resuming a specific conversation. {@link buildDeps} supplies the closure with
 *  everything wired in. */
export type SessionConfigBase = () => SessionConfig & { events: EventStore };

/**
 * The set of live Sessions a server process drives, keyed by session id.
 *
 * The harness keeps every conversation in one shared log, but a {@link Session}
 * (the in-memory thing that holds working context and accepts turns) is more
 * expensive: it carries the rehydrated history and pending embeds. So we keep a
 * pool, not one-Session-per-process and not one-per-conversation-ever: a
 * conversation becomes live the first time it's chatted into ({@link resolve}
 * resumes it from the log), stays live for the process's life, and any
 * conversation in the log can be resumed this way. {@link has} lets a read
 * endpoint report which conversations are currently live (in the pool) versus
 * merely on disk.
 */
export class SessionPool {
    private readonly sessions = new Map<string, Session>();
    /** Builds the shared config every conversation runs under (see
     *  {@link SessionConfigBase}); the pool pins the id for a resume. */
    private readonly config: SessionConfigBase;
    /** The conversation a chat with no session id lands on, so a client that
     *  never names one keeps talking to the same conversation across turns. */
    private readonly defaultId: string;

    /** Register one brand-new live conversation at boot via {@link config}, so
     *  the server always has an addressable live conversation even before anyone
     *  resumes a past one; its id becomes the default. */
    constructor(config: SessionConfigBase) {
        this.config = config;
        const initial = new Session(config());
        this.sessions.set(initial.id, initial);
        this.defaultId = initial.id;
    }

    /** Whether a conversation is currently live (has an in-memory Session in the
     *  pool), as opposed to only existing in the log. */
    has(id: string): boolean {
        return this.sessions.has(id);
    }

    /** The ids of every live conversation in the pool. */
    ids(): string[] {
        return [...this.sessions.keys()];
    }

    /**
     * Get the live Session for `id`, resuming it from the log into the pool if it
     * isn't live yet. With no `id` the caller wants "a live conversation to talk
     * to": return the default conversation so a client that never sends a session
     * id keeps landing on the same one across turns.
     *
     * Resuming rehydrates the conversation's prior turns into the new Session's
     * working context (see {@link Session.resume}), so the first turn after a
     * resume builds on the real exchange, not an empty history. A brand-new id
     * (one with no events in the log) resumes to an empty history, which is the
     * correct behavior: it just becomes a fresh live conversation under that id.
     */
    async resolve(id: string | undefined): Promise<Session> {
        if (!id) return this.sessions.get(this.defaultId)!;
        const live = this.sessions.get(id);
        if (live) return live;
        const resumed = await Session.resume({ ...this.config(), sessionId: id });
        this.sessions.set(resumed.id, resumed);
        return resumed;
    }
}

/** Where the knowledge-base markdown folder lives, mirroring MEMORY_DB. */
const DEFAULT_KB_DIR = "kb";

/** The provider-hosted tools enabled by default: live web access so the Construct
 *  can answer about the current world. Code execution is opt-in (it spins up a
 *  sandbox and bills accordingly), so it's left out of the default set. */
const DEFAULT_SERVER_TOOLS: ServerToolName[] = ["web_search", "web_fetch"];

/** The full set a caller may name in SERVER_TOOLS, for validation. */
const KNOWN_SERVER_TOOLS: ServerToolName[] = ["web_search", "web_fetch", "code_execution"];

/**
 * Resolve which provider-hosted tools to enable from the `SERVER_TOOLS` env var.
 * Unset uses {@link DEFAULT_SERVER_TOOLS} (web access on). A comma-separated list
 * picks an explicit set (e.g. `web_search,code_execution`); `none` (or an empty
 * value) disables them entirely. Unknown names are dropped with a warning so a
 * typo degrades to fewer tools rather than a crash.
 */
export function resolveServerTools(raw: string | undefined): ServerToolName[] {
    if (raw === undefined) return DEFAULT_SERVER_TOOLS;
    const trimmed = raw.trim().toLowerCase();
    if (trimmed === "" || trimmed === "none") return [];
    const out: ServerToolName[] = [];
    for (const part of trimmed
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)) {
        if ((KNOWN_SERVER_TOOLS as string[]).includes(part)) {
            out.push(part as ServerToolName);
        } else {
            console.warn(`SERVER_TOOLS: ignoring unknown tool "${part}"`);
        }
    }
    return out;
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
    const goals = new GoalStore(dbPath);
    const embedder = makeEmbedder();
    const compactAt = Number(process.env.COMPACT_AT) || DEFAULT_COMPACT_AT;
    const serverTools = resolveServerTools(process.env.SERVER_TOOLS);

    // The knowledge base shares the same database file (one schema, one migration
    // runner) and a markdown folder on disk. The NotesService starts its watcher
    // asynchronously: an initial scan adopts any files created while the process
    // was down, then live two-way sync begins. A failure to start the watcher is
    // logged but non-fatal: the in-app KB still works over the API.
    const kbDir = process.env.KB_DIR ?? DEFAULT_KB_DIR;
    const notesStore = new NotesStore(dbPath);
    const notes = new NotesService({ store: notesStore, root: kbDir, embedder });

    // One wiring shared by every live conversation. Every Session in the pool —
    // the boot one and every resumed past conversation — runs under this exact
    // configuration; the pool pins the `sessionId` when resuming a specific
    // conversation, leaving it fresh here. Keeping it in one closure is what
    // guarantees a resumed conversation behaves identically to the one it's
    // continuing: same tools, same compaction, same provider options.
    const sessionConfig: SessionConfigBase = () => ({
        client,
        system: BASE_SYSTEM,
        store,
        events,
        goals,
        embedder,
        // The agent opts into the KB: it gets the note tools (save/update/recall/
        // link) but notes are not auto-injected into context the way memories are.
        // It also gets use__user__shell, the unguarded local counterpart to the
        // sandboxed code_execution server tool wired in via serverTools below.
        tools: [...noteTools(notes, notesStore, embedder), ...shellTools()],
        compaction: { thresholdTokens: compactAt },
        // Cache the system prefix; turn on adaptive thinking with a summarized
        // display so the streaming path emits readable `thinking` deltas the
        // client can show (see streamChat); and give the Construct provider-hosted
        // web access (search + fetch) so it can answer about the live world. The
        // model runs these server-side, so there's no tool loop to drive them.
        providerOptions: {
            cacheSystem: true,
            thinking: true,
            thinkingDisplay: true,
            serverTools,
        },
    });
    const sessions = new SessionPool(sessionConfig);

    notes
        .start()
        .then(() => console.log(`  knowledge base watching: ${notes.kbRoot}`))
        .catch((err) =>
            console.warn(
                `knowledge-base sync disabled: ${err instanceof Error ? err.message : String(err)}`,
            ),
        );

    // The Session embeds each new message turn as it's logged, but a log that
    // predates this (turns recorded before an embedder was wired up) is
    // lexical-only until backfilled. Run one bounded catch-up pass at startup so
    // semantic transcript_recall covers the existing transcript too. Fire-and-
    // forget like the watcher above: an embedding outage just leaves those rows
    // lexical, and a slow embed must not block the server accepting requests.
    if (embedder) {
        backfillEventEmbeddings(events, embedder)
            .then((n) => {
                if (n) console.log(`  embedded ${n} past event${n === 1 ? "" : "s"} for recall`);
            })
            .catch((err) =>
                console.warn(
                    `event-embedding backfill skipped: ${err instanceof Error ? err.message : String(err)}`,
                ),
            );
    }

    return {
        store,
        events,
        goals,
        sessions,
        notes,
        notesStore,
        corsOrigin: process.env.CORS_ORIGIN,
        close() {
            // Stop the watcher first so no inbound event races the store closing.
            notes.close();
            // Close the handles. All point at the same file; closing checkpoints
            // the WAL, so order only affects which one truncates it. Either order
            // is correct; this is deterministic.
            events.close();
            goals.close();
            notesStore.close();
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
function cors(res: ServerResponse, origin = "*"): void {
    res.setHeader("Access-Control-Allow-Origin", origin || "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    if (origin && origin !== "*") res.setHeader("Vary", "Origin");
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

/** Serialize a note for the list wire shape (no body, for a compact list). */
function noteToSummaryJson(n: Note) {
    return {
        id: n.id,
        uuid: n.uuid,
        path: n.path,
        title: n.title,
        frontmatter: n.frontmatter,
        created: n.created,
        updated: n.updated,
    };
}

/** Serialize a note for the detail wire shape (with body and its links). */
function noteToDetailJson(notesStore: NotesStore, n: Note) {
    return {
        ...noteToSummaryJson(n),
        content: n.content,
        links: notesStore.linksFrom(n.id).map((l) => ({
            id: l.id,
            toMemory: l.toMemory,
            toNote: l.toNote,
            kind: l.kind,
        })),
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
                case "thinking":
                    // The model's reasoning trace, streamed so the client can show
                    // it live (collapsible). Only present when thinking is enabled
                    // on the provider (see buildDeps' providerOptions).
                    sse(res, "thinking", { text: event.text });
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
                // tool_call_start / tool_call_args / turn_start / loop_done are not
                // surfaced to the client (loop_done's payload is folded into the
                // `done` frame below, from the TurnResult).
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

// ── Knowledge-base routes ────────────────────────────────────────────────────

/** Read a request body and parse it as a JSON object, or return null on any
 *  failure (the caller then 400s). Bounded by {@link readBody}. */
async function readJsonObject(req: IncomingMessage): Promise<Record<string, unknown> | null> {
    try {
        const body = await readBody(req);
        const parsed: unknown = JSON.parse(body || "{}");
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
        return parsed as Record<string, unknown>;
    } catch {
        return null;
    }
}

/**
 * Handle every `/api/notes` and `/api/notes/:uuid` request: the KB's read and
 * write surface. All mutations go through {@link NotesService}, so a write here
 * and a file saved in an editor converge to the same row (the unified write
 * path). Translates {@link NoteError} (bad input, a path/uuid clash) to a 400 so
 * the client gets a real message rather than an opaque 500.
 *
 *  - GET    /api/notes            list (optional `q` search, `prefix` folder)
 *  - GET    /api/notes/:uuid      one note with body + links
 *  - POST   /api/notes            create { title, content, path?, frontmatter? }
 *  - PUT    /api/notes/:uuid      update { title?, content?, path?, frontmatter? }
 *  - DELETE /api/notes/:uuid      delete
 */
async function handleNotes(
    req: IncomingMessage,
    res: ServerResponse,
    deps: ServerDeps,
    url: URL,
    path: string,
): Promise<void> {
    const { notes, notesStore } = deps;
    if (!notes || !notesStore) {
        sendJson(res, 503, { error: "knowledge base is not configured" });
        return;
    }

    // The collection endpoint: list (GET) and create (POST).
    if (path === "/api/notes") {
        if (req.method === "GET") {
            const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
            const q = url.searchParams.get("q");
            const prefix = url.searchParams.get("prefix") ?? undefined;
            const rows = (
                q
                    ? notesStore.search(q, { limit, pathPrefix: prefix })
                    : notesStore.all({ limit, pathPrefix: prefix })
            ).map(noteToSummaryJson);
            sendJson(res, 200, {
                notes: rows,
                total: notesStore.count({ q: q ?? undefined, pathPrefix: prefix }),
            });
            return;
        }
        if (req.method === "POST") {
            const body = await readJsonObject(req);
            if (!body) {
                sendJson(res, 400, { error: "invalid JSON body" });
                return;
            }
            if (typeof body.title !== "string" || body.title.trim() === "") {
                sendJson(res, 400, { error: "title must be a non-empty string" });
                return;
            }
            try {
                const result = await notes.create({
                    title: body.title,
                    content: typeof body.content === "string" ? body.content : "",
                    path: typeof body.path === "string" ? body.path : undefined,
                    frontmatter: asFrontmatter(body.frontmatter),
                });
                sendJson(res, 201, { note: noteToDetailJson(notesStore, result.note) });
            } catch (err) {
                sendNoteError(res, err);
            }
            return;
        }
        sendJson(res, 405, { error: `method ${req.method} not allowed on /api/notes` });
        return;
    }

    // The item endpoint: /api/notes/:uuid (read, update, delete).
    const uuid = decodeURIComponent(path.slice("/api/notes/".length));
    if (uuid === "" || uuid.includes("/")) {
        sendJson(res, 404, { error: "note not found" });
        return;
    }
    const existing = notesStore.getByUuid(uuid);

    if (req.method === "GET") {
        if (!existing) {
            sendJson(res, 404, { error: "note not found" });
            return;
        }
        sendJson(res, 200, { note: noteToDetailJson(notesStore, existing) });
        return;
    }

    if (req.method === "PUT") {
        if (!existing) {
            sendJson(res, 404, { error: "note not found" });
            return;
        }
        const body = await readJsonObject(req);
        if (!body) {
            sendJson(res, 400, { error: "invalid JSON body" });
            return;
        }
        // Build a patch from only the provided fields, so an omitted field is left
        // untouched rather than cleared.
        const patch: Parameters<NotesService["update"]>[1] = {};
        if (typeof body.title === "string") patch.title = body.title;
        if (typeof body.content === "string") patch.content = body.content;
        if (typeof body.path === "string") patch.path = body.path;
        if ("frontmatter" in body) patch.frontmatter = asFrontmatter(body.frontmatter);
        try {
            const result = await notes.update(existing.id, patch);
            if (!result) {
                sendJson(res, 404, { error: "note not found" });
                return;
            }
            sendJson(res, 200, { note: noteToDetailJson(notesStore, result.note) });
        } catch (err) {
            sendNoteError(res, err);
        }
        return;
    }

    if (req.method === "DELETE") {
        if (!existing) {
            sendJson(res, 404, { error: "note not found" });
            return;
        }
        const removed = await notes.remove(existing.id);
        sendJson(res, 200, { deleted: removed });
        return;
    }

    sendJson(res, 405, { error: `method ${req.method} not allowed on ${path}` });
}

/** Coerce an unknown request field into a frontmatter map, dropping any value
 *  that isn't a supported scalar or string array (the store re-validates too). */
function asFrontmatter(value: unknown): NoteFrontmatter {
    if (value === null || typeof value !== "object" || Array.isArray(value)) return {};
    const out: NoteFrontmatter = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        if (v === null || typeof v === "string" || typeof v === "boolean") {
            out[k] = v;
        } else if (typeof v === "number" && Number.isFinite(v)) {
            out[k] = v;
        } else if (Array.isArray(v) && v.every((x) => typeof x === "string")) {
            out[k] = v as string[];
        }
    }
    return out;
}

/** Map a NoteError to a 400 (bad input / clash); anything else to a 500. */
function sendNoteError(res: ServerResponse, err: unknown): void {
    if (err instanceof NoteError) {
        sendJson(res, 400, { error: err.message });
        return;
    }
    const message = err instanceof Error ? err.message : String(err);
    sendJson(res, 500, { error: message });
}

// ── Router ──────────────────────────────────────────────────────────────────

/** Build the request handler over a set of deps. Pure routing: it owns no
 *  state beyond the deps, so it's straightforward to exercise in a test by
 *  pointing it at an in-memory store and a fake client. */
export function createHandler(deps: ServerDeps) {
    return async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
        cors(res, deps.corsOrigin);
        if (req.method === "OPTIONS") {
            res.writeHead(204);
            res.end();
            return;
        }

        const url = new URL(req.url ?? "/", "http://localhost");
        const path = url.pathname;

        try {
            if (req.method === "GET" && path === "/api/health") {
                sendJson(res, 200, { ok: true, sessions: deps.sessions.ids() });
                return;
            }

            if (req.method === "GET" && path === "/api/sessions") {
                const scan = clampPage(url.searchParams.get("scan"), 1000, 5000);
                const live = deps.sessions.ids();
                const liveSet = new Set(live);
                const sessions = summarizeSessions(deps.events, scan).map((s) => ({
                    ...s,
                    // Mark every conversation currently held live in the pool
                    // (in-memory, accepting turns). The rest live only in the log
                    // until they're resumed — which any of them can be, by sending
                    // a turn into them. So `live` is "loaded now", not "the only one
                    // you can talk to".
                    live: liveSet.has(s.session),
                }));
                sendJson(res, 200, { sessions, live });
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
                sendJson(res, 200, {
                    session,
                    live: deps.sessions.has(session),
                    events: rows,
                });
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

            // The slash-command catalogue: a static read of the registry, so the
            // client can list the same commands (name, signature parts, params)
            // in its `/` menu that the REPL prints under `/help`. No per-session
            // state: the menu is the same for every conversation.
            if (req.method === "GET" && path === "/api/commands") {
                sendJson(res, 200, { commands: BUILTIN_COMMANDS });
                return;
            }

            // Knowledge-base routes: list/read are GETs; create/update/delete are
            // the harness's first write endpoints, all funneling through the one
            // NotesService write path so a UI write and a file save converge.
            if (path === "/api/notes" || path.startsWith("/api/notes/")) {
                await handleNotes(req, res, deps, url, path);
                return;
            }

            if (req.method === "POST" && path === "/api/chat") {
                let text: string;
                let wantSession: string | undefined;
                try {
                    const body = await readBody(req);
                    const parsed = JSON.parse(body || "{}") as {
                        message?: unknown;
                        session?: unknown;
                    };
                    if (typeof parsed.message !== "string" || parsed.message.trim() === "") {
                        sendJson(res, 400, { error: "message must be a non-empty string" });
                        return;
                    }
                    text = parsed.message;
                    // An optional session id resumes (or continues) that
                    // conversation; omitted, the turn lands on the default live one.
                    if (typeof parsed.session === "string" && parsed.session !== "") {
                        wantSession = parsed.session;
                    }
                } catch {
                    sendJson(res, 400, { error: "invalid JSON body" });
                    return;
                }
                // Resolve the conversation before streaming: resuming reads the log
                // and rehydrates history, which can fail (a store error), so do it
                // here where we can still send a real error status. Once streamChat
                // writes the SSE head we can only report errors in-band.
                const session = await deps.sessions.resolve(wantSession);
                await streamChat(session, text, res);
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
        console.log(`  live session: ${deps.sessions.ids().join(", ")}`);
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
