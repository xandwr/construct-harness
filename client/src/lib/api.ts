/**
 * The client's single door to the harness backend.
 *
 * Every applet reads through here rather than calling `fetch` inline, so the
 * wire shapes live in one place and match what `src/server.ts` returns. Requests
 * go to `/api/*` same-origin: in dev the Vite proxy (see vite.config.ts) forwards
 * them to the standalone server, so nothing here hardcodes a host.
 *
 * Two kinds of call: plain JSON reads ({@link getSessions}, {@link getMemories},
 * …) and the one streaming write, {@link sendChat}, which consumes the chat SSE
 * stream and invokes a callback per event. The event vocabulary mirrors the
 * server's frames, which mirror the harness's own {@link LoopEvent} kinds.
 */

/** A single logged event, as `/api/events` and `/api/log` return it. Mirrors the
 *  harness `Event`, with nullable fields explicit for the wire. */
export interface WireEvent {
    id: number;
    ts: number;
    kind: string;
    role: string | null;
    content: string;
    meta: unknown;
    session: string | null;
    correlation: string | null;
}

/** One conversation in the `/api/sessions` list. `live` marks a conversation the
 *  server currently holds in memory (loaded in its session pool). Any
 *  conversation can be resumed by sending a turn into it, so `live` is "loaded
 *  now", not "the only one you can continue". */
export interface SessionSummary {
    session: string;
    when: number;
    count: number;
    preview: string;
    live: boolean;
}

/** A curated memory, as `/api/memories` returns it. */
export interface WireMemory {
    id: number;
    content: string;
    tags: string[];
    importance: number | null;
    created: number;
    updated: number;
}

/** A knowledge-base note in the list (summary) shape: no body. */
export interface WireNoteSummary {
    id: number;
    uuid: string;
    path: string;
    title: string;
    frontmatter: Record<string, string | number | boolean | null | string[]>;
    created: number;
    updated: number;
}

/** One relation a note holds (note -> memory or note -> note). */
export interface WireNoteLink {
    id: number;
    toMemory: number | null;
    toNote: number | null;
    kind: string | null;
}

/** A note in the detail shape: summary fields plus body and links. */
export interface WireNote extends WireNoteSummary {
    content: string;
    links: WireNoteLink[];
}

/** One parameter a slash command accepts, as `/api/commands` returns it. Mirrors
 *  the harness `CommandParam`: a placeholder name, a hint, and whether it's
 *  required (which decides the `<name>` vs `[name]` bracket in the signature). */
export interface WireCommandParam {
    name: string;
    description: string;
    required: boolean;
}

/** A slash command as `/api/commands` advertises it. Mirrors the harness
 *  `SlashCommand`: the keyword (no leading slash), a one-line description, its
 *  parameters in signature order, and any alias keywords. The chat composer
 *  lists these in its `/` menu. */
export interface WireCommand {
    name: string;
    description: string;
    params: WireCommandParam[];
    aliases?: string[];
}

/** Thrown when a JSON read fails; carries the HTTP status so a caller can tell
 *  a 401 (no/invalid API key on the server) from a 502 (upstream blip). */
export class ApiError extends Error {
    constructor(
        message: string,
        readonly status: number,
    ) {
        super(message);
        this.name = "ApiError";
    }
}

/** Run a JSON GET, throwing {@link ApiError} on a non-2xx so callers can branch
 *  on `.status` and surface a real message instead of a blank failure. */
async function getJson<T>(path: string, fetchFn: typeof fetch = fetch): Promise<T> {
    const res = await fetchFn(path);
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new ApiError(
            (body as { error?: string }).error ?? `request failed (${res.status})`,
            res.status,
        );
    }
    return res.json() as Promise<T>;
}

/** The conversation list, newest first. `live` lists the ids of the
 *  conversations the server currently holds in memory (also flagged per-row). */
export function getSessions(
    fetchFn?: typeof fetch,
): Promise<{ sessions: SessionSummary[]; live: string[] }> {
    return getJson("/api/sessions", fetchFn);
}

/** One conversation's transcript, oldest first (reading order). */
export function getEvents(
    session: string,
    fetchFn?: typeof fetch,
): Promise<{ session: string; live: boolean; events: WireEvent[] }> {
    return getJson(`/api/events?session=${encodeURIComponent(session)}`, fetchFn);
}

/** The curated memory store, ordered by importance then recency. */
export function getMemories(
    fetchFn?: typeof fetch,
): Promise<{ memories: WireMemory[]; total: number }> {
    return getJson("/api/memories", fetchFn);
}

/** The raw event log, newest first. */
export function getLog(fetchFn?: typeof fetch): Promise<{ events: WireEvent[]; total: number }> {
    return getJson("/api/log", fetchFn);
}

/** The slash-command catalogue, for the chat composer's `/` menu. Static for the
 *  process, so a caller can fetch it once and filter client-side as the human
 *  types. */
export function getCommands(fetchFn?: typeof fetch): Promise<{ commands: WireCommand[] }> {
    return getJson("/api/commands", fetchFn);
}

/** Run a JSON write (POST/PUT/DELETE), throwing {@link ApiError} on a non-2xx so
 *  callers can surface the server's message (a 400 path clash, a 404, ...). */
async function writeJson<T>(
    method: "POST" | "PUT" | "DELETE",
    path: string,
    body?: unknown,
    fetchFn: typeof fetch = fetch,
): Promise<T> {
    const res = await fetchFn(path, {
        method,
        headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
        body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new ApiError(
            (errBody as { error?: string }).error ?? `request failed (${res.status})`,
            res.status,
        );
    }
    return res.json() as Promise<T>;
}

/** The knowledge-base note list (summary shape), newest first. Optional `q`
 *  full-text search and `prefix` folder filter. */
export function getNotes(
    opts: { q?: string; prefix?: string } = {},
    fetchFn?: typeof fetch,
): Promise<{ notes: WireNoteSummary[]; total: number }> {
    const params = new URLSearchParams();
    if (opts.q) params.set("q", opts.q);
    if (opts.prefix) params.set("prefix", opts.prefix);
    const qs = params.toString();
    return getJson(`/api/notes${qs ? `?${qs}` : ""}`, fetchFn);
}

/** One note with its body and links. */
export function getNote(uuid: string, fetchFn?: typeof fetch): Promise<{ note: WireNote }> {
    return getJson(`/api/notes/${encodeURIComponent(uuid)}`, fetchFn);
}

/** Create a note. Returns the created note (detail shape). */
export function createNote(
    input: {
        title: string;
        content: string;
        path?: string;
        frontmatter?: WireNoteSummary["frontmatter"];
    },
    fetchFn?: typeof fetch,
): Promise<{ note: WireNote }> {
    return writeJson("POST", "/api/notes", input, fetchFn);
}

/** Update a note by uuid. Only the provided fields change. */
export function updateNote(
    uuid: string,
    patch: {
        title?: string;
        content?: string;
        path?: string;
        frontmatter?: WireNoteSummary["frontmatter"];
    },
    fetchFn?: typeof fetch,
): Promise<{ note: WireNote }> {
    return writeJson("PUT", `/api/notes/${encodeURIComponent(uuid)}`, patch, fetchFn);
}

/** Delete a note by uuid. */
export function deleteNote(uuid: string, fetchFn?: typeof fetch): Promise<{ deleted: boolean }> {
    return writeJson("DELETE", `/api/notes/${encodeURIComponent(uuid)}`, undefined, fetchFn);
}

/** The events the chat SSE stream delivers, in the same `kind` vocabulary the
 *  server frames. A turn yields zero or more of these, then exactly one of
 *  `done` or `error`. */
export type ChatEvent =
    | { kind: "open"; session: string }
    | { kind: "text"; text: string }
    | { kind: "thinking"; text: string }
    | { kind: "tool"; phase: "start" | "end"; name: string; args?: unknown; isError?: boolean }
    | { kind: "compacted"; turn: number }
    | {
          kind: "done";
          text: string;
          modelTurns: number;
          stoppedAtMaxTurns: boolean;
          compactions: number;
          usage: { inputTokens: number; outputTokens: number; cacheReadTokens: number };
      }
    | { kind: "error"; errorKind: string; message: string };

/**
 * Send one chat message and consume the reply as it streams.
 *
 * POSTs to `/api/chat` and parses the SSE response by hand (the streams `fetch`
 * gives us are friendlier here than `EventSource`, which can't POST). `onEvent`
 * fires for every frame as it arrives — `text` deltas to append to the reply,
 * `tool` events to show activity, and a terminal `done` or `error`. Resolves
 * when the stream closes.
 *
 * `opts.session` continues (resuming if needed) that conversation; omit it to
 * land on the server's default live conversation. The first frame is always an
 * `open` carrying the session id the turn actually ran under, so a caller that
 * started a fresh conversation learns its id.
 *
 * `signal` aborts the turn (the server sees the connection drop). A network
 * failure before any frame is thrown as an {@link ApiError}; a model failure
 * mid-stream arrives as an `error` event, not a throw, because by then the
 * response is a committed 200 (see the server's streamChat).
 */
export async function sendChat(
    message: string,
    onEvent: (event: ChatEvent) => void,
    opts: { session?: string; signal?: AbortSignal; fetchFn?: typeof fetch } = {},
): Promise<void> {
    const fetchFn = opts.fetchFn ?? fetch;
    const res = await fetchFn("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(opts.session ? { message, session: opts.session } : { message }),
        signal: opts.signal,
    });

    if (!res.ok || !res.body) {
        const body = await res.json().catch(() => ({}));
        throw new ApiError(
            (body as { error?: string }).error ?? `chat failed (${res.status})`,
            res.status,
        );
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    // SSE frames are separated by a blank line; accumulate bytes and split on the
    // double-newline boundary, parsing each complete frame as it lands.
    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sep: number;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            const event = parseFrame(frame);
            if (event) onEvent(event);
        }
    }
}

/** Parse one SSE frame (`event:` + `data:` lines) into a {@link ChatEvent}, or
 *  null if it isn't one we recognize. The server only emits `data` as a single
 *  JSON line per frame, so we don't handle multi-line data. */
function parseFrame(frame: string): ChatEvent | null {
    let name = "";
    let data = "";
    for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data = line.slice(5).trim();
    }
    if (!name || !data) return null;

    let payload: Record<string, unknown>;
    try {
        payload = JSON.parse(data);
    } catch {
        return null;
    }

    switch (name) {
        case "open":
            return { kind: "open", session: String(payload.session) };
        case "text":
            return { kind: "text", text: String(payload.text) };
        case "thinking":
            return { kind: "thinking", text: String(payload.text) };
        case "tool":
            return {
                kind: "tool",
                phase: payload.phase === "end" ? "end" : "start",
                name: String(payload.name),
                args: payload.args,
                isError: Boolean(payload.isError),
            };
        case "compacted":
            return { kind: "compacted", turn: Number(payload.turn) };
        case "done": {
            const u = (payload.usage ?? {}) as Record<string, unknown>;
            return {
                kind: "done",
                text: String(payload.text ?? ""),
                modelTurns: Number(payload.modelTurns ?? 0),
                stoppedAtMaxTurns: Boolean(payload.stoppedAtMaxTurns),
                compactions: Number(payload.compactions ?? 0),
                usage: {
                    inputTokens: Number(u.inputTokens ?? 0),
                    outputTokens: Number(u.outputTokens ?? 0),
                    cacheReadTokens: Number(u.cacheReadTokens ?? 0),
                },
            };
        }
        case "error":
            return {
                kind: "error",
                errorKind: String(payload.kind ?? "unknown"),
                message: String(payload.message ?? "stream error"),
            };
        default:
            return null;
    }
}
