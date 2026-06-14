/**
 * Bridges the {@link EventStore} into the agentic loop: the Construct's tool for
 * recalling its own past.
 *
 * Where {@link memoryTools} expose the *curated* facts the agent chose to keep,
 * this exposes the raw, total record beneath them: every message, tool call, and
 * tool result the log captured, queryable by meaning, by word, or by recency.
 * It's the sensory counterpart to memory: memory is what the Construct decided
 * was worth remembering; the transcript is what actually happened. An agent that
 * can read it can answer "what did I do earlier", "did this tool already run",
 * "what was decided three turns back" without the harness having to replay the
 * whole conversation into context.
 *
 * Read-only by contract, mirroring the store: the log has no content UPDATE or
 * DELETE (immutability is the substrate's whole value), so there is no
 * `transcript_forget`. The tool speaks plain JSON in and out (its `run` result
 * drops straight into a `tool_result` part) and never throws past the loop: a
 * bad query degrades to an empty result, not an error.
 */

import type { ToolDef } from "./types.ts";
import { EventStore, EventError } from "./events.ts";
import type { Event, EventQuery } from "./events.ts";
import { embedOne, EmbeddingError, type Embedder } from "./embeddings.ts";

/** How many events `transcript_recall` returns by default. Smaller than memory's
 *  recall: a transcript line is verbose (a whole tool result), so a tighter
 *  default keeps the turn's budget in check while still spanning several turns. */
export const DEFAULT_TRANSCRIPT_LIMIT = 8;

/** Per-event body cap in the recall view. A single tool_result can be enormous
 *  (a whole file, a long search dump); trim it so recalling several events stays
 *  bounded. The agent still sees enough to recognize the event and can re-run the
 *  underlying tool if it needs the full payload. */
const RECALL_CONTENT_CAP = 2_000;

/** The serializable view of a logged event handed back to the model. A subset of
 *  the stored {@link Event}: the fields useful for reasoning about the past,
 *  without the embedding machinery or the raw `meta` blob (its salient parts ride
 *  in `content`/`kind`/`role`). `correlation` is kept so the agent can thread a
 *  tool_call to its tool_result itself. */
export interface EventView {
    id: number;
    ts: number;
    kind: string;
    role?: string;
    content: string;
    correlation?: string;
}

function toView(e: Event): EventView {
    let content = e.content;
    if (content.length > RECALL_CONTENT_CAP) {
        content = content.slice(0, RECALL_CONTENT_CAP) + "\n…[truncated]";
    }
    return {
        id: e.id,
        ts: e.ts,
        kind: e.kind,
        role: e.role,
        content,
        correlation: e.correlation,
    };
}

/** Narrow an unknown args bag to a record without trusting its fields yet. */
function asRecord(args: unknown): Record<string, unknown> {
    return typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
}

/** Coerce an unknown to a finite number, or undefined. Used for the `since` /
 *  `until` time bounds and `limit`, which the model may omit or fudge. */
function asNumber(value: unknown): number | undefined {
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * The shared transcript ranking, the events counterpart to memory's and notes'.
 * Order of preference:
 *   1. Semantic (cosine) match, when an embedder is configured and the query
 *      embeds: finds events that *mean* the same thing. Most events never get a
 *      vector (the index is selective), so this only surfaces the embedded ones.
 *   2. Lexical (FTS/bm25) match: shared words, across the whole log (every event
 *      is FTS-indexed for free).
 *   3. Recency order: when there's no query or nothing matched, the most recent
 *      events under the filters.
 * Each step falls through to the next so recall is never worse than recency.
 */
async function recallEvents(
    store: EventStore,
    embedder: Embedder | undefined,
    query: string,
    opts: EventQuery,
): Promise<Event[]> {
    const trimmed = query.trim();

    if (trimmed && embedder) {
        try {
            const vec = await embedOne(embedder, trimmed);
            const hits = store.semanticSearch(vec, opts).map((h) => h.event);
            if (hits.length) return hits;
        } catch (err) {
            // Fall through to lexical on any embedding failure (mirrors memory).
            if (!(err instanceof EmbeddingError)) throw err;
        }
    }

    if (trimmed) {
        const lexical = store.searchRelevant(trimmed, opts);
        if (lexical.length) return lexical;
    }

    // No query, or nothing matched: most-recent under the filters.
    return store.recent(opts);
}

/**
 * Build the transcript tool set over an event log.
 *
 * `sessionId` scopes recall to *this* Construct's own conversation by default:
 * introspecting your own past is the common case, and it keeps one session from
 * dredging up another's turns unless it asks. The `all_sessions` arg opts out of
 * that scoping for the rare cross-conversation lookup. Pass an {@link Embedder}
 * to enable semantic recall (best-effort: an embedding outage degrades to
 * lexical, never failing a tool call).
 *
 * Mirrors {@link memoryTools}: the loop's own arg validation enforces `required`
 * (there is none here: every field is optional), and the handler defends the
 * rest, translating {@link EventError} (e.g. a non-finite time bound that slips
 * past `asNumber`) into a clean message the model can read.
 */
export function eventTools(
    store: EventStore,
    options: { sessionId?: string; embedder?: Embedder } = {},
): ToolDef[] {
    const { sessionId, embedder } = options;

    const recall: ToolDef = {
        name: "transcript_recall",
        description:
            "Search your own conversation transcript: the durable log of past " +
            "messages, tool calls, and tool results, which outlives the in-context " +
            "history. Use it to recall what happened earlier, whether a tool already " +
            "ran, or what was decided. Returns the most relevant first (by meaning, " +
            "then shared words, then recency). Omit `query` to list recent events; " +
            "filter by `kind` ('message' | 'tool_call' | 'tool_result' | …) or a " +
            "`since`/`until` time window (epoch ms).",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "What to look for. Omit to list recent events.",
                },
                kind: {
                    type: "string",
                    description:
                        "Only return events of this kind, e.g. 'message', " +
                        "'tool_call', 'tool_result'.",
                },
                since: {
                    type: "number",
                    description: "Only events at or after this time (epoch ms).",
                },
                until: {
                    type: "number",
                    description: "Only events at or before this time (epoch ms).",
                },
                all_sessions: {
                    type: "boolean",
                    description:
                        "Search across every conversation, not just this one. " +
                        "Defaults to false (your own transcript only).",
                },
                limit: {
                    type: "number",
                    description: "Max results (default 8).",
                },
            },
        },
        async run(args) {
            const a = asRecord(args);
            const query = typeof a.query === "string" ? a.query : "";
            const limit = asNumber(a.limit) ?? DEFAULT_TRANSCRIPT_LIMIT;
            // Scope to this session unless the caller explicitly widened it. A
            // tool with no configured sessionId (a bare log) is global anyway.
            const session = a.all_sessions === true ? undefined : sessionId;
            const opts: EventQuery = {
                limit,
                ...(session !== undefined ? { session } : {}),
                ...(typeof a.kind === "string" ? { kind: a.kind } : {}),
                ...(asNumber(a.since) !== undefined ? { since: asNumber(a.since) } : {}),
                ...(asNumber(a.until) !== undefined ? { until: asNumber(a.until) } : {}),
            };
            try {
                const hits = await recallEvents(store, embedder, query, opts);
                return { count: hits.length, events: hits.map(toView) };
            } catch (err) {
                // A bad time bound (or any store-validation failure) becomes a
                // readable result, not a thrown error the loop has to package.
                if (err instanceof EventError) return { count: 0, error: err.message };
                throw err;
            }
        },
    };

    return [recall];
}
