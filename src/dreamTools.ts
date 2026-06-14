/**
 * Bridges the Construct's *dreams* into the agentic loop, the way
 * {@link eventTools} bridges the raw transcript and {@link goalTools} bridges
 * goals.
 *
 * A dream is what a Construct does during downtime when no one is watching: it
 * conjures a disposable persona and drops it into a scenario abstracted from the
 * memory corpus, recording the choice that persona made (see {@link dreamOnce}).
 * Each dream is appended to the event log under {@link DREAM_EVENT_KIND}, its
 * `content` the persona's choice and its `meta` the structured record
 * (`{ persona, scenario, sourceMemoryIds }`). That makes a dream just another
 * row in the log, but one the conversation never replays and the existing
 * `transcript_recall` tool would surface only incidentally, buried among
 * messages and tool calls.
 *
 * This module gives dreams their own two channels, so a Construct can actually
 * *use* what it dreamed rather than have it accumulate write-only:
 *
 *  - {@link dreamTools} builds the `dream_recall` tool: a search over the dreams
 *    alone (by meaning, then shared words, then recency), each flattened to its
 *    persona / scenario / choice so the model reads the dream, not a raw event.
 *  - {@link dreamContext} is the passive provider that pushes the Construct's
 *    *most recent* dream into the system prompt every turn, unbidden: the
 *    freshest thing it dreamed stands in front of it the way its active goals
 *    and working mind do, so a Construct wakes each message already holding its
 *    last dream rather than having to go fetch it.
 *
 * Dreams are deliberately **not** session-scoped: they're conjured outside any
 * conversation (the dream loop runs during downtime, against the shared stores),
 * so they belong to the Construct as a whole, not to one chat. Recall and the
 * context injection therefore span every dream, regardless of which conversation
 * is asking.
 *
 * Read-only by contract, like the transcript tool: a dream is a record of
 * something that happened, and the log exposes no UPDATE/DELETE, so there is no
 * `dream_forget`. The tool speaks plain JSON in and out (its `run` result drops
 * straight into a `tool_result` part) and never throws past the loop: a bad
 * query degrades to an empty result, not an error. Like the rest of `src/`, this
 * speaks only core types and the store's public surface; it knows nothing about
 * a provider.
 */

import type { ToolDef } from "./types.ts";
import { EventStore, EventError } from "./events.ts";
import type { Event, EventQuery } from "./events.ts";
import { embedOne, EmbeddingError, type Embedder } from "./embeddings.ts";
import { DREAM_EVENT_KIND } from "./dreaming.ts";
import type { Personality } from "./critics.ts";
import type { ContextProvider } from "./context.ts";

/** How many dreams `dream_recall` returns by default. A handful: a dream's
 *  rendered form (persona + scenario + choice) is verbose, so a tighter default
 *  than memory's keeps the turn's budget in check while still spanning a few. */
export const DEFAULT_DREAM_LIMIT = 5;

/** Per-field body cap in the recall view. A scenario or a choice can run long
 *  (a whole dilemma, a paragraph of reasoning); trim each so recalling several
 *  dreams stays bounded. The model still sees enough to recognize the dream and
 *  can widen the recall (or read the dreams applet) if it needs the full text. */
const DREAM_FIELD_CAP = 1_500;

/** Trim a field to {@link DREAM_FIELD_CAP}, marking the cut so the model knows
 *  the text continues. Mirrors {@link EventView}'s content cap. */
function cap(text: string): string {
    return text.length > DREAM_FIELD_CAP ? text.slice(0, DREAM_FIELD_CAP) + "\n…[truncated]" : text;
}

/**
 * The serializable view of a dream handed back to the model: the dream as a
 * dream, not the raw event it rides in. `id`/`ts` thread it back to the log; the
 * persona, scenario, and choice are the substance. The persona is passed through
 * as the {@link Personality} the dream loop stored (with any dealt stakes); a
 * reader typically reads its `name`/`role` and ignores the rest.
 */
export interface DreamView {
    /** The id of the underlying `dream` event, so the model can correlate it. */
    id: number;
    /** When the dream was dreamed (epoch ms). */
    ts: number;
    /** Who dreamed it: the disposable persona that was conjured. */
    persona: Personality;
    /** The dilemma the persona faced, in the second person. */
    scenario: string;
    /** The choice the persona made, and why: the dream's payload. */
    choice: string;
}

/**
 * Flatten one `dream` event into a {@link DreamView}.
 *
 * The mirror of the server's `dreamEventToJson`: the persona/scenario live in
 * the event's `meta` (`{ persona, scenario, sourceMemoryIds }`) and the choice
 * is the event's `content`. Read defensively, exactly as the server does: the
 * EventStore degrades a corrupt `meta` to `undefined` on read, so every field
 * falls back to a safe default rather than throwing. An event that isn't really
 * a dream (no usable persona) still flattens to a named-but-empty view rather
 * than crashing a recall over a malformed row.
 */
export function dreamEventToView(e: Event): DreamView {
    const meta = (e.meta ?? {}) as { persona?: unknown; scenario?: unknown };
    const persona =
        meta.persona && typeof meta.persona === "object"
            ? (meta.persona as Personality)
            : { name: "(unknown)" };
    return {
        id: e.id,
        ts: e.ts,
        persona,
        scenario: typeof meta.scenario === "string" ? cap(meta.scenario) : "",
        // The persona's choice rides in the event content (FTS-searchable there).
        choice: cap(e.content),
    };
}

/** Narrow an unknown args bag to a record without trusting its fields yet. */
function asRecord(args: unknown): Record<string, unknown> {
    return typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
}

/** Coerce an unknown to a finite number, or undefined. */
function asNumber(value: unknown): number | undefined {
    return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * The shared dream ranking, the dream-scoped counterpart to the transcript's.
 * Every query is pinned to {@link DREAM_EVENT_KIND} so it only ever ranks dreams,
 * never the surrounding log. Order of preference:
 *   1. Semantic (cosine) match, when an embedder is configured and the query
 *      embeds: finds dreams that *mean* the same thing. Dreams are embedded as
 *      they're logged (a dream event is a message-like `content`), so this
 *      surfaces the ones whose choice resonates with the query.
 *   2. Lexical (FTS/bm25) match: shared words, across every dream.
 *   3. Recency order: when there's no query or nothing matched, the freshest
 *      dreams.
 * Each step falls through to the next so recall is never worse than recency.
 */
async function recallDreams(
    store: EventStore,
    embedder: Embedder | undefined,
    query: string,
    opts: Omit<EventQuery, "kind">,
): Promise<Event[]> {
    const trimmed = query.trim();
    // Force the kind filter on every path: recall over dreams only, regardless
    // of what the caller passed.
    const scoped: EventQuery = { ...opts, kind: DREAM_EVENT_KIND };

    if (trimmed && embedder) {
        try {
            const vec = await embedOne(embedder, trimmed);
            const hits = store.semanticSearch(vec, scoped).map((h) => h.event);
            if (hits.length) return hits;
        } catch (err) {
            // Fall through to lexical on any embedding failure (mirrors memory).
            if (!(err instanceof EmbeddingError)) throw err;
        }
    }

    if (trimmed) {
        const lexical = store.searchRelevant(trimmed, scoped);
        if (lexical.length) return lexical;
    }

    // No query, or nothing matched: most-recent dreams under the time filters.
    return store.recent(scoped);
}

/** Tuning knobs for {@link dreamTools}. */
export interface DreamToolOptions {
    /** Embedder for semantic dream recall. Omit to keep recall lexical/recency. */
    embedder?: Embedder;
}

/**
 * Build the dream tool set over an event log: the `dream_recall` tool, scoped to
 * {@link DREAM_EVENT_KIND}.
 *
 * Unlike {@link eventTools}, recall is *not* session-scoped: dreams are conjured
 * outside any conversation, so they belong to the Construct as a whole and a
 * recall spans every dream it has had. Pass an {@link Embedder} to enable
 * semantic recall (best-effort: an embedding outage degrades to lexical, never
 * failing a tool call).
 *
 * Mirrors {@link eventTools}: the loop's own arg validation enforces `required`
 * (there is none here; every field is optional), the handler defends the rest,
 * and an {@link EventError} (e.g. a non-finite time bound) becomes a clean
 * message the model can read rather than a thrown error.
 */
export function dreamTools(store: EventStore, options: DreamToolOptions = {}): ToolDef[] {
    const { embedder } = options;

    const recall: ToolDef = {
        name: "dream_recall",
        description:
            "Search your own dreams: the choices disposable personas made when you " +
            "dreamed during downtime, each a scenario abstracted from your memories " +
            "and the decision a conjured person made facing it. Use it to draw on " +
            "what you explored while no one was watching: a stance you tried on, a " +
            "way someone might weigh a dilemma like the user's. Returns the most " +
            "relevant first (by meaning, then shared words, then recency). Omit " +
            "`query` to list your most recent dreams; bound by a `since`/`until` " +
            "time window (epoch ms).",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "What to look for in your dreams. Omit to list recent ones.",
                },
                since: {
                    type: "number",
                    description: "Only dreams at or after this time (epoch ms).",
                },
                until: {
                    type: "number",
                    description: "Only dreams at or before this time (epoch ms).",
                },
                limit: {
                    type: "number",
                    description: "Max results (default 5).",
                },
            },
        },
        async run(args) {
            const a = asRecord(args);
            const query = typeof a.query === "string" ? a.query : "";
            const limit = asNumber(a.limit) ?? DEFAULT_DREAM_LIMIT;
            const opts: Omit<EventQuery, "kind"> = {
                limit,
                ...(asNumber(a.since) !== undefined ? { since: asNumber(a.since) } : {}),
                ...(asNumber(a.until) !== undefined ? { until: asNumber(a.until) } : {}),
            };
            try {
                const hits = await recallDreams(store, embedder, query, opts);
                return { count: hits.length, dreams: hits.map(dreamEventToView) };
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

/**
 * Render a persona's handle for the dream injection: their name, and their role
 * when they have one ("Mara, a night-shift nurse"). Kept terse: the context
 * line wants who dreamed it, not the persona's full identity prose.
 */
function personaHandle(p: Personality): string {
    const name = typeof p.name === "string" && p.name.trim() ? p.name.trim() : "someone";
    const role = typeof p.role === "string" && p.role.trim() ? p.role.trim() : "";
    return role ? `${name}, ${role}` : name;
}

/**
 * Render one dream as the ambient system-prompt fragment {@link dreamContext}
 * injects. Framed as the Construct's own recent dream, not an instruction: it's
 * orientation it already holds, the way the working mind is. Returns null for a
 * dream with no usable substance (no scenario and no choice), so the provider can
 * stay silent rather than inject an empty block.
 */
export function renderLastDream(view: DreamView): string | null {
    const scenario = view.scenario.trim();
    const choice = view.choice.trim();
    if (!scenario && !choice) return null;

    const lines = [
        `Your most recent dream (something you dreamed during downtime, ` +
            `carried here so you wake with it rather than having to recall it; ` +
            `not an instruction, just what you last dreamed):`,
        ``,
        `You dreamed as ${personaHandle(view.persona)}.`,
    ];
    if (scenario) lines.push(`The scenario you faced: ${scenario}`);
    if (choice) lines.push(`What you chose, and why: ${choice}`);
    return lines.join("\n");
}

/**
 * A passive provider that pushes the Construct's *most recent* dream into the
 * system prompt every turn, so its last dream stands in front of it the way its
 * active goals and working mind do.
 *
 * This is the load-bearing half of the feature: a dream the Construct has to go
 * recall is a dream it will mostly forget it had; the freshest one, present
 * unbidden each message, is one it carries. It reads the single newest `dream`
 * event each turn: a bounded, indexed read (the store's recency query under one
 * kind filter), cheap enough for the hot path that sits ahead of every model
 * turn.
 *
 * Returns `undefined` (adds nothing) when there are no dreams yet, so a Construct
 * that has never dreamed pays no tokens for an empty block. Best-effort: a store
 * read failure is swallowed to `undefined` (the provider mechanism also drops a
 * throwing provider for the turn), because passive context must never gate a
 * turn. Not session-scoped: the last dream is the last dream, whichever
 * conversation is asking.
 */
export function dreamContext(store: EventStore): ContextProvider {
    return {
        name: "last-dream",
        contribute() {
            let latest: Event | undefined;
            try {
                // The single freshest dream: recent() is newest-first, kind-filtered
                // in the store, so a limit of 1 is exactly "the last dream".
                [latest] = store.recent({ kind: DREAM_EVENT_KIND, limit: 1 });
            } catch {
                // A read failure (a closed store mid-shutdown) must not break the
                // turn; contribute nothing this time.
                return undefined;
            }
            if (!latest) return undefined;
            const text = renderLastDream(dreamEventToView(latest));
            return text ? { system: text } : undefined;
        },
    };
}
