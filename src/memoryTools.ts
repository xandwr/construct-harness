/**
 * Bridges the {@link MemoryStore} into the agentic loop.
 *
 * Two ways the agent gets memory:
 *  - {@link memoryTools} builds `ToolDef`s the model can call to save, recall,
 *    and forget memories during a run.
 *  - {@link recallContext} pulls the most relevant memories up front so the
 *    harness can inject them into the system prompt — passive recall the model
 *    benefits from without having to ask.
 *
 * The tools speak plain JSON in and out: their `run` results are objects the
 * loop drops straight into a `tool_result` part, so they must stay
 * serializable (no class instances leaking through).
 */

import type { ToolDef } from "./types.ts";
import { Memory, MemoryError, MemoryStore } from "./memory.ts";

/** How many memories auto-recall injects by default. */
export const DEFAULT_RECALL_LIMIT = 10;

/** The serializable view of a memory we hand back to the model. */
export interface MemoryView {
    id: number;
    content: string;
    tags: string[];
    importance?: number;
    created: number;
    updated: number;
}

function toView(m: Memory): MemoryView {
    return {
        id: m.id,
        content: m.content,
        tags: m.tags,
        importance: m.importance,
        created: m.created,
        updated: m.updated,
    };
}

/** Narrow an unknown args bag to a record without trusting its fields yet. */
function asRecord(args: unknown): Record<string, unknown> {
    return typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
}

/** Coerce an unknown to a string[] of tags, ignoring non-string entries. */
function asTags(value: unknown): string[] | undefined {
    if (value === undefined || value === null) return undefined;
    if (!Array.isArray(value)) return undefined;
    return value.filter((t): t is string => typeof t === "string");
}

/**
 * Build the memory tool set over a given store. The loop's own arg validation
 * (`validateArgs`) enforces `required`; these handlers defend the rest and
 * translate {@link MemoryError} into a clean message the model can read rather
 * than letting it surface as an opaque thrown error.
 */
export function memoryTools(store: MemoryStore): ToolDef[] {
    const save: ToolDef = {
        name: "memory_save",
        description:
            "Save a durable memory for future conversations. Use for stable facts, " +
            "preferences, and decisions worth remembering — not transient chatter.",
        parameters: {
            type: "object",
            properties: {
                content: { type: "string", description: "The fact to remember." },
                tags: {
                    type: "array",
                    items: { type: "string" },
                    description: "Optional labels for later filtering.",
                },
                importance: {
                    type: "number",
                    description: "Optional relevance score from 0 (low) to 1 (high).",
                },
            },
            required: ["content"],
        },
        async run(args) {
            const a = asRecord(args);
            try {
                const memory = new Memory({
                    content: a.content as string,
                    tags: asTags(a.tags),
                    importance: typeof a.importance === "number" ? a.importance : undefined,
                });
                store.save(memory);
                return { saved: true, memory: toView(memory) };
            } catch (err) {
                if (err instanceof MemoryError) return { saved: false, error: err.message };
                throw err;
            }
        },
    };

    const recall: ToolDef = {
        name: "memory_recall",
        description:
            "Search saved memories. Returns the most relevant first (importance, " +
            "then recency). Omit `query` to list recent memories; filter by `tags`.",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "Substring to match against memory content.",
                },
                tags: {
                    type: "array",
                    items: { type: "string" },
                    description: "Only return memories carrying all of these tags.",
                },
                limit: { type: "number", description: "Max results (default 10)." },
            },
        },
        async run(args) {
            const a = asRecord(args);
            const query = typeof a.query === "string" ? a.query : "";
            const limit = typeof a.limit === "number" ? a.limit : DEFAULT_RECALL_LIMIT;
            const tags = asTags(a.tags);
            const hits = query.trim()
                ? store.search(query, { limit, tags })
                : store.all({ limit, tags });
            return { count: hits.length, memories: hits.map(toView) };
        },
    };

    const forget: ToolDef = {
        name: "memory_forget",
        description: "Delete a memory by its id. Returns whether a row was removed.",
        parameters: {
            type: "object",
            properties: {
                id: { type: "number", description: "The id of the memory to delete." },
            },
            required: ["id"],
        },
        async run(args) {
            const a = asRecord(args);
            if (typeof a.id !== "number" || !Number.isFinite(a.id)) {
                return { forgotten: false, error: "id must be a finite number" };
            }
            return { forgotten: store.delete(a.id) };
        },
    };

    return [save, recall, forget];
}

/** Options controlling which memories auto-recall surfaces. */
export interface RecallOptions {
    /** Max memories to inject. Defaults to {@link DEFAULT_RECALL_LIMIT}. */
    limit?: number;
    /**
     * The current turn's text. When given, memories are ranked by lexical
     * relevance to it (FTS/bm25) rather than by global importance — so recall
     * surfaces what's relevant to *this* turn. Falls back to importance order
     * when omitted or when nothing matches.
     */
    query?: string;
}

/**
 * Render the memories worth recalling as a system-prompt fragment, or `null`
 * when the store is empty. Callers append the returned text to their system
 * message so the model starts each run already aware of what it knows.
 *
 * Pass the user's current turn as `query` to get turn-relevant recall: it ranks
 * by lexical match, and only falls back to importance order when there's no
 * query or no memory matches it.
 *
 * Back-compat: also accepts a bare number as the second argument, meaning
 * `{ limit }` with no query — the original signature.
 */
export function recallContext(
    store: MemoryStore,
    options: RecallOptions | number = {},
): string | null {
    const opts: RecallOptions = typeof options === "number" ? { limit: options } : options;
    const limit = opts.limit ?? DEFAULT_RECALL_LIMIT;
    const query = typeof opts.query === "string" ? opts.query.trim() : "";

    // Relevance first; fall back to importance order when the query is empty or
    // matches nothing, so recall is never worse than the old behavior.
    let memories = query ? store.searchRelevant(query, { limit }) : [];
    if (memories.length === 0) memories = store.all({ limit });
    if (memories.length === 0) return null;

    const lines = memories.map((m) => {
        const tags = m.tags.length ? ` [${m.tags.join(", ")}]` : "";
        return `- (#${m.id})${tags} ${m.content}`;
    });
    return `Relevant things you remember:\n${lines.join("\n")}`;
}
