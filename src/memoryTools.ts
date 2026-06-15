/**
 * Bridges the {@link MemoryStore} into the agentic loop.
 *
 * Two ways the agent gets memory:
 *  - {@link memoryTools} builds `ToolDef`s the model can call to save, recall,
 *    and forget memories during a run.
 *  - {@link recallContext} pulls the most relevant memories up front so the
 *    harness can inject them into the system prompt: passive recall the model
 *    benefits from without having to ask.
 *
 * The tools speak plain JSON in and out: their `run` results are objects the
 * loop drops straight into a `tool_result` part, so they must stay
 * serializable (no class instances leaking through).
 */

import type { ToolDef } from "./types.ts";
import { Memory, MemoryError, MemoryStore, MAX_LIMIT } from "./memory.ts";
import { embedOne, EmbeddingError, type Embedder } from "./embeddings.ts";
import type { WorkingMind } from "./workingMind.ts";

/** How many memories auto-recall injects by default. */
export const DEFAULT_RECALL_LIMIT = 10;

/**
 * Cosine-similarity threshold at or above which a new memory is treated as a
 * near-duplicate of an existing one and the save is skipped (see
 * {@link findDuplicate}). Set deliberately high: 0.95 catches reworded restatements
 * of the same fact ("user likes dark mode" / "the user prefers dark mode") while
 * leaving genuinely distinct-but-related facts ("user likes dark mode" / "user
 * dislikes bright screens") as separate rows. Tunable per-tool-set via
 * {@link MemoryToolOptions.dedupeThreshold}; pass 1 to require an effectively
 * exact vector match, or a value > 1 to disable semantic dedup entirely.
 */
export const DEFAULT_DEDUPE_THRESHOLD = 0.95;

/** The serializable view of a memory we hand back to the model. */
export interface MemoryView {
    id: number;
    content: string;
    tags: string[];
    importance?: number;
    created: number;
    updated: number;
    /** Durable, resurfacing-earned salience (the stored value; see
     *  {@link Memory.strength}). Surfaced so a caller inspecting a memory can see
     *  how reinforced it is, distinct from the explicit `importance` dial. */
    strength: number;
}

function toView(m: Memory): MemoryView {
    return {
        id: m.id,
        content: m.content,
        tags: m.tags,
        importance: m.importance,
        created: m.created,
        updated: m.updated,
        strength: m.strength,
    };
}

/** Narrow an unknown args bag to a record without trusting its fields yet. */
function asRecord(args: unknown): Record<string, unknown> {
    return typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
}

/** Distinguish a bare {@link Embedder} (the old second arg) from a
 *  {@link MemoryToolOptions} bag, so `memoryTools` stays back-compatible. An
 *  Embedder is identified by its `embed` method. */
function isEmbedder(value: Embedder | MemoryToolOptions | undefined): value is Embedder {
    return typeof (value as Embedder | undefined)?.embed === "function";
}

/** Coerce an unknown to a string[] of tags, ignoring non-string entries. */
function asTags(value: unknown): string[] | undefined {
    if (value === undefined || value === null) return undefined;
    if (!Array.isArray(value)) return undefined;
    return value.filter((t): t is string => typeof t === "string");
}

/**
 * Embed a memory and store its vector, swallowing embedding failures. Returns
 * whether a vector was stored. We deliberately catch {@link EmbeddingError}
 * here: an embedding outage should degrade recall (back to lexical), never break
 * a save or a recall request.
 */
async function embedIfPossible(
    store: MemoryStore,
    embedder: Embedder | undefined,
    memory: Memory,
): Promise<boolean> {
    if (!embedder) return false;
    try {
        const vec = await embedOne(embedder, memory.content);
        return store.setEmbedding(memory.id, vec);
    } catch (err) {
        if (err instanceof EmbeddingError) return false;
        throw err;
    }
}

/**
 * The shared recall ranking, used by both the `memory_recall` tool and
 * {@link recallContext}. Order of preference:
 *   1. Semantic (cosine) match, when an embedder is configured and the query
 *      embeds successfully: finds memories that *mean* the same thing.
 *   2. Lexical (FTS/bm25) match: shared words.
 *   3. Importance/recency order: when there's no query or nothing matched.
 * Each step falls through to the next so recall is never worse than before
 * embeddings existed.
 */
async function recallMemories(
    store: MemoryStore,
    embedder: Embedder | undefined,
    query: string,
    opts: { limit: number; tags?: string[] },
): Promise<Memory[]> {
    const trimmed = query.trim();

    if (trimmed && embedder) {
        try {
            const vec = await embedOne(embedder, trimmed);
            const hits = store.semanticSearch(vec, opts).map((h) => h.memory);
            if (hits.length) return hits;
        } catch (err) {
            // Fall through to lexical recall on any embedding failure.
            if (!(err instanceof EmbeddingError)) throw err;
        }
    }

    if (trimmed) {
        const lexical = store.searchRelevant(trimmed, opts);
        if (lexical.length) return lexical;
        // searchRelevant is token-based; fall back to substring for queries it
        // can't tokenize, preserving the tool's original behavior.
        const substring = store.search(trimmed, opts);
        if (substring.length) return substring;
    }

    return store.all(opts);
}

/** An existing memory a save was found to duplicate, with the score that decided
 *  it (the cosine similarity, or 1 for an exact-content match on the no-embedder
 *  path) and which path caught it. `semantic_duplicate` is a meaning-match above
 *  the cosine threshold (a reworded restatement of the same fact); `exact_match`
 *  is a byte-identical (case/whitespace-folded) content match, the no-embedder
 *  floor and the semantic path's backstop. The save surfaces this `reason` so the
 *  Construct can tell "you already said this in other words" from "this is
 *  literally already saved" and decide whether to update the existing memory or
 *  force a separate save. */
type DuplicateReason = "semantic_duplicate" | "exact_match";

interface DuplicateHit {
    memory: Memory;
    similarity: number;
    reason: DuplicateReason;
}

/**
 * Find an existing memory the given content duplicates, so a save can skip
 * re-storing the same fact. Two paths, preferring the stronger signal:
 *
 *  1. Semantic: when an embedder is configured and the content embeds, the
 *     nearest stored memory by cosine similarity. If that score is at or above
 *     `threshold`, it's a duplicate. This catches a reworded restatement of the
 *     same fact, which an exact-string check never would.
 *  2. Exact-content fallback: with no embedder (or when embedding fails), a
 *     case-and-whitespace-insensitive exact match on the trimmed content. We
 *     deliberately do NOT treat a mere lexical (FTS) overlap as a duplicate here:
 *     shared words are far too weak a signal to silently drop a save on.
 *
 * Returns the duplicate (with its score) or undefined when the content is novel.
 * Best-effort throughout: an embedding outage degrades to the exact-content
 * check, never failing the save.
 */
async function findDuplicate(
    store: MemoryStore,
    embedder: Embedder | undefined,
    content: string,
    threshold: number,
): Promise<DuplicateHit | undefined> {
    const trimmed = content.trim();
    if (!trimmed) return undefined;

    if (embedder) {
        try {
            const vec = await embedOne(embedder, trimmed);
            const [top] = store.semanticSearch(vec, { limit: 1 });
            if (top && top.score >= threshold) {
                return { memory: top.memory, similarity: top.score, reason: "semantic_duplicate" };
            }
            // A confident semantic *non*-match still falls through to the exact
            // check below: a brand-new memory that happens to be byte-identical to
            // an un-embedded older one should still dedupe.
        } catch (err) {
            if (!(err instanceof EmbeddingError)) throw err;
            // Embedding outage: fall through to the exact-content check.
        }
    }

    // Exact-content match (case- and surrounding-whitespace-insensitive), the
    // floor that works with no embedder and as the semantic path's backstop.
    const folded = trimmed.toLowerCase();
    for (const m of store.search(trimmed, { limit: MAX_LIMIT })) {
        if (m.content.trim().toLowerCase() === folded) {
            return { memory: m, similarity: 1, reason: "exact_match" };
        }
    }
    return undefined;
}

/** Tuning knobs for {@link memoryTools}. */
export interface MemoryToolOptions {
    /** Embedder for semantic recall and semantic save-time dedup. Omit to keep
     *  recall lexical and dedup exact-content only. */
    embedder?: Embedder;
    /**
     * Cosine threshold above which `memory_save` treats a new memory as a
     * near-duplicate of an existing one and skips the insert. Defaults to
     * {@link DEFAULT_DEDUPE_THRESHOLD}. Pass a value > 1 to disable semantic dedup
     * (exact-content dedup still applies).
     */
    dedupeThreshold?: number;
    /**
     * The Construct's working mind. When given, an *explicit* `memory_recall`
     * feeds the memories it surfaced into the mind's warm-memory band the same
     * way passive auto-recall does (see {@link Session.buildSystem}), so a memory
     * the Construct deliberately reached for stays warm afterwards rather than
     * blinking out. Optional, same pattern as {@link embedder}: omit it and the
     * recall tool just returns its hits.
     */
    mind?: WorkingMind;
}

/**
 * Build the memory tool set over a given store. The loop's own arg validation
 * (`validateArgs`) enforces `required`; these handlers defend the rest and
 * translate {@link MemoryError} into a clean message the model can read rather
 * than letting it surface as an opaque thrown error.
 *
 * Pass an {@link Embedder} to enable semantic recall: saved memories are
 * embedded so `memory_recall` can match by *meaning* (cosine similarity), not
 * just shared words. Embedding is best-effort: if the embedding service fails,
 * the save still succeeds and recall transparently falls back to lexical (FTS)
 * then importance order, so the harness never loses a memory or a turn to an
 * embedding outage.
 *
 * `memory_save` also dedupes: a save whose content closely matches an existing
 * memory (semantically when embeddings are on, else exact-content) is skipped and
 * the existing memory returned, so the same fact saved across sessions doesn't
 * accumulate near-identical rows. `memory_update` revises a memory in place,
 * keeping its id (and therefore its provenance and any links pointing at it),
 * which forget-then-re-save would lose.
 *
 * Back-compat: the second argument may still be a bare {@link Embedder} (the
 * original signature) or the new {@link MemoryToolOptions} bag.
 */
export function memoryTools(
    store: MemoryStore,
    embedderOrOptions?: Embedder | MemoryToolOptions,
): ToolDef[] {
    const options: MemoryToolOptions = isEmbedder(embedderOrOptions)
        ? { embedder: embedderOrOptions }
        : (embedderOrOptions ?? {});
    const embedder = options.embedder;
    const dedupeThreshold = options.dedupeThreshold ?? DEFAULT_DEDUPE_THRESHOLD;
    const mind = options.mind;
    const save: ToolDef = {
        name: "memory_save",
        description:
            "Save a durable memory for future conversations. Use for stable facts, " +
            "preferences, and decisions worth remembering: not transient chatter. " +
            "A save that closely matches an existing memory is skipped and the " +
            "existing one returned (deduped); pass force:true to save it anyway, or " +
            "use memory_update to revise the existing memory in place.",
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
                force: {
                    type: "boolean",
                    description:
                        "Save even if a near-duplicate already exists (skips dedup). " +
                        "Default false.",
                },
            },
            required: ["content"],
        },
        async run(args) {
            const a = asRecord(args);
            try {
                const content = a.content as string;
                // Dedup before inserting: a near-identical fact is skipped and the
                // existing memory returned, so the same fact saved across sessions
                // doesn't pile up near-duplicate rows. `force` opts out.
                if (a.force !== true && typeof content === "string") {
                    const dup = await findDuplicate(store, embedder, content, dedupeThreshold);
                    if (dup) {
                        return {
                            saved: false,
                            deduped: true,
                            // Which check caught it: a meaning-match above the
                            // cosine threshold vs a byte-identical content match. Let
                            // the Construct decide between memory_update (revise the
                            // existing one) and force:true (save a distinct memory).
                            reason: dup.reason,
                            similarity: dup.similarity,
                            memory: toView(dup.memory),
                        };
                    }
                }
                const memory = new Memory({
                    content,
                    tags: asTags(a.tags),
                    importance: typeof a.importance === "number" ? a.importance : undefined,
                });
                store.save(memory);
                // Best-effort embedding: a failure here must not fail the save.
                // The memory is already persisted; it just won't be semantically
                // searchable until a later backfill embeds it.
                await embedIfPossible(store, embedder, memory);
                return { saved: true, memory: toView(memory) };
            } catch (err) {
                if (err instanceof MemoryError) return { saved: false, error: err.message };
                throw err;
            }
        },
    };

    const update: ToolDef = {
        name: "memory_update",
        description:
            "Revise an existing memory in place by its id, keeping the same id (and " +
            "so its provenance and any links pointing at it), which forgetting and " +
            "re-saving would lose. Only the fields you pass change; omit the rest. " +
            "Pass tags:[] to clear all tags. Editing `content` re-embeds the memory.",
        parameters: {
            type: "object",
            properties: {
                id: { type: "number", description: "The id of the memory to revise." },
                content: { type: "string", description: "New fact text." },
                tags: {
                    type: "array",
                    items: { type: "string" },
                    description: "Replacement tag list (pass [] to clear).",
                },
                importance: {
                    type: "number",
                    description: "New relevance score from 0 (low) to 1 (high).",
                },
            },
            required: ["id"],
        },
        async run(args) {
            const a = asRecord(args);
            if (typeof a.id !== "number" || !Number.isFinite(a.id)) {
                return { updated: false, error: "id must be a finite number" };
            }
            const tags = asTags(a.tags);
            const hasImportance = typeof a.importance === "number";
            if (a.content === undefined && tags === undefined && !hasImportance) {
                return {
                    updated: false,
                    error: "provide content, tags, and/or importance to update",
                };
            }
            try {
                const patch: Parameters<MemoryStore["update"]>[1] = {};
                if (typeof a.content === "string") patch.content = a.content;
                if (tags !== undefined) patch.tags = tags;
                if (hasImportance) patch.importance = a.importance as number;

                const before = store.get(a.id);
                const updated = store.update(a.id, patch);
                if (!updated) return { updated: false, error: `no memory with id ${a.id}` };

                // Editing content invalidates the old vector (the store's trigger
                // drops it); re-embed best-effort so the memory stays semantically
                // searchable. A metadata-only edit keeps its vector, so skip it.
                if (before && updated.content !== before.content) {
                    await embedIfPossible(store, embedder, updated);
                }
                return { updated: true, memory: toView(updated) };
            } catch (err) {
                if (err instanceof MemoryError) return { updated: false, error: err.message };
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
            const hits = await recallMemories(store, embedder, query, { limit, tags });
            // Reinforce every memory an *explicit* recall surfaced. Reaching for a
            // memory deliberately is a stronger relevance signal than passively
            // matching one during auto-recall (which buildSystem already
            // reinforces), so an explicit hit should strengthen at least as much.
            // Best-effort, mirroring buildSystem's discipline: a strength write is
            // an earned ranking signal, not load-bearing for the tool call, so a
            // failure (closed store, vanished row) is swallowed.
            for (const m of hits) {
                try {
                    store.reinforce(m.id);
                } catch {
                    // Swallowed exactly like the passive-recall reinforce loop.
                }
                // And feed it into the working mind's warm-memory band, the same
                // way passively-surfaced memories enter it, keyed by store id so a
                // memory recalled again refreshes its warmth rather than stacking.
                mind?.note("memory", m.content, `m${m.id}`);
            }
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

    return [save, update, recall, forget];
}

/** Options controlling which memories auto-recall surfaces. */
export interface RecallOptions {
    /** Max memories to inject. Defaults to {@link DEFAULT_RECALL_LIMIT}. */
    limit?: number;
    /**
     * The current turn's text. When given, memories are ranked by relevance to
     * it: semantically if an {@link embedder} is provided, otherwise lexically
     * (FTS/bm25): rather than by global importance, so recall surfaces what's
     * relevant to *this* turn. Falls back to importance order when omitted or
     * when nothing matches.
     */
    query?: string;
    /**
     * Embedder for semantic recall. When given alongside a `query`, recall ranks
     * by meaning (cosine similarity) and degrades to lexical/importance order if
     * embedding fails. Omit to keep purely lexical behavior.
     */
    embedder?: Embedder;
}

/**
 * Render the memories worth recalling as a system-prompt fragment, or `null`
 * when the store is empty. Callers append the returned text to their system
 * message so the model starts each run already aware of what it knows.
 *
 * Pass the user's current turn as `query` to get turn-relevant recall: with an
 * `embedder` it ranks by semantic similarity, otherwise by lexical match, and
 * only falls back to importance order when there's no query or nothing matches.
 *
 * This is async because semantic recall embeds the query (a network call). With
 * no embedder it still awaits but does no I/O beyond the synchronous store.
 *
 * Back-compat: also accepts a bare number as the second argument, meaning
 * `{ limit }` with no query: the original signature.
 */
export async function recallContext(
    store: MemoryStore,
    options: RecallOptions | number = {},
): Promise<string | null> {
    return (await recallContextDetailed(store, options)).text;
}

/**
 * Like {@link recallContext}, but also returns the {@link Memory} objects that
 * surfaced, not just the rendered fragment. The Session uses this to feed its
 * working mind's warm-memory band (a memory that surfaced is kept warm a while
 * after) without running recall twice: one query, both the prose for the system
 * prompt and the structured memories for the mind.
 *
 * `text` is exactly what {@link recallContext} returns (the fragment, or `null`
 * when nothing surfaced); `memories` is the same set the fragment was rendered
 * from, in the order they ranked.
 */
export async function recallContextDetailed(
    store: MemoryStore,
    options: RecallOptions | number = {},
): Promise<{ text: string | null; memories: Memory[] }> {
    const opts: RecallOptions = typeof options === "number" ? { limit: options } : options;
    const limit = opts.limit ?? DEFAULT_RECALL_LIMIT;
    const query = typeof opts.query === "string" ? opts.query : "";

    const memories = await recallMemories(store, opts.embedder, query, { limit });
    if (memories.length === 0) return { text: null, memories };

    const lines = memories.map((m) => {
        const tags = m.tags.length ? ` [${m.tags.join(", ")}]` : "";
        return `- (#${m.id})${tags} ${m.content}`;
    });
    return { text: `Relevant things you remember:\n${lines.join("\n")}`, memories };
}
