/**
 * Memory gardening: surfacing the weak, idle, redundant memories worth
 * consolidating — without ever deciding to delete them.
 *
 * The memory store accumulates. Some of what it keeps turns out to be a worse,
 * weaker restatement of something it already holds better: two memories that say
 * nearly the same thing, one strong and surfacing often, one weak and untouched
 * for a month. Left alone they both rank, dilute recall, and the weak one is
 * mostly noise. Gardening finds those — but it does not pull them. The split is
 * the same one {@link ./salience.ts} draws for concerns: the *harness identifies*
 * candidates; the *Construct curates*.
 *
 * So a gardening pass:
 *  1. Pulls the genuinely-faded memories — below {@link GARDEN_MAX_IMPORTANCE}
 *     importance AND below {@link GARDEN_MAX_STRENGTH} effective (decayed-to-now)
 *     strength AND not surfaced in over {@link GARDEN_MIN_IDLE_MS}. A memory has
 *     to be weak on *all three* axes to be a gardening subject: explicitly low
 *     importance, low earned strength, and long idle. Anything the Construct
 *     marked important, keeps reinforcing, or touched recently is left alone.
 *  2. For each, runs one semantic search: does any *other* memory score above
 *     {@link GARDEN_SIMILARITY} against it? If so, that pair is a consolidation
 *     candidate — the weak one is probably a lesser version of the stronger one.
 *  3. Writes a {@link GARDEN_EVENT_KIND} event flagging the pair. It does NOT
 *     forget or merge anything. The event is the surfaced candidate; acting on it
 *     is the Construct's call, through the `garden_review` tool (see
 *     {@link gardenTools}), which lists the flagged pairs so the Construct can
 *     `memory_forget` the weak one or `memory_update` the strong one to absorb it.
 *
 * Needs an embedder (the consolidation test is semantic); with none it does
 * nothing rather than guess from lexical overlap, which is far too weak a signal
 * to flag a memory for retirement on. Best-effort throughout, like the rest of
 * the downtime work: an embedding outage or a store hiccup yields no candidates
 * rather than throwing.
 *
 * Speaks only core types, the {@link MemoryStore}/{@link EventStore} surfaces, and
 * the embedder: no model, no provider.
 */

import type { ToolDef } from "./types.ts";
import { MemoryStore, MAX_LIMIT } from "./memory.ts";
import type { Memory } from "./memory.ts";
import { EventStore } from "./events.ts";
import type { Event } from "./events.ts";
import { embedOne, EmbeddingError, type Embedder } from "./embeddings.ts";

/** The event `kind` a gardening candidate is flagged under, the gardening
 *  counterpart to the dream/goal kinds. A reader filters candidates with
 *  `recent({ kind: GARDEN_EVENT_KIND })`; the `garden_review` tool does exactly
 *  that. */
export const GARDEN_EVENT_KIND = "memory_garden";

/** Importance at or below which a memory is gardening-eligible. Below the
 *  midpoint: a memory the Construct dialed up in importance is never a gardening
 *  subject, however idle. */
export const GARDEN_MAX_IMPORTANCE = 0.3;

/** Effective (decayed-to-now) strength below which a memory is gardening-eligible.
 *  A memory that keeps resurfacing has high strength and is left alone; this only
 *  catches ones that have faded. */
export const GARDEN_MAX_STRENGTH = 0.4;

/** How long a memory must have gone unsurfaced to be gardening-eligible. Thirty
 *  days: a month of never coming up, on top of being unimportant and weak. */
export const GARDEN_MIN_IDLE_MS = 30 * 24 * 60 * 60 * 1000;

/** Cosine similarity above which a stronger memory is treated as the weak one's
 *  consolidation target — "this is a worse version of that". Conservative (0.85)
 *  so only genuinely-overlapping pairs are flagged, not merely related ones. */
export const GARDEN_SIMILARITY = 0.85;

/** How many memories to pull per pass before filtering to the faded ones. Set to
 *  the store's hard cap ({@link MAX_LIMIT}) on purpose: {@link MemoryStore.all}
 *  orders by importance DESC, so the low-importance memories gardening targets sit
 *  at the *tail* — a smaller window would scan past the very rows it wants. For a
 *  personal store (thousands of rows) this covers everything; a corpus larger than
 *  the cap would examine only a bounded window, the same linear-scan limitation
 *  the store's other reads carry (an indexed importance-ascending filter would be
 *  a future migration, not a change here). */
export const GARDEN_SCAN = MAX_LIMIT;

/** A flagged consolidation candidate: the weak memory and the stronger one it
 *  duplicates, with the similarity that paired them. The weak one is the
 *  retirement candidate; the strong one is the keeper (or the absorb target). */
export interface GardenPair {
    /** The faded memory (low importance, low strength, long idle): the candidate
     *  to forget or fold into {@link strong}. */
    weakId: number;
    /** A short snippet of the weak memory's content, for the review listing. */
    weakContent: string;
    /** The stronger memory it semantically duplicates: the keeper. */
    strongId: number;
    /** A short snippet of the strong memory's content. */
    strongContent: string;
    /** Cosine similarity between the two (≥ {@link GARDEN_SIMILARITY}). */
    similarity: number;
}

/** Trim content for the candidate listing so a flagged pair's event/meta stays
 *  compact. */
function snippet(text: string, cap = 200): string {
    const t = text.trim();
    return t.length > cap ? t.slice(0, cap) + "…" : t;
}

/** Options for {@link gardenMemories}. */
export interface GardenOptions {
    /** The corpus to garden. Required. */
    store: MemoryStore;
    /** The log to flag candidates on. Required. */
    events: EventStore;
    /** Embedder for the consolidation search. Omit and the pass is a no-op (the
     *  test is semantic; lexical overlap is too weak to retire a memory on). */
    embedder?: Embedder;
    /** Max weak memories to examine. Default {@link GARDEN_SCAN}. */
    scan?: number;
    /** "Now" for the idle/strength reckoning, epoch-ms. Injectable for tests. */
    now?: number;
}

/**
 * Run one gardening pass: find faded memories that duplicate a stronger one and
 * flag each pair as a {@link GARDEN_EVENT_KIND} event. Returns the pairs flagged
 * (also written to the log), newest-relevant first by similarity. Never deletes
 * or merges — surfacing the candidate is the whole job; the Construct decides.
 *
 * No-op (returns `[]`) without an embedder, or when nothing qualifies. Best-
 * effort: an embedding outage on a given memory skips that memory rather than
 * failing the pass; a flag-write failure is swallowed (the candidate is still
 * returned to the caller, just not persisted).
 */
export async function gardenMemories(options: GardenOptions): Promise<GardenPair[]> {
    const { store, events, embedder } = options;
    if (!embedder) return [];
    const now = options.now ?? Date.now();
    const scan = options.scan ?? GARDEN_SCAN;

    // Read a page and filter to the gardening-eligible ones in JS: the store has
    // no compound importance/strength/idle filter, and the eligible set is small
    // (weak + unimportant + idle), so a bounded scan is enough.
    let pool: Memory[];
    try {
        pool = store.all({ limit: Math.min(scan, MAX_LIMIT) });
    } catch {
        return [];
    }

    const weak = pool.filter((m) => isFaded(store, m, now));
    if (weak.length === 0) return [];

    const pairs: GardenPair[] = [];
    for (const m of weak) {
        // Embed the weak memory's content and find its nearest *other* memory.
        let vec: Float32Array;
        try {
            vec = await embedOne(embedder, m.content);
        } catch (err) {
            if (err instanceof EmbeddingError) continue; // skip this one, keep going
            throw err;
        }
        // Top 2: the nearest is the memory itself (if it's embedded), so look past
        // it for the strongest *other* memory above the threshold.
        const hits = store.semanticSearch(vec, { limit: 2, now });
        const other = hits.find((h) => h.memory.id !== m.id);
        if (!other || other.score < GARDEN_SIMILARITY) continue;
        // The other memory should be the *stronger* one for this to be a "worse
        // version of that"; if the weak one is somehow stronger, it isn't the
        // retirement candidate, so skip (gardening retires the lesser duplicate).
        const weakStrength = store.strengthOf(m.id, now) ?? m.strength;
        const otherStrength = store.strengthOf(other.memory.id, now) ?? other.memory.strength;
        if (otherStrength < weakStrength) continue;

        pairs.push({
            weakId: m.id,
            weakContent: snippet(m.content),
            strongId: other.memory.id,
            strongContent: snippet(other.memory.content),
            similarity: other.score,
        });
    }

    // Strongest overlap first.
    pairs.sort((a, b) => b.similarity - a.similarity);

    // Flag each as an event so the candidate survives the process and the
    // garden_review tool can list it. Best-effort: a write failure leaves the
    // candidate in the returned array but unlogged, rather than failing the pass.
    for (const p of pairs) {
        try {
            events.append({
                kind: GARDEN_EVENT_KIND,
                role: "system",
                content: `Consolidation candidate: memory #${p.weakId} duplicates #${p.strongId}.`,
                meta: {
                    weakId: p.weakId,
                    strongId: p.strongId,
                    similarity: p.similarity,
                    weakContent: p.weakContent,
                    strongContent: p.strongContent,
                },
            });
        } catch {
            // The log is an observer, never a gate: an unlogged candidate is still
            // returned to the caller.
        }
    }

    return pairs;
}

/** Whether a memory is gardening-eligible: weak on all three axes — low
 *  importance, low effective strength, long idle. Importance defaults to "low" so
 *  a memory with no importance set is eligible (it was never marked to keep). The
 *  effective strength is read decayed-to-now via the store. A memory that never
 *  surfaced (no lastSurfaced) is treated as fully idle. */
function isFaded(store: MemoryStore, m: Memory, now: number): boolean {
    const importance = m.importance ?? 0;
    if (importance > GARDEN_MAX_IMPORTANCE) return false;
    const strength = store.strengthOf(m.id, now) ?? m.strength;
    if (strength >= GARDEN_MAX_STRENGTH) return false;
    const idleMs = m.lastSurfaced === undefined ? Infinity : now - m.lastSurfaced;
    return idleMs > GARDEN_MIN_IDLE_MS;
}

// ── The review tool ───────────────────────────────────────────────────────────

/** Cap on how many flagged pairs `garden_review` returns at once. */
const DEFAULT_REVIEW_LIMIT = 20;

/** Narrow an unknown args bag to a record without trusting its fields yet. */
function asRecord(args: unknown): Record<string, unknown> {
    return typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
}

/** Flatten a {@link GARDEN_EVENT_KIND} event into the review shape, reading its
 *  meta defensively (a corrupt meta degrades to ids of 0, which the Construct can
 *  recognize as unusable rather than acting on). */
function gardenEventToView(e: Event) {
    const meta = (e.meta ?? {}) as Record<string, unknown>;
    const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : 0);
    const str = (v: unknown) => (typeof v === "string" ? v : "");
    return {
        flaggedAt: e.ts,
        weakId: num(meta.weakId),
        strongId: num(meta.strongId),
        similarity: num(meta.similarity),
        weakContent: str(meta.weakContent),
        strongContent: str(meta.strongContent),
    };
}

/**
 * Build the `garden_review` tool: lets the Construct review the consolidation
 * candidates gardening flagged, so it can act on them (the harness flags; the
 * Construct curates). It only *reads* the flags; acting is done with the existing
 * `memory_forget` (retire the weak duplicate) and `memory_update` (fold it into
 * the strong one) tools, which the Construct already has. Keeping the review
 * read-only and the action on the existing memory tools means gardening never
 * gains its own mutation path — the Construct curates with the same verbs it
 * always has.
 *
 * Read-only by contract like `transcript_recall`/`dream_recall`: a bad query
 * degrades to an empty result, never a thrown error past the loop.
 */
export function gardenTools(events: EventStore): ToolDef[] {
    const review: ToolDef = {
        name: "garden_review",
        description:
            "Review the memory-consolidation candidates flagged during downtime: " +
            "pairs where a weak, long-idle, low-importance memory looks like a worse " +
            "restatement of a stronger one you already hold. Each pair gives the weak " +
            "memory's id (the retirement candidate) and the strong one's id (the " +
            "keeper). Nothing is deleted for you — use memory_forget to drop the weak " +
            "one, or memory_update to fold its detail into the strong one, when you " +
            "agree. Returns the most recent flags first; omit `limit` for the default.",
        parameters: {
            type: "object",
            properties: {
                limit: { type: "number", description: "Max candidates (default 20)." },
            },
        },
        async run(args) {
            const a = asRecord(args);
            const limit = typeof a.limit === "number" ? a.limit : DEFAULT_REVIEW_LIMIT;
            try {
                const rows = events
                    .recent({ kind: GARDEN_EVENT_KIND, limit })
                    .map(gardenEventToView);
                return { count: rows.length, candidates: rows };
            } catch {
                return { count: 0, candidates: [] };
            }
        },
    };
    return [review];
}
