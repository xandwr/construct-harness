/**
 * Tests for memory gardening ({@link gardenMemories}, {@link gardenTools}).
 *
 * Gardening surfaces weak, idle, redundant memories as consolidation candidates —
 * and never deletes. These tests pin: the three-axis eligibility filter (low
 * importance AND low strength AND long idle), the semantic-duplicate pairing, the
 * "flag, don't forget" contract (a candidate becomes an event, the memory is
 * untouched), the no-embedder no-op, and the read-only garden_review tool.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { MemoryStore, Memory } from "../src/memory.ts";
import { EventStore } from "../src/events.ts";
import { EmbeddingError, type Embedder } from "../src/embeddings.ts";
import {
    gardenMemories,
    gardenTools,
    GARDEN_EVENT_KIND,
    GARDEN_MIN_IDLE_MS,
} from "../src/memoryGarden.ts";

/** A deterministic 2D embedder: maps texts to vectors via a table so cosine
 *  similarity is fully predictable, no network. Unknown texts get a "far" vector. */
function fakeEmbedder(
    table: Record<string, [number, number]>,
    opts: { fail?: boolean } = {},
): Embedder {
    const norm = ([x, y]: [number, number]): Float32Array => {
        const len = Math.hypot(x, y) || 1;
        return Float32Array.from([x / len, y / len]);
    };
    return {
        provider: "fake",
        model: "fake-2d",
        dimensions: 2,
        async embed(texts) {
            if (opts.fail) throw new EmbeddingError("embedding down");
            return texts.map((t) => norm(table[t] ?? [-1, -1]));
        },
    };
}

const NOW = 1_800_000_000_000;
/** A timestamp comfortably past the idle threshold. */
const LONG_AGO = NOW - GARDEN_MIN_IDLE_MS - 1;

/** Save a memory and set its embedding from the table, returning it. */
async function saveEmbedded(
    store: MemoryStore,
    embedder: Embedder,
    fields: { content: string; importance?: number; lastSurfaced?: number; strength?: number },
): Promise<Memory> {
    const m = new Memory({
        content: fields.content,
        importance: fields.importance,
        lastSurfaced: fields.lastSurfaced,
        strength: fields.strength,
        created: NOW - GARDEN_MIN_IDLE_MS - 10,
    });
    store.save(m);
    const [vec] = await embedder.embed([fields.content]);
    store.setEmbedding(m.id, vec!);
    return m;
}

test("a faded memory duplicating a stronger one is flagged as a consolidation candidate", async () => {
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    // Two memories pointing the same direction (cosine ~1): one strong & important
    // & recent (the keeper), one weak & unimportant & long-idle (the candidate).
    const embedder = fakeEmbedder({
        "user runs fish shell on linux": [1, 0],
        "the user uses fish because they set up omarchy": [0.99, 0.01],
    });
    const strong = await saveEmbedded(store, embedder, {
        content: "user runs fish shell on linux",
        importance: 0.9,
        lastSurfaced: NOW - 1000, // recent
        strength: 0.9,
    });
    const weak = await saveEmbedded(store, embedder, {
        content: "the user uses fish because they set up omarchy",
        importance: 0.1, // low importance
        lastSurfaced: LONG_AGO, // long idle
        strength: 0.1, // low strength
    });

    const pairs = await gardenMemories({ store, events, embedder, now: NOW });
    assert.equal(pairs.length, 1);
    assert.equal(pairs[0]!.weakId, weak.id);
    assert.equal(pairs[0]!.strongId, strong.id);
    assert.ok(pairs[0]!.similarity >= 0.85);

    // Flag-don't-forget: the weak memory is still there, untouched.
    assert.ok(store.get(weak.id), "gardening must not delete the weak memory");
    // And a garden event was written for the review tool to find.
    const flags = events.recent({ kind: GARDEN_EVENT_KIND });
    assert.equal(flags.length, 1);
});

test("a recent OR important OR strong memory is never gardened, even if duplicated", async () => {
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    const embedder = fakeEmbedder({ "topic a": [1, 0], "topic a restated": [0.99, 0.01] });
    // The would-be weak one is long-idle and low-strength but IMPORTANT: ineligible.
    await saveEmbedded(store, embedder, {
        content: "topic a",
        importance: 0.9,
        lastSurfaced: NOW - 1000,
        strength: 0.9,
    });
    await saveEmbedded(store, embedder, {
        content: "topic a restated",
        importance: 0.8, // above the 0.3 ceiling: protected
        lastSurfaced: LONG_AGO,
        strength: 0.1,
    });
    const pairs = await gardenMemories({ store, events, embedder, now: NOW });
    assert.equal(pairs.length, 0, "an important memory is never a gardening subject");
});

test("a faded memory with no semantic duplicate is left alone", async () => {
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    // Two faded memories, but pointing in orthogonal directions (cosine ~0).
    const embedder = fakeEmbedder({ "lonely fact one": [1, 0], "lonely fact two": [0, 1] });
    await saveEmbedded(store, embedder, {
        content: "lonely fact one",
        importance: 0.1,
        lastSurfaced: LONG_AGO,
        strength: 0.1,
    });
    await saveEmbedded(store, embedder, {
        content: "lonely fact two",
        importance: 0.1,
        lastSurfaced: LONG_AGO,
        strength: 0.1,
    });
    const pairs = await gardenMemories({ store, events, embedder, now: NOW });
    assert.equal(pairs.length, 0, "no consolidation target ⇒ no candidate");
});

test("without an embedder, gardening is a no-op (lexical overlap is too weak to retire on)", async () => {
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    const m = new Memory({
        content: "a faded duplicate",
        importance: 0.1,
        lastSurfaced: LONG_AGO,
        strength: 0.1,
        created: LONG_AGO,
    });
    store.save(m);
    const pairs = await gardenMemories({ store, events, now: NOW });
    assert.equal(pairs.length, 0);
});

test("garden_review lists flagged candidates and is read-only", async () => {
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    const embedder = fakeEmbedder({ keeper: [1, 0], "weak dup": [0.99, 0.01] });
    const strong = await saveEmbedded(store, embedder, {
        content: "keeper",
        importance: 0.9,
        lastSurfaced: NOW - 1000,
        strength: 0.9,
    });
    const weak = await saveEmbedded(store, embedder, {
        content: "weak dup",
        importance: 0.1,
        lastSurfaced: LONG_AGO,
        strength: 0.1,
    });
    await gardenMemories({ store, events, embedder, now: NOW });

    const [review] = gardenTools(events);
    const result = (await review!.run({})) as {
        count: number;
        candidates: Array<{ weakId: number; strongId: number }>;
    };
    assert.equal(result.count, 1);
    assert.equal(result.candidates[0]!.weakId, weak.id);
    assert.equal(result.candidates[0]!.strongId, strong.id);
    // The review only read: both memories still exist.
    assert.ok(store.get(weak.id) && store.get(strong.id));
});
