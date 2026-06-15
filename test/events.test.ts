/**
 * Tests for the append-only event log ({@link EventStore}).
 *
 * Like the memory suite, every store is backed by an in-memory SQLite database
 * (`:memory:`) so the suite never touches disk and never shares state, and `ts`
 * timestamps are injected rather than read from the clock wherever ordering
 * matters. A handful of tests need a real file (shared-migration, WAL); those
 * use a temp directory.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

import { EventStore, EventError, MAX_ATTACHMENT_BYTES } from "../src/events.ts";
import { Memory, MemoryStore, SCHEMA_VERSION, MAX_CONTENT_LENGTH } from "../src/memory.ts";

function withTempDir(fn: (dir: string) => void): void {
    const dir = mkdtempSync(join(tmpdir(), "eventstore-"));
    try {
        fn(dir);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

/** Fresh isolated store per test; caller closes it (or lets the process exit). */
function freshStore(): EventStore {
    return new EventStore(":memory:");
}

/** A tiny normalized 2-D vector, for deterministic similarity tests. */
function unit(x: number, y: number): Float32Array {
    const len = Math.hypot(x, y) || 1;
    return Float32Array.from([x / len, y / len]);
}

// ---------------------------------------------------------------------------
// Migration & shared schema
// ---------------------------------------------------------------------------

test("a fresh EventStore is migrated to SCHEMA_VERSION", () => {
    const store = freshStore();
    assert.equal(store.version, SCHEMA_VERSION);
    assert.ok(SCHEMA_VERSION >= 4); // memory's 3 migrations, the events one, the provenance overlay
    store.close();
});

test("MemoryStore then EventStore on one file share a single user_version", () => {
    withTempDir((dir) => {
        const path = join(dir, "shared.sqlite");
        // Open memory first: it runs the whole MIGRATIONS array, events table included.
        const mem = new MemoryStore(path);
        assert.equal(mem.version, SCHEMA_VERSION);
        mem.close();

        // Opening the event store over the same file is a clean no-op migrate.
        const events = new EventStore(path);
        assert.equal(events.version, SCHEMA_VERSION);
        events.append({ kind: "message", content: "hi" });
        assert.equal(events.count(), 1);
        events.close();

        // One user_version on disk, and both tables present.
        const raw = new DatabaseSync(path);
        const { user_version } = raw.prepare("PRAGMA user_version").get() as {
            user_version: number;
        };
        assert.equal(user_version, SCHEMA_VERSION);
        const tables = (
            raw
                .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .all() as Array<{ name: string }>
        ).map((r) => r.name);
        assert.ok(tables.includes("memory"));
        assert.ok(tables.includes("events"));
        raw.close();
    });
});

test("the response_cancelled partial index exists and covers only cancellation events", () => {
    withTempDir((dir) => {
        const path = join(dir, "cancel-idx.sqlite");
        const events = new EventStore(path);
        // A cancellation marker plus an ordinary message, so the partial index has
        // something to cover and something to exclude.
        events.append({ kind: "response_cancelled", content: "Response cancelled by user." });
        events.append({ kind: "message", role: "agent", content: "a normal reply" });
        events.close();

        const raw = new DatabaseSync(path);
        // The migration created the partial index (a named index with a WHERE
        // clause over the cancellation kind).
        const idx = raw
            .prepare(
                "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_events_cancelled'",
            )
            .get() as { sql: string } | undefined;
        assert.ok(idx, "the idx_events_cancelled partial index must exist");
        assert.match(idx!.sql, /WHERE kind = 'response_cancelled'/);

        // The query the index serves returns only the cancellation row.
        const cancelled = raw
            .prepare("SELECT content FROM events WHERE kind = 'response_cancelled'")
            .all() as Array<{ content: string }>;
        assert.equal(cancelled.length, 1);
        assert.equal(cancelled[0]!.content, "Response cancelled by user.");
        raw.close();
    });
});

test("EventStore first, then MemoryStore on the same file, both work", () => {
    withTempDir((dir) => {
        const path = join(dir, "events-first.sqlite");
        const events = new EventStore(path);
        events.append({ kind: "dream", content: "an idea" });
        events.close();

        // A MemoryStore opened AFTER the events migration must still work: the
        // shared-migration path is symmetric.
        const mem = new MemoryStore(path);
        assert.equal(mem.version, SCHEMA_VERSION);
        assert.doesNotThrow(() => mem.save(new Memory({ content: "m" })));
        mem.close();
    });
});

// ---------------------------------------------------------------------------
// append / get round-trip
// ---------------------------------------------------------------------------

test("append assigns an id and round-trips every field via get", () => {
    const store = freshStore();
    const ev = store.append({
        kind: "tool_call",
        content: "search the web",
        role: "agent",
        meta: { tool: "web_search", args: { q: "weather" }, n: 3 },
        session: "sess-1",
        correlation: "corr-42",
        ts: 1000,
    });
    assert.ok(ev.id > 0);
    assert.equal(ev.ts, 1000);

    const got = store.get(ev.id);
    assert.ok(got);
    assert.equal(got.kind, "tool_call");
    assert.equal(got.content, "search the web");
    assert.equal(got.role, "agent");
    assert.deepEqual(got.meta, { tool: "web_search", args: { q: "weather" }, n: 3 });
    assert.equal(got.session, "sess-1");
    assert.equal(got.correlation, "corr-42");
    assert.equal(got.ts, 1000);
    store.close();
});

test("append fills ts from the clock when omitted", () => {
    const store = freshStore();
    const before = Date.now();
    const ev = store.append({ kind: "message", content: "now" });
    const after = Date.now();
    assert.ok(ev.ts >= before && ev.ts <= after);
    store.close();
});

test("optional fields default to undefined on read", () => {
    const store = freshStore();
    const ev = store.append({ kind: "system", content: "boot" });
    const got = store.get(ev.id);
    assert.ok(got);
    assert.equal(got.role, undefined);
    assert.equal(got.meta, undefined);
    assert.equal(got.session, undefined);
    assert.equal(got.correlation, undefined);
    store.close();
});

test("get returns undefined for a missing id", () => {
    const store = freshStore();
    assert.equal(store.get(999), undefined);
    store.close();
});

// ---------------------------------------------------------------------------
// meta serialization
// ---------------------------------------------------------------------------

test("meta JSON round-trips, including nested structures and arrays", () => {
    const store = freshStore();
    const meta = { a: [1, 2, { b: "c" }], nested: { deep: { x: true } }, s: "str" };
    const ev = store.append({ kind: "tool_result", content: "done", meta });
    assert.deepEqual(store.get(ev.id)?.meta, meta);
    store.close();
});

test("unserializable meta (BigInt) throws EventError and inserts nothing", () => {
    const store = freshStore();
    assert.throws(() => store.append({ kind: "x", content: "y", meta: { n: 1n } }), EventError);
    assert.equal(store.count(), 0);
    store.close();
});

test("unserializable meta (circular object) throws EventError", () => {
    const store = freshStore();
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    assert.throws(() => store.append({ kind: "x", content: "y", meta: circular }), EventError);
    store.close();
});

test("corrupt meta on read degrades to undefined instead of throwing", () => {
    const store = freshStore();
    const ev = store.append({ kind: "x", content: "y", meta: { ok: true } });
    // Reach past the public API to simulate a legacy/corrupt row.
    // @ts-expect-error accessing private db for the corruption test
    store.db.prepare("UPDATE events SET meta = ? WHERE id = ?").run("{not json", ev.id);
    const got = store.get(ev.id);
    assert.ok(got);
    assert.equal(got.meta, undefined);
    // A query over the corrupt row must not throw either.
    assert.doesNotThrow(() => store.recent());
    store.close();
});

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

test("validation rejects empty / missing / non-string kind", () => {
    const store = freshStore();
    assert.throws(() => store.append({ kind: "", content: "c" }), EventError);
    assert.throws(() => store.append({ kind: "   ", content: "c" }), EventError);
    // @ts-expect-error deliberately wrong type
    assert.throws(() => store.append({ content: "c" }), EventError);
    // @ts-expect-error deliberately wrong type
    assert.throws(() => store.append({ kind: 5, content: "c" }), EventError);
    store.close();
});

test("validation rejects empty / missing / non-string / over-length content", () => {
    const store = freshStore();
    assert.throws(() => store.append({ kind: "k", content: "" }), EventError);
    assert.throws(() => store.append({ kind: "k", content: "   " }), EventError);
    // @ts-expect-error deliberately wrong type
    assert.throws(() => store.append({ kind: "k" }), EventError);
    // @ts-expect-error deliberately wrong type
    assert.throws(() => store.append({ kind: "k", content: 5 }), EventError);
    assert.throws(
        () => store.append({ kind: "k", content: "x".repeat(MAX_CONTENT_LENGTH + 1) }),
        EventError,
    );
    // Exactly at the ceiling is fine.
    assert.doesNotThrow(() => store.append({ kind: "k", content: "x".repeat(MAX_CONTENT_LENGTH) }));
    store.close();
});

test("validation rejects a non-string role/session/correlation", () => {
    const store = freshStore();
    // @ts-expect-error deliberately wrong type
    assert.throws(() => store.append({ kind: "k", content: "c", role: 1 }), EventError);
    // @ts-expect-error deliberately wrong type
    assert.throws(() => store.append({ kind: "k", content: "c", session: {} }), EventError);
    // @ts-expect-error deliberately wrong type
    assert.throws(() => store.append({ kind: "k", content: "c", correlation: [] }), EventError);
    store.close();
});

test("validation rejects a non-finite ts", () => {
    const store = freshStore();
    assert.throws(() => store.append({ kind: "k", content: "c", ts: NaN }), EventError);
    assert.throws(() => store.append({ kind: "k", content: "c", ts: Infinity }), EventError);
    // @ts-expect-error deliberately wrong type
    assert.throws(() => store.append({ kind: "k", content: "c", ts: "soon" }), EventError);
    store.close();
});

// ---------------------------------------------------------------------------
// appendMany: transactional
// ---------------------------------------------------------------------------

test("appendMany inserts a whole batch and returns events in order", () => {
    const store = freshStore();
    const evs = store.appendMany([
        { kind: "message", content: "first", ts: 1 },
        { kind: "message", content: "second", ts: 2 },
        { kind: "message", content: "third", ts: 3 },
    ]);
    assert.equal(evs.length, 3);
    assert.deepEqual(
        evs.map((e) => e.content),
        ["first", "second", "third"],
    );
    assert.ok(evs[0].id < evs[1].id && evs[1].id < evs[2].id);
    assert.equal(store.count(), 3);
    store.close();
});

test("appendMany is all-or-nothing: one bad input rolls the batch back", () => {
    const store = freshStore();
    assert.throws(
        () =>
            store.appendMany([
                { kind: "message", content: "ok-1" },
                { kind: "", content: "bad kind" }, // invalid
                { kind: "message", content: "ok-2" },
            ]),
        EventError,
    );
    // Nothing was inserted: the valid rows before the bad one rolled back too.
    assert.equal(store.count(), 0);
    store.close();
});

test("appendMany on an empty array is a no-op", () => {
    const store = freshStore();
    assert.deepEqual(store.appendMany([]), []);
    assert.equal(store.count(), 0);
    store.close();
});

test("a failed appendMany leaves the store usable for the next append", () => {
    const store = freshStore();
    assert.throws(() => store.appendMany([{ kind: "k", content: "" }]), EventError);
    // The rolled-back transaction must not have left the connection wedged.
    const ev = store.append({ kind: "k", content: "after" });
    assert.equal(store.count(), 1);
    assert.equal(store.get(ev.id)?.content, "after");
    store.close();
});

// ---------------------------------------------------------------------------
// recent: ordering, pagination, filters
// ---------------------------------------------------------------------------

test("recent returns events newest first, id breaking ts ties", () => {
    const store = freshStore();
    store.append({ kind: "message", content: "old", ts: 100 });
    const mid1 = store.append({ kind: "message", content: "tie-a", ts: 200 });
    const mid2 = store.append({ kind: "message", content: "tie-b", ts: 200 });
    store.append({ kind: "message", content: "new", ts: 300 });

    const ordered = store.recent().map((e) => e.content);
    // Same ts (200): later-inserted (higher id) comes first.
    assert.ok(mid2.id > mid1.id);
    assert.deepEqual(ordered, ["new", "tie-b", "tie-a", "old"]);
    store.close();
});

test("recent honors limit and offset", () => {
    const store = freshStore();
    for (let i = 0; i < 10; i++) store.append({ kind: "m", content: `e${i}`, ts: i });
    const page1 = store.recent({ limit: 3, offset: 0 }).map((e) => e.content);
    const page2 = store.recent({ limit: 3, offset: 3 }).map((e) => e.content);
    assert.deepEqual(page1, ["e9", "e8", "e7"]);
    assert.deepEqual(page2, ["e6", "e5", "e4"]);
    store.close();
});

test("recent filters by kind", () => {
    const store = freshStore();
    store.append({ kind: "message", content: "m", ts: 1 });
    store.append({ kind: "tool_call", content: "t", ts: 2 });
    store.append({ kind: "message", content: "m2", ts: 3 });
    const msgs = store.recent({ kind: "message" }).map((e) => e.content);
    assert.deepEqual(msgs, ["m2", "m"]);
    store.close();
});

test("recent filters by session", () => {
    const store = freshStore();
    store.append({ kind: "m", content: "a", session: "s1", ts: 1 });
    store.append({ kind: "m", content: "b", session: "s2", ts: 2 });
    store.append({ kind: "m", content: "c", session: "s1", ts: 3 });
    const s1 = store.recent({ session: "s1" }).map((e) => e.content);
    assert.deepEqual(s1, ["c", "a"]);
    store.close();
});

test("recent filters by since/until (inclusive window)", () => {
    const store = freshStore();
    for (let i = 1; i <= 5; i++) store.append({ kind: "m", content: `e${i}`, ts: i * 100 });
    const window = store.recent({ since: 200, until: 400 }).map((e) => e.content);
    assert.deepEqual(window, ["e4", "e3", "e2"]);
    // Filters compose.
    const k = store.recent({ since: 200, until: 400, kind: "m", limit: 1 }).map((e) => e.content);
    assert.deepEqual(k, ["e4"]);
    store.close();
});

test("recent rejects a non-finite since/until", () => {
    const store = freshStore();
    store.append({ kind: "m", content: "x" });
    assert.throws(() => store.recent({ since: NaN }), EventError);
    assert.throws(() => store.recent({ until: Infinity }), EventError);
    store.close();
});

// ---------------------------------------------------------------------------
// searchRelevant: FTS5 / bm25
// ---------------------------------------------------------------------------

test("searchRelevant matches via porter stemming (deploys finds deploy)", () => {
    const store = freshStore();
    store.append({ kind: "m", content: "deploys happen on fridays", ts: 1 });
    store.append({ kind: "m", content: "completely unrelated", ts: 2 });
    const hits = store.searchRelevant("how does the deploy work").map((e) => e.content);
    assert.deepEqual(hits, ["deploys happen on fridays"]);
    store.close();
});

test("searchRelevant OR-matches any shared token across a sentence query", () => {
    const store = freshStore();
    store.append({ kind: "m", content: "deploys happen on fridays", ts: 1 });
    store.append({ kind: "m", content: "the staging database is read-only", ts: 2 });
    store.append({ kind: "m", content: "nothing in common", ts: 3 });
    const hits = store
        .searchRelevant("remind me how the database deploy works")
        .map((e) => e.content)
        .sort();
    assert.deepEqual(hits, ["deploys happen on fridays", "the staging database is read-only"]);
    store.close();
});

test("searchRelevant returns nothing for a token-less query", () => {
    const store = freshStore();
    store.append({ kind: "m", content: "something" });
    assert.deepEqual(store.searchRelevant("   "), []);
    assert.deepEqual(store.searchRelevant("!!! ??? ..."), []);
    store.close();
});

test("searchRelevant treats FTS operators as literal terms, not syntax", () => {
    const store = freshStore();
    store.append({ kind: "m", content: "notes about oranges" });
    assert.doesNotThrow(() => store.searchRelevant('oranges AND "(*:'));
    const hits = store.searchRelevant("oranges NOT apples").map((e) => e.content);
    assert.deepEqual(hits, ["notes about oranges"]);
    store.close();
});

test("searchRelevant composes with kind/session filters and honors limit", () => {
    const store = freshStore();
    store.append({ kind: "message", content: "deploy the api", session: "s1", ts: 1 });
    store.append({ kind: "tool_call", content: "deploy the worker", session: "s1", ts: 2 });
    store.append({ kind: "message", content: "deploy the api", session: "s2", ts: 3 });

    const byKind = store.searchRelevant("deploy", { kind: "message" }).map((e) => e.content);
    assert.deepEqual(byKind.sort(), ["deploy the api", "deploy the api"]);

    const bySession = store.searchRelevant("deploy", { session: "s2" }).map((e) => e.content);
    assert.deepEqual(bySession, ["deploy the api"]);

    assert.equal(store.searchRelevant("deploy", { limit: 1 }).length, 1);
    store.close();
});

test("the event FTS index follows inserts (append is indexed for free)", () => {
    const store = freshStore();
    assert.deepEqual(store.searchRelevant("penguins"), []);
    store.append({ kind: "m", content: "a note about penguins" });
    assert.equal(store.searchRelevant("penguins")[0]?.content, "a note about penguins");
    store.close();
});

// ---------------------------------------------------------------------------
// semanticSearch: cosine ranking
// ---------------------------------------------------------------------------

test("semanticSearch ranks by cosine similarity, highest first", () => {
    const store = freshStore();
    const cat = store.append({ kind: "m", content: "cats are great pets" });
    const dog = store.append({ kind: "m", content: "dogs are loyal" });
    const car = store.append({ kind: "m", content: "the engine needs oil" });
    store.setEmbedding(cat.id, unit(1, 0));
    store.setEmbedding(dog.id, unit(0.9, 0.4));
    store.setEmbedding(car.id, unit(-1, 0));

    const hits = store.semanticSearch(unit(1, 0));
    assert.deepEqual(
        hits.map((h) => h.event.content),
        ["cats are great pets", "dogs are loyal", "the engine needs oil"],
    );
    assert.ok(Math.abs(hits[0].score - 1) < 1e-6);
    store.close();
});

test("events without an embedding are invisible to semanticSearch", () => {
    const store = freshStore();
    const a = store.append({ kind: "m", content: "embedded" });
    store.append({ kind: "m", content: "not embedded" });
    store.setEmbedding(a.id, unit(1, 0));
    const hits = store.semanticSearch(unit(1, 0));
    assert.deepEqual(
        hits.map((h) => h.event.content),
        ["embedded"],
    );
    store.close();
});

test("a dimension-mismatched stored vector scores 0 and ranks last", () => {
    const store = freshStore();
    const good = store.append({ kind: "m", content: "2d match" });
    const wrong = store.append({ kind: "m", content: "3d vector" });
    store.setEmbedding(good.id, unit(1, 0));
    store.setEmbedding(wrong.id, Float32Array.from([0.5, 0.5, 0.7])); // 3-D, mismatched

    const hits = store.semanticSearch(unit(1, 0));
    assert.equal(hits.length, 2);
    assert.equal(hits[0].event.content, "2d match");
    assert.ok(Math.abs(hits[0].score - 1) < 1e-6);
    // The mismatched-dimension vector contributes a 0 score (cosineSimilarity
    // returns 0 for unequal lengths), so it sorts to the bottom.
    assert.equal(hits[1].event.content, "3d vector");
    assert.equal(hits[1].score, 0);
    store.close();
});

test("semanticSearch honors limit, offset, and kind/session filters", () => {
    const store = freshStore();
    const a = store.append({ kind: "message", content: "alpha", ts: 1 });
    const b = store.append({ kind: "message", content: "beta", ts: 2 });
    const c = store.append({ kind: "tool_call", content: "gamma", ts: 3 });
    store.setEmbedding(a.id, unit(0.9, 0.4)); // near the query
    store.setEmbedding(b.id, unit(0.5, 0.9)); // a bit further
    store.setEmbedding(c.id, unit(1, 0)); // exact match, but a different kind

    // kind filter drops gamma even though it's the closest.
    const filtered = store.semanticSearch(unit(1, 0), { kind: "message" });
    assert.deepEqual(
        filtered.map((h) => h.event.content),
        ["alpha", "beta"],
    );

    // Unfiltered, gamma (exact match) ranks first; limit caps the result.
    assert.equal(store.semanticSearch(unit(1, 0), { limit: 1 })[0].event.content, "gamma");
    // Offset past the top hit returns the next best.
    const second = store.semanticSearch(unit(1, 0), { limit: 1, offset: 1 });
    assert.equal(second.length, 1);
    assert.equal(second[0].event.content, "alpha");
    store.close();
});

test("semanticSearch breaks score ties by recency", () => {
    const store = freshStore();
    const older = store.append({ kind: "m", content: "older", ts: 100 });
    const newer = store.append({ kind: "m", content: "newer", ts: 200 });
    // Identical vectors → identical score; newer ts must win the tiebreak.
    store.setEmbedding(older.id, unit(1, 0));
    store.setEmbedding(newer.id, unit(1, 0));
    const hits = store.semanticSearch(unit(1, 0));
    assert.deepEqual(
        hits.map((h) => h.event.content),
        ["newer", "older"],
    );
    store.close();
});

// ---------------------------------------------------------------------------
// Embeddings: set / has / delete / backfill
// ---------------------------------------------------------------------------

test("setEmbedding stores a vector; hasEmbedding reports it", () => {
    const store = freshStore();
    const a = store.append({ kind: "m", content: "x" });
    assert.equal(store.hasEmbedding(a.id), false);
    assert.equal(store.setEmbedding(a.id, unit(1, 0)), true);
    assert.equal(store.hasEmbedding(a.id), true);
    store.close();
});

test("setEmbedding refuses an orphan id and reports false", () => {
    const store = freshStore();
    assert.equal(store.setEmbedding(999, unit(1, 0)), false);
    store.close();
});

test("setEmbedding replaces an existing vector (upsert)", () => {
    const store = freshStore();
    const a = store.append({ kind: "m", content: "x" });
    store.setEmbedding(a.id, unit(1, 0));
    store.setEmbedding(a.id, unit(0, 1));
    const hits = store.semanticSearch(unit(0, 1));
    assert.ok(Math.abs(hits[0].score - 1) < 1e-6);
    store.close();
});

test("deleteEmbedding removes the vector and reports whether a row went", () => {
    const store = freshStore();
    const a = store.append({ kind: "m", content: "x" });
    store.setEmbedding(a.id, unit(1, 0));
    assert.equal(store.deleteEmbedding(a.id), true);
    assert.equal(store.hasEmbedding(a.id), false);
    assert.equal(store.deleteEmbedding(a.id), false); // already gone
    assert.deepEqual(store.semanticSearch(unit(1, 0)), []);
    store.close();
});

test("idsMissingEmbedding is the backfill work-list, newest first", () => {
    const store = freshStore();
    const a = store.append({ kind: "m", content: "a", ts: 1 });
    const b = store.append({ kind: "m", content: "b", ts: 2 });
    const c = store.append({ kind: "m", content: "c", ts: 3 });
    // All three are missing; newest first.
    assert.deepEqual(store.idsMissingEmbedding(), [c.id, b.id, a.id]);

    store.setEmbedding(b.id, unit(1, 0));
    // Only the still-unembedded ones remain, still newest first.
    assert.deepEqual(store.idsMissingEmbedding(), [c.id, a.id]);

    // The limit bounds the work-list.
    assert.deepEqual(store.idsMissingEmbedding(1), [c.id]);
    store.close();
});

test("append never embeds: a fresh event has no vector", () => {
    const store = freshStore();
    const a = store.append({ kind: "m", content: "the log is total, the index is selective" });
    assert.equal(store.hasEmbedding(a.id), false);
    assert.deepEqual(store.idsMissingEmbedding(), [a.id]);
    store.close();
});

// ---------------------------------------------------------------------------
// count
// ---------------------------------------------------------------------------

test("count reflects total and respects filters", () => {
    const store = freshStore();
    store.append({ kind: "message", content: "a", session: "s1", ts: 1 });
    store.append({ kind: "tool_call", content: "b", session: "s1", ts: 2 });
    store.append({ kind: "message", content: "c", session: "s2", ts: 3 });
    assert.equal(store.count(), 3);
    assert.equal(store.count({ kind: "message" }), 2);
    assert.equal(store.count({ session: "s1" }), 2);
    assert.equal(store.count({ kind: "message", session: "s2" }), 1);
    assert.equal(store.count({ since: 2 }), 2);
    store.close();
});

// ---------------------------------------------------------------------------
// Lifecycle: closed store, isolation
// ---------------------------------------------------------------------------

test("operations on a closed store throw EventError", () => {
    const store = freshStore();
    const a = store.append({ kind: "m", content: "x" });
    store.setEmbedding(a.id, unit(1, 0));
    store.close();
    assert.throws(() => store.append({ kind: "m", content: "y" }), EventError);
    assert.throws(() => store.appendMany([{ kind: "m", content: "y" }]), EventError);
    assert.throws(() => store.get(a.id), EventError);
    assert.throws(() => store.recent(), EventError);
    assert.throws(() => store.searchRelevant("x"), EventError);
    assert.throws(() => store.semanticSearch(unit(1, 0)), EventError);
    assert.throws(() => store.setEmbedding(a.id, unit(1, 0)), EventError);
    assert.throws(() => store.hasEmbedding(a.id), EventError);
    assert.throws(() => store.idsMissingEmbedding(), EventError);
    assert.throws(() => store.count(), EventError);
    // close is idempotent.
    assert.doesNotThrow(() => store.close());
});

test("separate stores do not share state", () => {
    const a = freshStore();
    const b = freshStore();
    a.append({ kind: "m", content: "only in a" });
    assert.equal(a.count(), 1);
    assert.equal(b.count(), 0);
    a.close();
    b.close();
});

// ---------------------------------------------------------------------------
// WAL & persistence
// ---------------------------------------------------------------------------

test("file-backed store enables WAL by default; :memory: never does", () => {
    withTempDir((dir) => {
        const file = new EventStore(join(dir, "wal.sqlite"));
        assert.equal(file.wal, true);
        file.close();
    });
    const mem = freshStore();
    assert.equal(mem.wal, false);
    mem.close();
});

test("events persist across reopen and checkpoint is repeatable", () => {
    withTempDir((dir) => {
        const path = join(dir, "persist.sqlite");
        const a = new EventStore(path);
        for (let i = 0; i < 20; i++) a.append({ kind: "m", content: `e${i}`, ts: i });
        assert.doesNotThrow(() => a.checkpoint());
        assert.doesNotThrow(() => a.checkpoint());
        a.close();

        const b = new EventStore(path);
        assert.equal(b.count(), 20);
        assert.equal(b.recent({ limit: 1 })[0]?.content, "e19");
        b.close();
    });
});

// ---------------------------------------------------------------------------
// Image attachments (selective side table)
// ---------------------------------------------------------------------------

test("appendAttachment round-trips bytes; attachmentsFor lists metadata only", () => {
    const store = freshStore();
    const ev = store.append({ kind: "message", role: "user", content: "look [image: a.png]" });
    const bytes = Uint8Array.from([1, 2, 3, 4, 5]);
    const id = store.appendAttachment(ev.id, {
        mediaType: "image/png",
        data: bytes,
        filename: "a.png",
    });

    // The list read carries metadata, never the bytes.
    const metas = store.attachmentsFor(ev.id);
    assert.equal(metas.length, 1);
    assert.equal(metas[0].id, id);
    assert.equal(metas[0].eventId, ev.id);
    assert.equal(metas[0].mediaType, "image/png");
    assert.equal(metas[0].filename, "a.png");
    assert.equal("data" in metas[0], false);

    // The bytes-bearing read returns the exact bytes back.
    const full = store.getAttachment(id);
    assert.ok(full);
    assert.deepEqual(Array.from(full!.data), [1, 2, 3, 4, 5]);
    assert.equal(full!.mediaType, "image/png");

    store.close();
});

test("attachmentsFor orders oldest-first and is empty for an event with none", () => {
    const store = freshStore();
    const ev = store.append({ kind: "message", role: "user", content: "two pics" });
    const first = store.appendAttachment(ev.id, {
        mediaType: "image/png",
        data: Uint8Array.from([1]),
    });
    const second = store.appendAttachment(ev.id, {
        mediaType: "image/jpeg",
        data: Uint8Array.from([2]),
    });
    assert.deepEqual(
        store.attachmentsFor(ev.id).map((a) => a.id),
        [first, second],
    );
    // An unrelated event has no attachments.
    const other = store.append({ kind: "message", role: "agent", content: "reply" });
    assert.deepEqual(store.attachmentsFor(other.id), []);
    store.close();
});

test("getAttachment returns undefined for an unknown id", () => {
    const store = freshStore();
    assert.equal(store.getAttachment(999), undefined);
    store.close();
});

test("appendAttachment rejects an unknown event id (no orphans)", () => {
    const store = freshStore();
    assert.throws(
        () => store.appendAttachment(404, { mediaType: "image/png", data: Uint8Array.from([1]) }),
        EventError,
    );
    store.close();
});

test("appendAttachment validates media type, empty data, and the byte cap", () => {
    const store = freshStore();
    const ev = store.append({ kind: "message", role: "user", content: "x" });
    assert.throws(
        () =>
            store.appendAttachment(ev.id, {
                // a type the provider/bridge doesn't accept
                mediaType: "image/gif" as never,
                data: Uint8Array.from([1]),
            }),
        EventError,
    );
    assert.throws(
        () => store.appendAttachment(ev.id, { mediaType: "image/png", data: new Uint8Array(0) }),
        EventError,
    );
    assert.throws(
        () =>
            store.appendAttachment(ev.id, {
                mediaType: "image/png",
                data: new Uint8Array(MAX_ATTACHMENT_BYTES + 1),
            }),
        EventError,
    );
    store.close();
});

test("attachments persist across reopen and cascade-delete with their event", () => {
    withTempDir((dir) => {
        const path = join(dir, "attach.sqlite");
        const a = new EventStore(path);
        const ev = a.append({ kind: "message", role: "user", content: "pic [image: p.png]" });
        const id = a.appendAttachment(ev.id, {
            mediaType: "image/png",
            data: Uint8Array.from([9, 8, 7]),
            filename: "p.png",
        });
        a.close();

        // Survives a reopen with bytes intact.
        const b = new EventStore(path);
        const got = b.getAttachment(id);
        assert.ok(got);
        assert.deepEqual(Array.from(got!.data), [9, 8, 7]);

        // Deleting the parent event cascades the attachment away (events are
        // append-only at the API; this exercises the FK cascade directly).
        b.close();
        const raw = new DatabaseSync(path);
        raw.exec("PRAGMA foreign_keys = ON");
        raw.prepare("DELETE FROM events WHERE id = ?").run(ev.id);
        const remaining = raw
            .prepare("SELECT COUNT(*) AS n FROM event_attachments WHERE id = ?")
            .get(id) as { n: number };
        assert.equal(remaining.n, 0);
        raw.close();
    });
});
