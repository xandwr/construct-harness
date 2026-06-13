/**
 * Tests for the hardened memory store ({@link MemoryStore}).
 *
 * Every store is backed by an in-memory SQLite database (`:memory:`), so the
 * suite never touches disk, never shares state between tests, and is fully
 * deterministic — `created` timestamps are injected rather than read from the
 * clock wherever ordering or staleness matters.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

import {
    Memory,
    MemoryStore,
    MemoryError,
    MigrationError,
    SCHEMA_VERSION,
    MAX_CONTENT_LENGTH,
    MAX_TAGS,
    MAX_TAG_LENGTH,
    DEFAULT_LIMIT,
    MAX_LIMIT,
} from "../src/memory.ts";

/**
 * Run `fn` with a path to a fresh temp directory, cleaned up afterward. Used by
 * the WAL tests, which need a real on-disk file (WAL is a no-op for `:memory:`).
 */
function withTempDir(fn: (dir: string) => void): void {
    const dir = mkdtempSync(join(tmpdir(), "memstore-"));
    try {
        fn(dir);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

/** Fresh isolated store per test; caller closes it (or lets the process exit). */
function freshStore(): MemoryStore {
    return new MemoryStore(":memory:");
}

function mem(
    content: string,
    extra: Partial<ConstructorParameters<typeof Memory>[0]> = {},
): Memory {
    return new Memory({ content, ...extra });
}

// ---------------------------------------------------------------------------
// Construction & validation
// ---------------------------------------------------------------------------

test("Memory normalizes tags: trims, dedupes, drops empties", () => {
    const m = mem("hi", { tags: [" a ", "a", "", "b", "  "] });
    assert.deepEqual(m.tags, ["a", "b"]);
});

test("rejects empty / whitespace-only content", () => {
    assert.throws(() => mem(""), MemoryError);
    assert.throws(() => mem("   "), MemoryError);
});

test("rejects non-string content", () => {
    // @ts-expect-error deliberately wrong type
    assert.throws(() => new Memory({ content: 123 }), MemoryError);
});

test("rejects content over the length ceiling", () => {
    assert.throws(() => mem("x".repeat(MAX_CONTENT_LENGTH + 1)), MemoryError);
    // exactly at the ceiling is fine
    assert.doesNotThrow(() => mem("x".repeat(MAX_CONTENT_LENGTH)));
});

test("rejects non-finite or out-of-range importance", () => {
    assert.throws(() => mem("hi", { importance: NaN }), MemoryError);
    assert.throws(() => mem("hi", { importance: Infinity }), MemoryError);
    assert.throws(() => mem("hi", { importance: -0.1 }), MemoryError);
    assert.throws(() => mem("hi", { importance: 1.1 }), MemoryError);
    // boundaries are valid
    assert.doesNotThrow(() => mem("hi", { importance: 0 }));
    assert.doesNotThrow(() => mem("hi", { importance: 1 }));
});

test("rejects too many or too-long tags", () => {
    const many = Array.from({ length: MAX_TAGS + 1 }, (_, i) => `t${i}`);
    assert.throws(() => mem("hi", { tags: many }), MemoryError);
    assert.throws(() => mem("hi", { tags: ["x".repeat(MAX_TAG_LENGTH + 1)] }), MemoryError);
});

test("rejects non-string tag entries", () => {
    // @ts-expect-error deliberately wrong type
    assert.throws(() => new Memory({ content: "hi", tags: [1, 2] }), MemoryError);
});

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

test("save assigns an id and round-trips via get", () => {
    const store = freshStore();
    const saved = store.save(mem("remember the milk", { tags: ["chore"], importance: 0.5 }));
    assert.ok(saved.id > 0);

    const got = store.get(saved.id);
    assert.ok(got);
    assert.equal(got.content, "remember the milk");
    assert.deepEqual(got.tags, ["chore"]);
    assert.equal(got.importance, 0.5);
    assert.equal(got.created, saved.created);
    store.close();
});

test("get returns undefined for a missing id", () => {
    const store = freshStore();
    assert.equal(store.get(999), undefined);
    store.close();
});

test("count reflects inserts and deletes", () => {
    const store = freshStore();
    assert.equal(store.count(), 0);
    const a = store.save(mem("a"));
    store.save(mem("b"));
    assert.equal(store.count(), 2);
    assert.equal(store.delete(a.id), true);
    assert.equal(store.delete(a.id), false); // already gone
    assert.equal(store.count(), 1);
    store.close();
});

test("clear empties the store and returns the count removed", () => {
    const store = freshStore();
    store.save(mem("a"));
    store.save(mem("b"));
    assert.equal(store.clear(), 2);
    assert.equal(store.count(), 0);
    assert.equal(store.clear(), 0);
    store.close();
});

test("update edits fields, stamps `updated`, and preserves `created`", () => {
    const store = freshStore();
    const saved = store.save(mem("old", { created: 1000, importance: 0.2 }));
    const updated = store.update(saved.id, { content: "new", importance: 0.9 }, 2000);
    assert.ok(updated);
    assert.equal(updated.content, "new");
    assert.equal(updated.importance, 0.9);
    assert.equal(updated.created, 1000); // immutable
    assert.equal(updated.updated, 2000);

    const reloaded = store.get(saved.id);
    assert.equal(reloaded?.content, "new");
    assert.equal(reloaded?.created, 1000);
    assert.equal(reloaded?.updated, 2000);
    store.close();
});

test("update can clear importance and revalidates the patch", () => {
    const store = freshStore();
    const saved = store.save(mem("x", { importance: 0.5 }));
    const cleared = store.update(saved.id, { importance: undefined });
    assert.equal(cleared?.importance, undefined);
    // a bad patch is rejected
    assert.throws(() => store.update(saved.id, { importance: 5 }), MemoryError);
    // empty content patch is rejected
    assert.throws(() => store.update(saved.id, { content: "  " }), MemoryError);
    store.close();
});

test("update returns undefined for a missing id", () => {
    const store = freshStore();
    assert.equal(store.update(123, { content: "nope" }), undefined);
    store.close();
});

// ---------------------------------------------------------------------------
// Ordering, limits, pagination
// ---------------------------------------------------------------------------

test("all orders by importance desc, then created desc, nulls last", () => {
    const store = freshStore();
    store.save(mem("low-old", { importance: 0.1, created: 100 }));
    store.save(mem("high", { importance: 0.9, created: 50 }));
    store.save(mem("none-new", { created: 999 }));
    store.save(mem("none-old", { created: 1 }));

    const ordered = store.all().map((m) => m.content);
    assert.deepEqual(ordered, ["high", "low-old", "none-new", "none-old"]);
    store.close();
});

test("all defaults to DEFAULT_LIMIT rows and caps at MAX_LIMIT", () => {
    const store = freshStore();
    for (let i = 0; i < DEFAULT_LIMIT + 5; i++) store.save(mem(`m${i}`, { created: i }));
    assert.equal(store.all().length, DEFAULT_LIMIT);
    // explicit over-cap is clamped, not honored
    assert.equal(store.all({ limit: MAX_LIMIT + 1000 }).length, store.count());
    // junk limits fall back to default
    assert.equal(store.all({ limit: -5 }).length, DEFAULT_LIMIT);
    assert.equal(store.all({ limit: NaN }).length, DEFAULT_LIMIT);
    store.close();
});

test("limit and offset paginate", () => {
    const store = freshStore();
    for (let i = 0; i < 10; i++) store.save(mem(`m${i}`, { importance: i / 10 }));
    const page1 = store.all({ limit: 3, offset: 0 }).map((m) => m.content);
    const page2 = store.all({ limit: 3, offset: 3 }).map((m) => m.content);
    assert.equal(page1.length, 3);
    assert.equal(page2.length, 3);
    assert.notDeepEqual(page1, page2);
    // highest importance first
    assert.equal(page1[0], "m9");
    store.close();
});

// ---------------------------------------------------------------------------
// Search & tag filtering
// ---------------------------------------------------------------------------

test("search does case-insensitive substring match over content", () => {
    const store = freshStore();
    store.save(mem("Buy Oat Milk"));
    store.save(mem("call the dentist"));
    const hits = store.search("milk").map((m) => m.content);
    assert.deepEqual(hits, ["Buy Oat Milk"]);
    store.close();
});

test("search treats LIKE wildcards as literals", () => {
    const store = freshStore();
    store.save(mem("100% done"));
    store.save(mem("nothing here"));
    // '%' must match literally, not as a wildcard
    assert.deepEqual(
        store.search("100%").map((m) => m.content),
        ["100% done"],
    );
    // a bare wildcard should not match everything
    assert.deepEqual(store.search("_").length, 0);
    store.close();
});

test("empty search behaves like all()", () => {
    const store = freshStore();
    store.save(mem("a"));
    store.save(mem("b"));
    assert.equal(store.search("   ").length, 2);
    store.close();
});

test("tag filter requires ALL given tags (AND) with exact-token match", () => {
    const store = freshStore();
    store.save(mem("both", { tags: ["work", "urgent"] }));
    store.save(mem("one", { tags: ["work"] }));
    store.save(mem("substring-trap", { tags: ["workshop"] }));

    const both = store.all({ tags: ["work", "urgent"] }).map((m) => m.content);
    assert.deepEqual(both, ["both"]);

    // "work" must not match "workshop"
    const work = store
        .all({ tags: ["work"] })
        .map((m) => m.content)
        .sort();
    assert.deepEqual(work, ["both", "one"]);
    store.close();
});

test("search and tag filter compose", () => {
    const store = freshStore();
    store.save(mem("milk for work", { tags: ["chore"] }));
    store.save(mem("milk at home", { tags: ["home"] }));
    const hits = store.search("milk", { tags: ["chore"] }).map((m) => m.content);
    assert.deepEqual(hits, ["milk for work"]);
    store.close();
});

// ---------------------------------------------------------------------------
// Relevance search (FTS5 / bm25)
// ---------------------------------------------------------------------------

test("searchRelevant ranks by lexical match, not importance", () => {
    const store = freshStore();
    // The dentist note is *more important* but unrelated to the query; the
    // oat-milk note is less important but a direct match and must rank first.
    store.save(mem("call the dentist about a filling", { importance: 0.9 }));
    store.save(mem("user likes oat milk in coffee", { importance: 0.1 }));

    // The milk note must rank first — bm25 relevance beats the dentist note's
    // higher importance. (Both may appear: stopword-ish tokens like "the" can
    // make the dentist note a weak match; what matters is that it ranks below.)
    const hits = store.searchRelevant("what milk does the user prefer?");
    assert.ok(hits.length >= 1);
    assert.equal(hits[0].content, "user likes oat milk in coffee");
    store.close();
});

test("searchRelevant matches any shared token (sentence query)", () => {
    const store = freshStore();
    store.save(mem("deploys happen on fridays"));
    store.save(mem("the staging database is read-only"));
    store.save(mem("completely unrelated"));

    const hits = store
        .searchRelevant("remind me how the database deploy works")
        .map((m) => m.content)
        .sort();
    assert.deepEqual(hits, ["deploys happen on fridays", "the staging database is read-only"]);
    store.close();
});

test("searchRelevant returns nothing for a token-less query", () => {
    const store = freshStore();
    store.save(mem("something"));
    assert.deepEqual(store.searchRelevant("   "), []);
    assert.deepEqual(store.searchRelevant("!!! ??? ..."), []);
    store.close();
});

test("searchRelevant treats FTS operators as literal terms, not syntax", () => {
    const store = freshStore();
    store.save(mem("notes about oranges"));
    // Bareword AND/OR/NOT and punctuation would be FTS operators if unescaped;
    // here they must be harmless and simply find the 'oranges' token.
    assert.doesNotThrow(() => store.searchRelevant('oranges AND "(*:'));
    const hits = store.searchRelevant("oranges NOT apples").map((m) => m.content);
    assert.deepEqual(hits, ["notes about oranges"]);
    store.close();
});

test("searchRelevant composes with tag filtering and honors limit", () => {
    const store = freshStore();
    store.save(mem("milk for work", { tags: ["chore"] }));
    store.save(mem("milk at home", { tags: ["home"] }));
    const hits = store.searchRelevant("milk", { tags: ["chore"] }).map((m) => m.content);
    assert.deepEqual(hits, ["milk for work"]);
    assert.equal(store.searchRelevant("milk", { limit: 1 }).length, 1);
    store.close();
});

test("the FTS index follows updates and deletes", () => {
    const store = freshStore();
    const saved = store.save(mem("originally about penguins"));
    // After an update, the old term stops matching and the new one starts.
    store.update(saved.id, { content: "now about walruses" });
    assert.deepEqual(store.searchRelevant("penguins"), []);
    assert.equal(store.searchRelevant("walruses")[0]?.content, "now about walruses");
    // After delete, nothing matches.
    store.delete(saved.id);
    assert.deepEqual(store.searchRelevant("walruses"), []);
    store.close();
});

// ---------------------------------------------------------------------------
// Vector / semantic search
// ---------------------------------------------------------------------------

/** A tiny normalized 2-D vector, for deterministic similarity tests. */
function unit(x: number, y: number): Float32Array {
    const len = Math.hypot(x, y) || 1;
    return Float32Array.from([x / len, y / len]);
}

test("setEmbedding stores a vector and semanticSearch ranks by cosine", () => {
    const store = freshStore();
    const cat = store.save(mem("cats are great pets"));
    const dog = store.save(mem("dogs are loyal companions"));
    const car = store.save(mem("the engine needs oil"));

    // Place the query near the cat vector, with the dog nearby and the car far.
    store.setEmbedding(cat.id, unit(1, 0));
    store.setEmbedding(dog.id, unit(0.9, 0.4));
    store.setEmbedding(car.id, unit(-1, 0));

    const hits = store.semanticSearch(unit(1, 0));
    assert.deepEqual(
        hits.map((h) => h.memory.content),
        ["cats are great pets", "dogs are loyal companions", "the engine needs oil"],
    );
    // Top hit is an exact direction match → score ~1.
    assert.ok(Math.abs(hits[0].score - 1) < 1e-6);
    store.close();
});

test("setEmbedding refuses an orphan and reports it", () => {
    const store = freshStore();
    assert.equal(store.setEmbedding(999, unit(1, 0)), false);
    store.close();
});

test("semanticSearch honors limit, offset, and tag filtering", () => {
    const store = freshStore();
    const a = store.save(mem("alpha", { tags: ["keep"] }));
    const b = store.save(mem("beta", { tags: ["keep"] }));
    const c = store.save(mem("gamma", { tags: ["skip"] }));
    store.setEmbedding(a.id, unit(0.9, 0.4)); // a near the query
    store.setEmbedding(b.id, unit(0.5, 0.9)); // b a bit further
    store.setEmbedding(c.id, unit(1, 0)); // exact match — would rank top, but filtered out

    // Tag filter drops gamma even though it's the closest; remaining ranked by similarity.
    const tagged = store.semanticSearch(unit(1, 0), { tags: ["keep"] });
    assert.deepEqual(
        tagged.map((h) => h.memory.content),
        ["alpha", "beta"],
    );

    // Unfiltered, gamma (exact match) ranks first; limit caps the result.
    assert.equal(store.semanticSearch(unit(1, 0), { limit: 1 })[0].memory.content, "gamma");
    // offset past the top hit returns the next-best, not gamma again.
    const second = store.semanticSearch(unit(1, 0), { limit: 1, offset: 1 });
    assert.equal(second.length, 1);
    assert.equal(second[0].memory.content, "alpha");
    store.close();
});

test("memories without an embedding are invisible to semanticSearch", () => {
    const store = freshStore();
    const a = store.save(mem("embedded"));
    store.save(mem("not embedded"));
    store.setEmbedding(a.id, unit(1, 0));
    const hits = store.semanticSearch(unit(1, 0));
    assert.deepEqual(
        hits.map((h) => h.memory.content),
        ["embedded"],
    );
    store.close();
});

test("hasEmbedding and idsMissingEmbedding track the backfill work-list", () => {
    const store = freshStore();
    const a = store.save(mem("a", { created: 1 }));
    const b = store.save(mem("b", { created: 2 }));
    assert.equal(store.hasEmbedding(a.id), false);
    assert.deepEqual(store.idsMissingEmbedding().sort(), [a.id, b.id].sort());

    store.setEmbedding(a.id, unit(1, 0));
    assert.equal(store.hasEmbedding(a.id), true);
    assert.deepEqual(store.idsMissingEmbedding(), [b.id]); // only the unembedded one
    store.close();
});

test("deleting a memory cascades away its embedding", () => {
    const store = freshStore();
    const a = store.save(mem("doomed"));
    store.setEmbedding(a.id, unit(1, 0));
    assert.equal(store.hasEmbedding(a.id), true);
    store.delete(a.id);
    assert.equal(store.hasEmbedding(a.id), false);
    assert.deepEqual(store.semanticSearch(unit(1, 0)), []);
    store.close();
});

test("editing content invalidates the stale embedding; metadata edits keep it", () => {
    const store = freshStore();
    const a = store.save(mem("original content", { importance: 0.3 }));
    store.setEmbedding(a.id, unit(1, 0));

    // A metadata-only edit must NOT drop the (still-valid) embedding.
    store.update(a.id, { importance: 0.9 });
    assert.equal(store.hasEmbedding(a.id), true);

    // A content edit invalidates it — the vector no longer matches the text.
    store.update(a.id, { content: "completely different now" });
    assert.equal(store.hasEmbedding(a.id), false);
    assert.deepEqual(store.idsMissingEmbedding(), [a.id]);
    store.close();
});

test("setEmbedding replaces an existing vector (upsert)", () => {
    const store = freshStore();
    const a = store.save(mem("x"));
    store.setEmbedding(a.id, unit(1, 0));
    store.setEmbedding(a.id, unit(0, 1));
    const hits = store.semanticSearch(unit(0, 1));
    assert.ok(Math.abs(hits[0].score - 1) < 1e-6);
    store.close();
});

test("vector methods throw on a closed store", () => {
    const store = freshStore();
    const a = store.save(mem("x"));
    store.setEmbedding(a.id, unit(1, 0));
    store.close();
    assert.throws(() => store.semanticSearch(unit(1, 0)), MemoryError);
    assert.throws(() => store.setEmbedding(a.id, unit(1, 0)), MemoryError);
    assert.throws(() => store.idsMissingEmbedding(), MemoryError);
});

// ---------------------------------------------------------------------------
// Robustness: corrupt rows, lifecycle, mutation-before-save
// ---------------------------------------------------------------------------

test("corrupt tags payload degrades to empty tags instead of throwing", () => {
    const store = freshStore();
    const saved = store.save(mem("x", { tags: ["a"] }));
    // Reach past the public API to simulate a legacy/corrupt row.
    // @ts-expect-error accessing private db for the corruption test
    store.db.prepare("UPDATE memory SET tags = ? WHERE id = ?").run("{not json", saved.id);
    const got = store.get(saved.id);
    assert.ok(got);
    assert.deepEqual(got.tags, []);
    // a query over the corrupt row must not throw either
    assert.doesNotThrow(() => store.all());
    store.close();
});

test("save re-validates a Memory mutated after construction", () => {
    const store = freshStore();
    const m = mem("ok");
    m.content = "   "; // sneak in bad content post-construction
    assert.throws(() => store.save(m), MemoryError);
    store.close();
});

test("operations on a closed store throw MemoryError", () => {
    const store = freshStore();
    store.save(mem("a"));
    store.close();
    assert.throws(() => store.all(), MemoryError);
    assert.throws(() => store.count(), MemoryError);
    assert.throws(() => store.save(mem("b")), MemoryError);
    assert.throws(() => store.searchRelevant("a"), MemoryError);
    // close is idempotent
    assert.doesNotThrow(() => store.close());
});

test("separate stores do not share state", () => {
    const a = freshStore();
    const b = freshStore();
    a.save(mem("only in a"));
    assert.equal(a.count(), 1);
    assert.equal(b.count(), 0);
    a.close();
    b.close();
});

// ---------------------------------------------------------------------------
// WAL mode & concurrency
// ---------------------------------------------------------------------------

test("file-backed store enables WAL by default", () => {
    withTempDir((dir) => {
        const store = new MemoryStore(join(dir, "wal.sqlite"));
        assert.equal(store.wal, true);
        store.close();
    });
});

test(":memory: store never uses WAL", () => {
    const store = new MemoryStore(":memory:");
    assert.equal(store.wal, false);
    store.close();
});

test("wal:false opts a file-backed store out of WAL", () => {
    withTempDir((dir) => {
        const store = new MemoryStore({ location: join(dir, "nowal.sqlite"), wal: false });
        assert.equal(store.wal, false);
        store.close();
    });
});

test("StoreOptions form configures location like the string form", () => {
    withTempDir((dir) => {
        const path = join(dir, "opts.sqlite");
        const a = new MemoryStore({ location: path });
        a.save(mem("persisted", { importance: 0.5 }));
        a.close();

        // Reopen the same file: data survived, so location was honored.
        const b = new MemoryStore(path);
        assert.equal(b.count(), 1);
        assert.equal(b.all()[0]?.content, "persisted");
        b.close();
    });
});

test("a concurrent reader sees a writer's committed rows under WAL", () => {
    withTempDir((dir) => {
        const path = join(dir, "concurrent.sqlite");
        const writer = new MemoryStore(path);
        const reader = new MemoryStore(path);
        assert.equal(writer.wal, true);
        assert.equal(reader.wal, true);

        writer.save(mem("from writer"));
        // Second connection reads the committed row without blocking.
        assert.equal(reader.count(), 1);
        assert.equal(reader.all()[0]?.content, "from writer");

        writer.close();
        reader.close();
    });
});

test("checkpoint folds the WAL back and is safe to call repeatedly", () => {
    withTempDir((dir) => {
        const path = join(dir, "ckpt.sqlite");
        const store = new MemoryStore(path);
        for (let i = 0; i < 50; i++) store.save(mem(`m${i}`));
        assert.doesNotThrow(() => store.checkpoint());
        assert.doesNotThrow(() => store.checkpoint());
        // data intact after checkpointing
        assert.equal(store.count(), 50);
        store.close();
    });
});

test("checkpoint is a no-op on a non-WAL store", () => {
    const store = new MemoryStore(":memory:");
    assert.doesNotThrow(() => store.checkpoint());
    store.close();
});

test("close checkpoints so no populated -wal sidecar is left behind", () => {
    withTempDir((dir) => {
        const path = join(dir, "sidecar.sqlite");
        const store = new MemoryStore(path);
        for (let i = 0; i < 20; i++) store.save(mem(`m${i}`));
        store.close();
        // TRUNCATE checkpoint on close leaves the -wal file empty (0 bytes) or
        // absent; either way it must not still hold committed pages.
        const walPath = `${path}-wal`;
        if (existsSync(walPath)) {
            assert.equal(statSync(walPath).size, 0);
        }
        // Reopening still sees every row, proving the data reached the main db.
        const reopened = new MemoryStore(path);
        assert.equal(reopened.count(), 20);
        reopened.close();
    });
});

test("checkpoint throws on a closed store", () => {
    withTempDir((dir) => {
        const store = new MemoryStore(join(dir, "closed.sqlite"));
        store.close();
        assert.throws(() => store.checkpoint(), MemoryError);
    });
});

// ---------------------------------------------------------------------------
// Schema migrations
// ---------------------------------------------------------------------------

test("a fresh store is migrated to SCHEMA_VERSION", () => {
    const store = new MemoryStore(":memory:");
    assert.equal(store.version, SCHEMA_VERSION);
    assert.ok(SCHEMA_VERSION >= 1);
    store.close();
});

test("opening writes user_version to the database file", () => {
    withTempDir((dir) => {
        const path = join(dir, "ver.sqlite");
        const store = new MemoryStore(path);
        store.close();

        const raw = new DatabaseSync(path);
        const { user_version } = raw.prepare("PRAGMA user_version").get() as {
            user_version: number;
        };
        raw.close();
        assert.equal(user_version, SCHEMA_VERSION);
    });
});

test("adopts a pre-versioning database without losing data", () => {
    withTempDir((dir) => {
        const path = join(dir, "legacy.sqlite");
        // Build a db the way the old code did: full table, but user_version still 0.
        const raw = new DatabaseSync(path);
        raw.exec(`
            CREATE TABLE memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created INTEGER NOT NULL,
                updated INTEGER NOT NULL,
                tags TEXT,
                importance REAL
            );
        `);
        raw.exec("INSERT INTO memory (content, created, updated) VALUES ('legacy', 1, 1)");
        const { user_version } = raw.prepare("PRAGMA user_version").get() as {
            user_version: number;
        };
        assert.equal(user_version, 0);
        raw.close();

        // Opening through the store adopts it: bumps version, keeps the row.
        const store = new MemoryStore(path);
        assert.equal(store.version, SCHEMA_VERSION);
        assert.equal(store.count(), 1);
        assert.equal(store.all()[0]?.content, "legacy");
        store.close();
    });
});

test("reopening an already-migrated store is a no-op at the same version", () => {
    withTempDir((dir) => {
        const path = join(dir, "again.sqlite");
        const a = new MemoryStore(path);
        a.save(mem("kept"));
        a.close();

        const b = new MemoryStore(path);
        assert.equal(b.version, SCHEMA_VERSION);
        assert.equal(b.count(), 1);
        b.close();
    });
});

test("refuses to open a database newer than the code supports", () => {
    withTempDir((dir) => {
        const path = join(dir, "future.sqlite");
        const raw = new DatabaseSync(path);
        raw.exec(`PRAGMA user_version = ${SCHEMA_VERSION + 100}`);
        raw.close();

        assert.throws(() => new MemoryStore(path), MigrationError);
    });
});
