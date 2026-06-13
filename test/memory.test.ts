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

import {
    Memory,
    MemoryStore,
    MemoryError,
    MAX_CONTENT_LENGTH,
    MAX_TAGS,
    MAX_TAG_LENGTH,
    DEFAULT_LIMIT,
    MAX_LIMIT,
} from "../src/memory.ts";

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
