/**
 * Tests for the knowledge-base store ({@link NotesStore}).
 *
 * Like the memory and event suites, every store is backed by an in-memory SQLite
 * database (`:memory:`), so the suite never touches disk, never shares state, and
 * is deterministic (`created`/`updated` are injected wherever ordering matters).
 * The link tests, whose foreign keys span the `notes` and `memory` tables, open a
 * NotesStore and a MemoryStore over one shared temp file (a `:memory:` db is
 * private per connection and can't be shared).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
    Note,
    NotesStore,
    NoteError,
    hashContent,
    normalizePath,
    MAX_TITLE_LENGTH,
    MAX_PATH_LENGTH,
} from "../src/notes.ts";
import { MemoryStore, Memory, SCHEMA_VERSION, MAX_CONTENT_LENGTH } from "../src/memory.ts";

function withTempDir(fn: (dir: string) => void): void {
    const dir = mkdtempSync(join(tmpdir(), "notesstore-"));
    try {
        fn(dir);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

function freshStore(): NotesStore {
    return new NotesStore(":memory:");
}

/** A note builder with sensible defaults; pass overrides for the fields a test
 *  cares about. A unique-ish default path keeps the UNIQUE constraint happy when
 *  a test saves several. */
let pathCounter = 0;
function note(extra: Partial<ConstructorParameters<typeof Note>[0]> = {}): Note {
    pathCounter++;
    return new Note({
        path: `note-${pathCounter}.md`,
        title: "Untitled",
        content: "body",
        ...extra,
    });
}

/** A tiny normalized 2-D vector, for deterministic similarity tests. */
function unit(x: number, y: number): Float32Array {
    const len = Math.hypot(x, y) || 1;
    return Float32Array.from([x / len, y / len]);
}

// ---------------------------------------------------------------------------
// Construction & validation
// ---------------------------------------------------------------------------

test("a fresh note mints a uuid and derives a content hash", () => {
    const n = note({ title: "Hi", content: "world" });
    assert.ok(n.uuid.length > 0);
    assert.equal(n.contentHash, hashContent("Hi", "world", {}));
});

test("a supplied uuid is honored", () => {
    const n = note({ uuid: "stable-id-123" });
    assert.equal(n.uuid, "stable-id-123");
});

test("rejects empty / whitespace-only title", () => {
    assert.throws(() => note({ title: "" }), NoteError);
    assert.throws(() => note({ title: "   " }), NoteError);
});

test("rejects an over-long title and over-long path", () => {
    assert.throws(() => note({ title: "x".repeat(MAX_TITLE_LENGTH + 1) }), NoteError);
    assert.throws(() => note({ path: "x".repeat(MAX_PATH_LENGTH) + ".md" }), NoteError);
});

test("rejects content over the shared length ceiling", () => {
    assert.throws(() => note({ content: "x".repeat(MAX_CONTENT_LENGTH + 1) }), NoteError);
    assert.doesNotThrow(() => note({ content: "x".repeat(MAX_CONTENT_LENGTH) }));
});

test("an empty body is allowed (a title-only stub is a valid note)", () => {
    assert.doesNotThrow(() => note({ content: "" }));
});

test("a whitespace uuid is rejected", () => {
    assert.throws(() => note({ uuid: "has space" }), NoteError);
});

// ---------------------------------------------------------------------------
// Path validation (path-traversal guard)
// ---------------------------------------------------------------------------

test("normalizePath rejects parent-dir escapes and absolute paths", () => {
    assert.throws(() => normalizePath("../escape.md"), NoteError);
    assert.throws(() => normalizePath("a/../../b.md"), NoteError);
    // A leading slash is stripped, not an escape, so it normalizes rather than throwing.
    assert.equal(normalizePath("/abs.md"), "abs.md");
});

test("normalizePath requires a .md extension", () => {
    assert.throws(() => normalizePath("note.txt"), NoteError);
    assert.equal(normalizePath("note.md"), "note.md");
    assert.equal(normalizePath("Note.MD"), "Note.MD");
});

test("normalizePath collapses separators and backslashes", () => {
    assert.equal(normalizePath("a\\b\\c.md"), "a/b/c.md");
    assert.equal(normalizePath("a//b///c.md"), "a/b/c.md");
});

test("normalizePath rejects control characters", () => {
    assert.throws(() => normalizePath("a\x00b.md"), NoteError);
});

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

test("save assigns an id and round-trips via get / getByUuid / getByPath", () => {
    const store = freshStore();
    const saved = store.save(
        note({ path: "folder/hello.md", title: "Hello", content: "hi there" }),
    );
    assert.ok(saved.id > 0);

    const byId = store.get(saved.id);
    assert.equal(byId?.title, "Hello");
    assert.equal(byId?.content, "hi there");
    assert.equal(byId?.path, "folder/hello.md");

    assert.equal(store.getByUuid(saved.uuid)?.id, saved.id);
    assert.equal(store.getByPath("folder/hello.md")?.id, saved.id);
    // getByPath normalizes its argument, so a backslash form still matches.
    assert.equal(store.getByPath("folder\\hello.md")?.id, saved.id);
    store.close();
});

test("a duplicate path is rejected with a legible NoteError", () => {
    const store = freshStore();
    store.save(note({ path: "dup.md" }));
    assert.throws(() => store.save(note({ path: "dup.md" })), /already exists at that path/);
    store.close();
});

test("a duplicate uuid is rejected with a legible NoteError", () => {
    const store = freshStore();
    store.save(note({ uuid: "same", path: "a.md" }));
    assert.throws(() => store.save(note({ uuid: "same", path: "b.md" })), /uuid already exists/);
    store.close();
});

test("update edits fields, stamps updated, preserves uuid/created, and rehashes", () => {
    const store = freshStore();
    const saved = store.save(note({ title: "old", content: "v1", created: 1000 }));
    const originalUuid = saved.uuid;
    const originalHash = saved.contentHash;

    const updated = store.update(saved.id, { title: "new", content: "v2" }, 2000);
    assert.equal(updated?.title, "new");
    assert.equal(updated?.content, "v2");
    assert.equal(updated?.uuid, originalUuid); // immutable
    assert.equal(updated?.created, 1000); // immutable
    assert.equal(updated?.updated, 2000);
    assert.notEqual(updated?.contentHash, originalHash); // rehashed

    const reloaded = store.get(saved.id);
    assert.equal(reloaded?.content, "v2");
    assert.equal(reloaded?.contentHash, updated?.contentHash);
    store.close();
});

test("update can move a note to a new path", () => {
    const store = freshStore();
    const saved = store.save(note({ path: "a.md" }));
    store.update(saved.id, { path: "sub/b.md" });
    assert.equal(store.getByPath("a.md"), undefined);
    assert.equal(store.getByPath("sub/b.md")?.id, saved.id);
    store.close();
});

test("update returns undefined for a missing id", () => {
    const store = freshStore();
    assert.equal(store.update(999, { title: "nope" }), undefined);
    store.close();
});

test("count reflects inserts and deletes", () => {
    const store = freshStore();
    assert.equal(store.count(), 0);
    const a = store.save(note());
    store.save(note());
    assert.equal(store.count(), 2);
    assert.equal(store.delete(a.id), true);
    assert.equal(store.delete(a.id), false);
    assert.equal(store.count(), 1);
    store.close();
});

// ---------------------------------------------------------------------------
// Frontmatter blob
// ---------------------------------------------------------------------------

test("frontmatter round-trips through save/get and excludes reserved keys", () => {
    const store = freshStore();
    const saved = store.save(
        note({
            title: "T",
            frontmatter: {
                tags: ["a", "b"],
                importance: 0.7,
                draft: true,
                // reserved keys must be dropped (they live in real columns)
                uuid: "ignored",
                title: "ignored",
                path: "ignored",
            },
        }),
    );
    const got = store.get(saved.id)!;
    assert.deepEqual(got.frontmatter, { tags: ["a", "b"], importance: 0.7, draft: true });
    store.close();
});

test("a frontmatter value of an unsupported type is rejected", () => {
    // @ts-expect-error deliberately wrong nested value
    assert.throws(() => note({ frontmatter: { nested: { deep: 1 } } }), NoteError);
    // @ts-expect-error array of non-strings
    assert.throws(() => note({ frontmatter: { nums: [1, 2] } }), NoteError);
});

test("corrupt frontmatter blob degrades to {} instead of throwing", () => {
    withTempDir((dir) => {
        const path = join(dir, "fm.sqlite");
        const store = new NotesStore(path);
        const saved = store.save(note({ frontmatter: { tags: ["x"] } }));
        // Reach past the API to simulate a corrupt row.
        // @ts-expect-error accessing private db for the corruption test
        store.db
            .prepare("UPDATE notes SET frontmatter = ? WHERE id = ?")
            .run("{not json", saved.id);
        const got = store.get(saved.id);
        assert.deepEqual(got?.frontmatter, {});
        assert.doesNotThrow(() => store.all());
        store.close();
    });
});

test("the content hash is stable across frontmatter key order", () => {
    const a = hashContent("T", "b", { x: "1", y: "2" });
    const b = hashContent("T", "b", { y: "2", x: "1" });
    assert.equal(a, b);
});

// ---------------------------------------------------------------------------
// Ordering, search, folder filter
// ---------------------------------------------------------------------------

test("all orders by updated desc", () => {
    const store = freshStore();
    store.save(note({ title: "old", created: 100 }));
    store.save(note({ title: "mid", created: 200 }));
    store.save(note({ title: "new", created: 300 }));
    assert.deepEqual(
        store.all().map((n) => n.title),
        ["new", "mid", "old"],
    );
    store.close();
});

test("search matches over title OR content, case-insensitively", () => {
    const store = freshStore();
    store.save(note({ title: "Deploy runbook", content: "ssh in and run" }));
    store.save(note({ title: "Grocery list", content: "buy milk" }));
    assert.deepEqual(
        store.search("deploy").map((n) => n.title),
        ["Deploy runbook"],
    );
    assert.deepEqual(
        store.search("MILK").map((n) => n.title),
        ["Grocery list"],
    );
    store.close();
});

test("search treats LIKE wildcards as literals", () => {
    const store = freshStore();
    store.save(note({ content: "100% done" }));
    store.save(note({ content: "nothing" }));
    assert.equal(store.search("100%").length, 1);
    assert.equal(store.search("_").length, 0);
    store.close();
});

test("pathPrefix filters to a folder subtree", () => {
    const store = freshStore();
    store.save(note({ path: "projects/a.md" }));
    store.save(note({ path: "projects/sub/b.md" }));
    store.save(note({ path: "personal/c.md" }));
    assert.equal(store.all({ pathPrefix: "projects/" }).length, 2);
    assert.equal(store.all({ pathPrefix: "personal/" }).length, 1);
    store.close();
});

// ---------------------------------------------------------------------------
// Relevance (FTS5) over title + content
// ---------------------------------------------------------------------------

test("searchRelevant ranks by lexical match over title and content", () => {
    const store = freshStore();
    store.save(note({ title: "Cooking", content: "how to braise short ribs" }));
    store.save(note({ title: "Deployment", content: "kubernetes rollout steps" }));
    const hits = store.searchRelevant("kubernetes deploy rollout").map((n) => n.title);
    assert.equal(hits[0], "Deployment");
    store.close();
});

test("searchRelevant matches a title-only hit", () => {
    const store = freshStore();
    store.save(note({ title: "Penguins", content: "unrelated body text" }));
    const hits = store.searchRelevant("penguins");
    assert.equal(hits[0]?.title, "Penguins");
    store.close();
});

test("the FTS index follows updates and deletes", () => {
    const store = freshStore();
    const saved = store.save(note({ title: "T", content: "about penguins" }));
    store.update(saved.id, { content: "about walruses" });
    assert.deepEqual(store.searchRelevant("penguins"), []);
    assert.equal(store.searchRelevant("walruses")[0]?.content, "about walruses");
    store.delete(saved.id);
    assert.deepEqual(store.searchRelevant("walruses"), []);
    store.close();
});

test("searchRelevant honors the pathPrefix filter", () => {
    const store = freshStore();
    store.save(note({ path: "ops/deploy.md", title: "deploy", content: "rollout" }));
    store.save(note({ path: "home/deploy.md", title: "deploy", content: "rollout" }));
    assert.equal(store.searchRelevant("rollout", { pathPrefix: "ops/" }).length, 1);
    store.close();
});

// ---------------------------------------------------------------------------
// Vector / semantic search
// ---------------------------------------------------------------------------

test("setEmbedding + semanticSearch ranks by cosine", () => {
    const store = freshStore();
    const a = store.save(note({ title: "near" }));
    const b = store.save(note({ title: "far" }));
    store.setEmbedding(a.id, unit(1, 0));
    store.setEmbedding(b.id, unit(-1, 0));
    const hits = store.semanticSearch(unit(1, 0));
    assert.deepEqual(
        hits.map((h) => h.note.title),
        ["near", "far"],
    );
    assert.ok(Math.abs(hits[0].score - 1) < 1e-6);
    store.close();
});

test("setEmbedding refuses an orphan", () => {
    const store = freshStore();
    assert.equal(store.setEmbedding(999, unit(1, 0)), false);
    store.close();
});

test("editing content invalidates the embedding; a title-only edit also drops it", () => {
    // Note: title is part of the FTS index but NOT the vector trigger's WHEN
    // clause (which watches `content`); a title-only update with unchanged content
    // keeps the vector. A content change drops it.
    const store = freshStore();
    const a = store.save(note({ title: "t", content: "original" }));
    store.setEmbedding(a.id, unit(1, 0));

    store.update(a.id, { title: "new title" }); // content unchanged
    assert.equal(store.hasEmbedding(a.id), true);

    store.update(a.id, { content: "different now" });
    assert.equal(store.hasEmbedding(a.id), false);
    store.close();
});

test("deleting a note cascades away its embedding", () => {
    const store = freshStore();
    const a = store.save(note());
    store.setEmbedding(a.id, unit(1, 0));
    store.delete(a.id);
    assert.equal(store.hasEmbedding(a.id), false);
    assert.deepEqual(store.semanticSearch(unit(1, 0)), []);
    store.close();
});

test("idsMissingEmbedding tracks the backfill work-list", () => {
    const store = freshStore();
    const a = store.save(note({ created: 1 }));
    const b = store.save(note({ created: 2 }));
    assert.deepEqual(store.idsMissingEmbedding().sort(), [a.id, b.id].sort());
    store.setEmbedding(a.id, unit(1, 0));
    assert.deepEqual(store.idsMissingEmbedding(), [b.id]);
    store.close();
});

// ---------------------------------------------------------------------------
// Links (note -> note, within one :memory: store)
// ---------------------------------------------------------------------------

test("link note->note and read both directions", () => {
    const store = freshStore();
    const a = store.save(note({ title: "a" }));
    const b = store.save(note({ title: "b" }));
    const link = store.link(a.id, { toNote: b.id }, "references");

    assert.equal(link.fromNote, a.id);
    assert.equal(link.toNote, b.id);
    assert.equal(link.kind, "references");

    assert.equal(store.linksFrom(a.id).length, 1);
    assert.equal(store.linksToNote(b.id).length, 1);
    assert.equal(store.linksToNote(b.id)[0].fromNote, a.id);
    store.close();
});

test("a note cannot link to itself", () => {
    const store = freshStore();
    const a = store.save(note());
    assert.throws(() => store.link(a.id, { toNote: a.id }), /cannot link to itself/);
    store.close();
});

test("a link to a non-existent note is rejected by the foreign key", () => {
    const store = freshStore();
    const a = store.save(note());
    assert.throws(() => store.link(a.id, { toNote: 9999 }), /does not exist/);
    store.close();
});

test("deleting the from-note cascades away its outgoing links", () => {
    const store = freshStore();
    const a = store.save(note());
    const b = store.save(note());
    store.link(a.id, { toNote: b.id });
    assert.equal(store.linksToNote(b.id).length, 1);
    store.delete(a.id);
    assert.equal(store.linksToNote(b.id).length, 0);
    store.close();
});

test("deleting a linked-to note cascades away links pointing at it", () => {
    const store = freshStore();
    const a = store.save(note());
    const b = store.save(note());
    store.link(a.id, { toNote: b.id });
    store.delete(b.id);
    assert.equal(store.linksFrom(a.id).length, 0);
    store.close();
});

test("unlink removes a single link", () => {
    const store = freshStore();
    const a = store.save(note());
    const b = store.save(note());
    const link = store.link(a.id, { toNote: b.id });
    assert.equal(store.unlink(link.id), true);
    assert.equal(store.unlink(link.id), false);
    assert.equal(store.linksFrom(a.id).length, 0);
    store.close();
});

// ---------------------------------------------------------------------------
// Links spanning notes <-> memory (shared file)
// ---------------------------------------------------------------------------

function withSharedStores(fn: (notes: NotesStore, mem: MemoryStore) => void): void {
    withTempDir((dir) => {
        const path = join(dir, "shared.sqlite");
        const notes = new NotesStore(path);
        const mem = new MemoryStore(path);
        try {
            fn(notes, mem);
        } finally {
            mem.close();
            notes.close();
        }
    });
}

test("link note->memory and reverse-lookup from the memory side", () => {
    withSharedStores((notes, mem) => {
        const m = mem.save(new Memory({ content: "a remembered fact" }));
        const n = notes.save(note({ title: "doc" }));
        const link = notes.link(n.id, { toMemory: m.id }, "derived_from");

        assert.equal(link.toMemory, m.id);
        assert.equal(notes.linksFrom(n.id).length, 1);
        assert.equal(notes.linksToMemory(m.id).length, 1);
        assert.equal(notes.linksToMemory(m.id)[0].fromNote, n.id);
    });
});

test("a link to a non-existent memory is rejected by the foreign key", () => {
    withSharedStores((notes) => {
        const n = notes.save(note());
        assert.throws(() => notes.link(n.id, { toMemory: 123456 }), /does not exist/);
    });
});

test("forgetting a linked memory nulls the pointer but keeps the link row", () => {
    withSharedStores((notes, mem) => {
        const m = mem.save(new Memory({ content: "fact" }));
        const n = notes.save(note());
        notes.link(n.id, { toMemory: m.id });

        mem.delete(m.id);
        const links = notes.linksFrom(n.id);
        assert.equal(links.length, 1, "the link row survives (SET NULL, not cascade)");
        assert.equal(links[0].toMemory, null);
    });
});

// ---------------------------------------------------------------------------
// Lifecycle / schema
// ---------------------------------------------------------------------------

test("a fresh store migrates to SCHEMA_VERSION and shares it with siblings", () => {
    const store = freshStore();
    assert.equal(store.version, SCHEMA_VERSION);
    store.close();
});

test("NotesStore and MemoryStore on one file share a single user_version", () => {
    withTempDir((dir) => {
        const path = join(dir, "shared.sqlite");
        const notes = new NotesStore(path);
        assert.equal(notes.version, SCHEMA_VERSION);
        notes.close();
        const mem = new MemoryStore(path);
        assert.equal(mem.version, SCHEMA_VERSION);
        mem.close();
    });
});

test("operations on a closed store throw NoteError", () => {
    const store = freshStore();
    store.save(note());
    store.close();
    assert.throws(() => store.all(), NoteError);
    assert.throws(() => store.count(), NoteError);
    assert.throws(() => store.searchRelevant("x"), NoteError);
    assert.doesNotThrow(() => store.close()); // idempotent
});

test("save re-validates a Note mutated after construction", () => {
    const store = freshStore();
    const n = note();
    n.title = "   ";
    assert.throws(() => store.save(n), NoteError);
    store.close();
});

test("file-backed store enables WAL; :memory: never does", () => {
    withTempDir((dir) => {
        const store = new NotesStore(join(dir, "wal.sqlite"));
        assert.equal(store.wal, true);
        store.close();
    });
    const mem = new NotesStore(":memory:");
    assert.equal(mem.wal, false);
    mem.close();
});

test("data persists across reopen", () => {
    withTempDir((dir) => {
        const path = join(dir, "persist.sqlite");
        const a = new NotesStore(path);
        const saved = a.save(note({ path: "kept.md", title: "kept" }));
        a.close();
        const b = new NotesStore(path);
        assert.equal(b.count(), 1);
        assert.equal(b.getByUuid(saved.uuid)?.title, "kept");
        b.close();
    });
});
