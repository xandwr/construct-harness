/**
 * Tests for the two-way sync engine ({@link NotesService}).
 *
 * This is the riskiest subsystem in the KB, so it gets the hardest tests:
 *  - the unified write path converges DB and file,
 *  - the echo loop is suppressed (an outbound write does NOT bounce back as an
 *    inbound change),
 *  - a human-created file (no uuid) is adopted and its uuid is written back,
 *  - a rename on disk is recognized by uuid (not treated as create+delete),
 *  - a deletion removes the row,
 *  - a genuine concurrent edit is detected as a conflict and the loser is parked
 *    in a sidecar (both win-directions),
 *  - the live watcher actually fires.
 *
 * Most tests drive `reconcileFile` directly so they're deterministic; one
 * exercises the real recursive watcher with bounded polling. Each runs in a
 * fresh temp dir over a file-backed store, since the engine does real file I/O.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import {
    mkdtempSync,
    rmSync,
    existsSync,
    readFileSync,
    writeFileSync,
    readdirSync,
    mkdirSync,
} from "node:fs";
import { utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { NotesStore } from "../src/notes.ts";
import { NotesService } from "../src/notesService.ts";
import { parseNoteFile } from "../src/notesFile.ts";

interface Harness {
    dir: string;
    kb: string;
    store: NotesStore;
    service: NotesService;
    warnings: string[];
}

async function withService(fn: (h: Harness) => Promise<void>): Promise<void> {
    const dir = mkdtempSync(join(tmpdir(), "notessvc-"));
    const kb = join(dir, "kb");
    const store = new NotesStore(join(dir, "db.sqlite"));
    const warnings: string[] = [];
    const service = new NotesService({
        store,
        root: kb,
        debounceMs: 20,
        onWarn: (m) => warnings.push(m),
    });
    try {
        await service.ready();
        await fn({ dir, kb, store, service, warnings });
    } finally {
        service.close();
        store.close();
        rmSync(dir, { recursive: true, force: true });
    }
}

/** Read a note file's parsed form from the KB folder. */
function readFile(kb: string, rel: string) {
    return parseNoteFile(readFileSync(join(kb, rel), "utf8"));
}

/** Poll a predicate until true or a timeout, for the live-watcher test. */
async function waitFor(pred: () => boolean, timeoutMs = 2000): Promise<void> {
    const start = performance.now();
    while (!pred()) {
        if (performance.now() - start > timeoutMs) throw new Error("timed out waiting");
        await new Promise((r) => setTimeout(r, 10));
    }
}

// ---------------------------------------------------------------------------
// Unified write path (outbound)
// ---------------------------------------------------------------------------

test("create persists a row and writes its file", async () => {
    await withService(async ({ kb, store, service }) => {
        const { note, path } = await service.create({ title: "Hello", content: "world" });
        assert.ok(note.id > 0);
        assert.equal(path, "hello.md");
        assert.ok(existsSync(join(kb, "hello.md")));

        const onDisk = readFile(kb, "hello.md");
        assert.equal(onDisk.uuid, note.uuid);
        assert.equal(onDisk.title, "Hello");
        assert.equal(onDisk.body, "world");

        assert.equal(store.getByUuid(note.uuid)?.content, "world");
    });
});

test("create honors an explicit path", async () => {
    await withService(async ({ kb, service }) => {
        const { path } = await service.create({
            title: "T",
            content: "x",
            path: "projects/deep/note.md",
        });
        assert.equal(path, "projects/deep/note.md");
        assert.ok(existsSync(join(kb, "projects/deep/note.md")));
    });
});

test("update rewrites the file with new content", async () => {
    await withService(async ({ kb, service }) => {
        const { note } = await service.create({ title: "T", content: "v1" });
        await service.update(note.id, { content: "v2" });
        assert.equal(readFile(kb, note.path).body, "v2");
    });
});

test("update that changes the path moves the file atomically", async () => {
    await withService(async ({ kb, service }) => {
        const { note } = await service.create({ title: "T", content: "x", path: "a.md" });
        await service.update(note.id, { path: "sub/b.md" });
        assert.equal(existsSync(join(kb, "a.md")), false);
        assert.ok(existsSync(join(kb, "sub/b.md")));
        assert.equal(readFile(kb, "sub/b.md").uuid, note.uuid);
    });
});

test("remove deletes both the row and the file", async () => {
    await withService(async ({ kb, store, service }) => {
        const { note } = await service.create({ title: "T", content: "x" });
        assert.equal(await service.remove(note.id), true);
        assert.equal(store.get(note.id), undefined);
        assert.equal(existsSync(join(kb, note.path)), false);
        // Idempotent.
        assert.equal(await service.remove(note.id), false);
    });
});

// ---------------------------------------------------------------------------
// Echo-loop suppression (the classic two-way-sync bug)
// ---------------------------------------------------------------------------

test("an outbound write does not bounce back as an inbound change", async () => {
    await withService(async ({ store, service }) => {
        const { note } = await service.create({ title: "T", content: "x" });
        const updatedBefore = store.get(note.id)!.updated;

        // Reconcile the very file we just wrote: it must be recognized as our own
        // echo and produce no change (no new row, no re-update).
        const outcome = await service.reconcileFile(note.path);
        assert.equal(outcome.kind, "unchanged");
        assert.equal(store.count(), 1);
        assert.equal(store.get(note.id)!.updated, updatedBefore);
    });
});

// ---------------------------------------------------------------------------
// Inbound: adopting human-created files
// ---------------------------------------------------------------------------

test("a human-created file with no uuid is adopted and gets a uuid written back", async () => {
    await withService(async ({ kb, store, service }) => {
        // A person drops a plain markdown file into the KB folder.
        writeFileSync(join(kb, "idea.md"), "# An idea\n\njot jot");
        const outcome = await service.reconcileFile("idea.md");
        assert.equal(outcome.kind, "created");

        // The store now has it.
        const note = store.getByPath("idea.md");
        assert.ok(note);
        assert.equal(note!.content, "# An idea\n\njot jot");
        // Title falls back to the filename when the file has no frontmatter title.
        assert.equal(note!.title, "idea");

        // The uuid was written back into the file.
        const onDisk = readFile(kb, "idea.md");
        assert.equal(onDisk.uuid, note!.uuid);

        // And that write-back is itself an echo: reconciling again is a no-op.
        const again = await service.reconcileFile("idea.md");
        assert.equal(again.kind, "unchanged");
        assert.equal(store.count(), 1);
    });
});

test("a file whose frontmatter title is set uses it over the filename", async () => {
    await withService(async ({ kb, store, service }) => {
        writeFileSync(join(kb, "f.md"), "---\ntitle: Real Title\n---\nbody");
        await service.reconcileFile("f.md");
        assert.equal(store.getByPath("f.md")?.title, "Real Title");
    });
});

// ---------------------------------------------------------------------------
// Inbound: editing, rename, delete
// ---------------------------------------------------------------------------

test("editing a synced file applies the change to its row", async () => {
    await withService(async ({ kb, store, service }) => {
        const { note } = await service.create({ title: "T", content: "v1" });
        // Simulate an external editor save: rewrite the body, keep the uuid.
        const text = readFileSync(join(kb, note.path), "utf8").replace("v1", "v2-edited");
        writeFileSync(join(kb, note.path), text);

        const outcome = await service.reconcileFile(note.path);
        assert.equal(outcome.kind, "updated");
        assert.equal(store.get(note.id)?.content, "v2-edited");
    });
});

test("a rename on disk is recognized by uuid, not treated as create+delete", async () => {
    await withService(async ({ kb, store, service }) => {
        const { note } = await service.create({ title: "T", content: "x", path: "old.md" });
        const text = readFileSync(join(kb, "old.md"), "utf8");
        // Editor-style rename: new file with same uuid, old file gone.
        writeFileSync(join(kb, "new.md"), text);
        rmSync(join(kb, "old.md"));

        const outcome = await service.reconcileFile("new.md");
        assert.equal(outcome.kind, "moved");
        // Same row, new path; still exactly one note.
        assert.equal(store.count(), 1);
        assert.equal(store.get(note.id)?.path, "new.md");

        // Reconciling the vanished old path is a clean no-op (the row moved).
        const del = await service.reconcileFile("old.md");
        assert.equal(del.kind, "unchanged");
        assert.equal(store.count(), 1);
    });
});

test("deleting a file removes its row", async () => {
    await withService(async ({ kb, store, service }) => {
        const { note } = await service.create({ title: "T", content: "x" });
        rmSync(join(kb, note.path));
        const outcome = await service.reconcileFile(note.path);
        assert.equal(outcome.kind, "updated"); // delete applied
        assert.equal(store.get(note.id), undefined);
    });
});

// ---------------------------------------------------------------------------
// Conflict resolution
// ---------------------------------------------------------------------------

// A genuine concurrent edit means BOTH sides moved away from the last-agreed
// state independently. We reproduce it precisely: after create (DB and file
// agree at "base"), the agent mutates the row *directly through the store*
// (a DB-only change that did not flow to the file, exactly the race the plan
// describes), while the human's edit lands on the file. The next inbound
// reconcile sees both sides diverged from the snapshot.

test("a concurrent edit is a conflict; the newer file wins and the DB loser is parked", async () => {
    await withService(async ({ kb, store, service }) => {
        const { note } = await service.create({ title: "T", content: "base" });

        // Agent edits the row directly (DB-only divergence; file still says "base").
        store.update(note.id, { content: "db version" }, 1000);

        // Human edits the file; give it a clearly-newer mtime so the file wins LWW.
        const filePath = join(kb, note.path);
        writeFileSync(filePath, readFileSync(filePath, "utf8").replace("base", "human version"));
        const future = Date.now() / 1000 + 60;
        utimesSync(filePath, future, future);

        const outcome = await service.reconcileFile(note.path);
        assert.equal(outcome.kind, "conflict");

        // The file (human) version won: the row now holds it.
        assert.equal(store.get(note.id)?.content, "human version");

        // A sidecar with the DB's losing version exists.
        assert.ok("sidecar" in outcome && outcome.sidecar);
        const sidecar = (outcome as { sidecar: string }).sidecar;
        assert.ok(existsSync(join(kb, sidecar)), "sidecar file exists");
        assert.equal(readFile(kb, sidecar).body, "db version");
    });
});

test("when the DB is newer, the DB wins and the file's loser is parked", async () => {
    await withService(async ({ kb, store, service }) => {
        const { note } = await service.create({ title: "T", content: "base" });

        // Human edits the file with an OLD mtime (their edit predates the DB's).
        const filePath = join(kb, note.path);
        writeFileSync(filePath, readFileSync(filePath, "utf8").replace("base", "stale file edit"));
        const past = Date.now() / 1000 - 600;
        utimesSync(filePath, past, past);

        // Agent edits the row directly afterward, with a far-future `updated` so the
        // DB wins last-write-wins.
        store.update(note.id, { content: "fresh db edit" }, Date.now() + 600_000);

        const outcome = await service.reconcileFile(note.path);
        assert.equal(outcome.kind, "conflict");

        // DB version won: the row is unchanged and the file was overwritten with it.
        assert.equal(store.get(note.id)?.content, "fresh db edit");
        assert.equal(readFile(kb, note.path).body, "fresh db edit");

        // The file's losing edit is preserved in a sidecar.
        const sidecar = (outcome as { sidecar: string }).sidecar;
        assert.ok(existsSync(join(kb, sidecar)));
        assert.equal(readFile(kb, sidecar).body, "stale file edit");
    });
});

test("a sidecar file is not itself adopted as a new note", async () => {
    await withService(async ({ kb, store, service }) => {
        const { note } = await service.create({ title: "T", content: "base" });
        // Force a conflict to produce a sidecar.
        store.update(note.id, { content: "db version" }, 1000);
        const filePath = join(kb, note.path);
        writeFileSync(filePath, readFileSync(filePath, "utf8").replace("base", "human version"));
        const future = Date.now() / 1000 + 60;
        utimesSync(filePath, future, future);
        const result = await service.reconcileFile(note.path);
        assert.equal(result.kind, "conflict");

        // A scan over the folder (which now contains the note plus its sidecar)
        // must not turn the sidecar into a second note.
        const before = store.count();
        await service.scan();
        assert.equal(store.count(), before, "the sidecar was not adopted as a new note");
    });
});

// ---------------------------------------------------------------------------
// Scan & live watcher
// ---------------------------------------------------------------------------

test("scan adopts every pre-existing file in the folder", async () => {
    await withService(async ({ kb, store, service }) => {
        mkdirSync(join(kb, "sub"), { recursive: true });
        writeFileSync(join(kb, "a.md"), "alpha");
        writeFileSync(join(kb, "sub/b.md"), "beta");
        // A dotfile dir (e.g. .obsidian) must be ignored.
        mkdirSync(join(kb, ".obsidian"), { recursive: true });
        writeFileSync(join(kb, ".obsidian/config.md"), "ignore me");

        await service.scan();
        assert.equal(store.count(), 2);
        assert.ok(store.getByPath("a.md"));
        assert.ok(store.getByPath("sub/b.md"));
        assert.equal(store.getByPath(".obsidian/config.md"), undefined);
    });
});

test("the live watcher picks up a newly created file", async () => {
    await withService(async ({ kb, store, service }) => {
        await service.start();
        writeFileSync(join(kb, "watched.md"), "# watched\n\ncontent");
        await waitFor(() => store.getByPath("watched.md") !== undefined);
        assert.equal(store.getByPath("watched.md")?.content, "# watched\n\ncontent");
    });
});

test("the live watcher does not loop on the engine's own writes", async () => {
    await withService(async ({ store, service, warnings }) => {
        await service.start();
        const { note } = await service.create({ title: "T", content: "x" });
        // Give the watcher time to (not) react to our own write.
        await new Promise((r) => setTimeout(r, 200));
        assert.equal(store.count(), 1);
        assert.equal(store.get(note.id)?.content, "x");
        // No conflict warnings from a self-write.
        assert.equal(warnings.filter((w) => w.includes("conflict")).length, 0);
    });
});
