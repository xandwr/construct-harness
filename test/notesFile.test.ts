/**
 * Tests for the file half of the knowledge base ({@link notesFile.ts}): the
 * note<->file serialization and the safe, atomic filesystem primitives.
 *
 * These do real I/O (the point is the file behavior), so each runs in a fresh
 * temp directory cleaned up afterward. The load-bearing properties: a note
 * round-trips through serialize/parse, a path can never escape the KB root, and
 * a write is atomic (no temp file left behind, no torn file observable).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import {
    mkdtempSync,
    rmSync,
    readdirSync,
    existsSync,
    readFileSync,
    symlinkSync,
    mkdirSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Note } from "../src/notes.ts";
import {
    serializeNote,
    parseNoteFile,
    resolveInRoot,
    writeNoteFileAtomic,
    moveNoteFile,
    deleteNoteFile,
    readNoteFile,
    ensureRoot,
    defaultPathForNote,
    NotesFileError,
} from "../src/notesFile.ts";

async function withTempDir(fn: (dir: string) => Promise<void>): Promise<void> {
    const dir = mkdtempSync(join(tmpdir(), "notesfile-"));
    try {
        await fn(dir);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

function note(extra: Partial<ConstructorParameters<typeof Note>[0]> = {}): Note {
    return new Note({ path: "n.md", title: "T", content: "body", ...extra });
}

// ---------------------------------------------------------------------------
// Serialization <-> parse
// ---------------------------------------------------------------------------

test("serializeNote writes uuid and title first, then the body", () => {
    const n = note({ uuid: "u-1", title: "Hello", content: "the body" });
    const text = serializeNote(n);
    assert.ok(text.startsWith("---\nuuid: u-1\ntitle: Hello\n"));
    assert.ok(text.includes("\nthe body\n"));
});

test("a note round-trips through serialize -> parseNoteFile", () => {
    const n = note({
        uuid: "u-2",
        title: "Runbook",
        content: "# Steps\n\n1. ssh\n2. deploy",
        frontmatter: { tags: ["ops", "deploy"], importance: 0.8 },
    });
    const parsed = parseNoteFile(serializeNote(n));
    assert.equal(parsed.uuid, "u-2");
    assert.equal(parsed.title, "Runbook");
    assert.deepEqual(parsed.frontmatter, { tags: ["ops", "deploy"], importance: 0.8 });
    assert.equal(parsed.body, "# Steps\n\n1. ssh\n2. deploy");
});

test("parseNoteFile on a file with no frontmatter yields no uuid", () => {
    const parsed = parseNoteFile("# Just a heading\n\nsome text");
    assert.equal(parsed.uuid, undefined);
    assert.equal(parsed.title, undefined);
    assert.deepEqual(parsed.frontmatter, {});
    assert.equal(parsed.body, "# Just a heading\n\nsome text");
});

test("parseNoteFile keeps human frontmatter but pulls out the modeled keys", () => {
    const text = ["---", "uuid: u-3", "title: T", "author: jane", "tags: [x]", "---", "body"].join(
        "\n",
    );
    const parsed = parseNoteFile(text);
    assert.equal(parsed.uuid, "u-3");
    assert.deepEqual(parsed.frontmatter, { author: "jane", tags: ["x"] });
});

// ---------------------------------------------------------------------------
// Path safety
// ---------------------------------------------------------------------------

test("resolveInRoot keeps paths inside the root", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        const abs = resolveInRoot(root, "sub/note.md");
        assert.ok(abs.startsWith(root));
        assert.ok(abs.endsWith(join("sub", "note.md")));
    });
});

test("resolveInRoot rejects a traversal attempt", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        // normalizePath rejects `..` first (NoteError); both are caught upstream
        // as "this path is not allowed". Either way it must throw, never resolve.
        assert.throws(() => resolveInRoot(root, "../escape.md"));
        assert.throws(() => resolveInRoot(root, "a/../../escape.md"));
    });
});

test("ensureRoot resolves symlinks so the root is the real directory", async () => {
    await withTempDir(async (dir) => {
        const real = join(dir, "real-kb");
        mkdirSync(real);
        const link = join(dir, "link-kb");
        symlinkSync(real, link);
        const root = await ensureRoot(link);
        // The returned root is the real path, not the symlink path.
        assert.equal(existsSync(root), true);
        assert.ok(root.includes("real-kb"));
    });
});

// ---------------------------------------------------------------------------
// Atomic write / move / delete / read
// ---------------------------------------------------------------------------

test("writeNoteFileAtomic creates parent dirs and leaves no temp file", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        const abs = await writeNoteFileAtomic(root, "deep/sub/note.md", "hello", "w0");
        assert.equal(readFileSync(abs, "utf8"), "hello");
        // No leftover .tmp anywhere under the subdir.
        const files = readdirSync(join(root, "deep", "sub"));
        assert.deepEqual(files, ["note.md"]);
    });
});

test("writeNoteFileAtomic overwrites an existing file in place", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        await writeNoteFileAtomic(root, "n.md", "v1", "w1");
        const abs = await writeNoteFileAtomic(root, "n.md", "v2", "w2");
        assert.equal(readFileSync(abs, "utf8"), "v2");
    });
});

test("readNoteFile returns the contents, or undefined when absent", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        assert.equal(await readNoteFile(root, "missing.md"), undefined);
        await writeNoteFileAtomic(root, "there.md", "x", "w3");
        assert.equal(await readNoteFile(root, "there.md"), "x");
    });
});

test("moveNoteFile relocates a file and creates the destination dir", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        await writeNoteFileAtomic(root, "old.md", "data", "w4");
        await moveNoteFile(root, "old.md", "new/place.md");
        assert.equal(existsSync(join(root, "old.md")), false);
        assert.equal(readFileSync(join(root, "new", "place.md"), "utf8"), "data");
    });
});

test("moveNoteFile is a no-op when the source is missing", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        await assert.doesNotReject(() => moveNoteFile(root, "ghost.md", "dest.md"));
        assert.equal(existsSync(join(root, "dest.md")), false);
    });
});

test("deleteNoteFile removes a file and is idempotent", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        await writeNoteFileAtomic(root, "doomed.md", "x", "w5");
        assert.equal(await deleteNoteFile(root, "doomed.md"), true);
        assert.equal(await deleteNoteFile(root, "doomed.md"), false);
        assert.equal(existsSync(join(root, "doomed.md")), false);
    });
});

test("write/move/delete refuse to touch anything outside the root", async () => {
    await withTempDir(async (dir) => {
        const root = await ensureRoot(join(dir, "kb"));
        await assert.rejects(() => writeNoteFileAtomic(root, "../outside.md", "x", "w6"));
        await assert.rejects(() => deleteNoteFile(root, "../../etc/passwd-ish.md"));
    });
});

// ---------------------------------------------------------------------------
// Default path derivation
// ---------------------------------------------------------------------------

test("defaultPathForNote slugifies the title", () => {
    assert.equal(defaultPathForNote("My Great Note!", "u"), "my-great-note.md");
    assert.equal(defaultPathForNote("  spaced  out  ", "u"), "spaced-out.md");
});

test("defaultPathForNote falls back to the uuid for an unsluggable title", () => {
    assert.equal(defaultPathForNote("!!!", "abc-123"), "abc-123.md");
});
