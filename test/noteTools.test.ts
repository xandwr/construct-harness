/**
 * Tests for the knowledge-base tool bridge ({@link noteTools}).
 *
 * These exercise each tool directly (calling `ToolDef.run` with an args bag, the
 * way the loop would). Writes go through a real temp-dir {@link NotesService}, so
 * a `note_save` genuinely lands a file on disk as well as a row; reads go through
 * the {@link NotesStore}. An offline fake embedder makes semantic recall
 * deterministic without a network.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { NotesStore } from "../src/notes.ts";
import { NotesService } from "../src/notesService.ts";
import { MemoryStore, Memory } from "../src/memory.ts";
import { noteTools, DEFAULT_NOTE_RECALL_LIMIT } from "../src/noteTools.ts";
import { EmbeddingError, type Embedder } from "../src/embeddings.ts";
import type { ToolDef } from "../src/types.ts";

interface Harness {
    dir: string;
    kb: string;
    store: NotesStore;
    memStore: MemoryStore;
    service: NotesService;
    tools: ToolDef[];
}

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
            if (opts.fail) throw new EmbeddingError("down");
            return texts.map((t) => norm(table[t] ?? [-1, -1]));
        },
    };
}

async function withTools(fn: (h: Harness) => Promise<void>, embedder?: Embedder): Promise<void> {
    const dir = mkdtempSync(join(tmpdir(), "notetools-"));
    const kb = join(dir, "kb");
    const dbPath = join(dir, "db.sqlite");
    const store = new NotesStore(dbPath);
    const memStore = new MemoryStore(dbPath);
    // The embedder is shared by the service (so outbound writes embed) and the
    // tools (so recall embeds its query), mirroring the real server wiring.
    const service = new NotesService({
        store,
        root: kb,
        embedder,
        debounceMs: 20,
        onWarn: () => {},
    });
    await service.ready();
    const tools = noteTools(service, store, embedder);
    try {
        await fn({ dir, kb, store, memStore, service, tools });
    } finally {
        service.close();
        memStore.close();
        store.close();
        rmSync(dir, { recursive: true, force: true });
    }
}

function tool(tools: ToolDef[], name: string): ToolDef {
    const t = tools.find((x) => x.name === name);
    assert.ok(t, `expected a tool named ${name}`);
    return t;
}

// ---------------------------------------------------------------------------

test("note_save persists a note and writes its file", async () => {
    await withTools(async ({ kb, store, tools }) => {
        const res = (await tool(tools, "note_save").run({
            title: "Runbook",
            content: "step one",
            tags: ["ops"],
        })) as any;
        assert.equal(res.saved, true);
        assert.equal(res.note.title, "Runbook");
        assert.deepEqual(res.note.frontmatter, { tags: ["ops"] });
        assert.ok(existsSync(join(kb, res.note.path)));
        assert.ok(readFileSync(join(kb, res.note.path), "utf8").includes("step one"));
        assert.equal(store.getByUuid(res.note.uuid)?.content, "step one");
    });
});

test("note_save reports a path clash instead of throwing", async () => {
    await withTools(async ({ tools }) => {
        await tool(tools, "note_save").run({ title: "A", content: "x", path: "dup.md" });
        const res = (await tool(tools, "note_save").run({
            title: "B",
            content: "y",
            path: "dup.md",
        })) as any;
        assert.equal(res.saved, false);
        assert.match(res.error, /already exists/);
    });
});

test("note_update edits an existing note by uuid and rewrites the file", async () => {
    await withTools(async ({ kb, tools }) => {
        const saved = (await tool(tools, "note_save").run({ title: "T", content: "v1" })) as any;
        const res = (await tool(tools, "note_update").run({
            uuid: saved.note.uuid,
            content: "v2",
        })) as any;
        assert.equal(res.updated, true);
        assert.equal(res.note.content, "v2");
        assert.ok(readFileSync(join(kb, res.note.path), "utf8").includes("v2"));
    });
});

test("note_update on a missing uuid reports cleanly", async () => {
    await withTools(async ({ tools }) => {
        const res = (await tool(tools, "note_update").run({ uuid: "nope", content: "x" })) as any;
        assert.equal(res.updated, false);
        assert.match(res.error, /no note/);
    });
});

test("note_recall lists recent notes with no query", async () => {
    await withTools(async ({ tools }) => {
        await tool(tools, "note_save").run({ title: "One", content: "first" });
        await tool(tools, "note_save").run({ title: "Two", content: "second" });
        const res = (await tool(tools, "note_recall").run({})) as any;
        assert.equal(res.count, 2);
        assert.equal(res.notes.length, 2);
    });
});

test("note_recall ranks by lexical relevance", async () => {
    await withTools(async ({ tools }) => {
        await tool(tools, "note_save").run({ title: "Cooking", content: "braise short ribs" });
        await tool(tools, "note_save").run({ title: "Ops", content: "kubernetes rollout" });
        const res = (await tool(tools, "note_recall").run({ query: "kubernetes rollout" })) as any;
        assert.equal(res.notes[0].title, "Ops");
    });
});

test("note_recall ranks by meaning when an embedder is configured", async () => {
    const embedder = fakeEmbedder({
        "the engine needs oil": [1, 0],
        "cats are nice": [-1, 0],
        cars: [1, 0],
    });
    await withTools(async ({ tools }) => {
        await tool(tools, "note_save").run({ title: "auto", content: "the engine needs oil" });
        await tool(tools, "note_save").run({ title: "pets", content: "cats are nice" });
        const res = (await tool(tools, "note_recall").run({ query: "cars" })) as any;
        // Semantically nearest to "cars" is the engine/oil note.
        assert.equal(res.notes[0].title, "auto");
    }, embedder);
});

test("note_recall caps a pathological body", async () => {
    await withTools(async ({ tools }) => {
        await tool(tools, "note_save").run({ title: "big", content: "x".repeat(10_000) });
        const res = (await tool(tools, "note_recall").run({ query: "big" })) as any;
        assert.ok(res.notes[0].content.length < 10_000);
        assert.ok(res.notes[0].content.includes("[truncated]"));
    });
});

test("note_link links a note to another note", async () => {
    await withTools(async ({ store, tools }) => {
        const a = (await tool(tools, "note_save").run({ title: "A", content: "x" })) as any;
        const b = (await tool(tools, "note_save").run({ title: "B", content: "y" })) as any;
        const res = (await tool(tools, "note_link").run({
            from: a.note.uuid,
            toNote: b.note.uuid,
            kind: "references",
        })) as any;
        assert.equal(res.linked, true);
        assert.equal(store.linksFrom(a.note.id).length, 1);
    });
});

test("note_link links a note to a memory", async () => {
    await withTools(async ({ store, memStore, tools }) => {
        const m = memStore.save(new Memory({ content: "a fact" }));
        const a = (await tool(tools, "note_save").run({ title: "A", content: "x" })) as any;
        const res = (await tool(tools, "note_link").run({
            from: a.note.uuid,
            toMemory: m.id,
            kind: "derived_from",
        })) as any;
        assert.equal(res.linked, true);
        assert.equal(store.linksToMemory(m.id).length, 1);
    });
});

test("note_link requires exactly one target", async () => {
    await withTools(async ({ tools }) => {
        const a = (await tool(tools, "note_save").run({ title: "A", content: "x" })) as any;
        const none = (await tool(tools, "note_link").run({ from: a.note.uuid })) as any;
        assert.equal(none.linked, false);
        const both = (await tool(tools, "note_link").run({
            from: a.note.uuid,
            toNote: "x",
            toMemory: 1,
        })) as any;
        assert.equal(both.linked, false);
    });
});

test("note_forget deletes the row, its links, and the file on disk", async () => {
    await withTools(async ({ kb, store, tools }) => {
        const a = (await tool(tools, "note_save").run({ title: "A", content: "x" })) as any;
        const b = (await tool(tools, "note_save").run({ title: "B", content: "y" })) as any;
        await tool(tools, "note_link").run({ from: a.note.uuid, toNote: b.note.uuid });
        assert.ok(existsSync(join(kb, a.note.path)));
        assert.equal(store.linksFrom(a.note.id).length, 1);

        const res = (await tool(tools, "note_forget").run({ uuid: a.note.uuid })) as any;
        assert.equal(res.forgotten, true);
        assert.equal(store.getByUuid(a.note.uuid), undefined); // row gone
        assert.ok(!existsSync(join(kb, a.note.path))); // file gone
        // The link cascaded with the source note (from_note ON DELETE CASCADE).
        assert.equal(store.linksToNote(b.note.id).length, 0);
    });
});

test("note_forget on a missing uuid reports cleanly", async () => {
    await withTools(async ({ tools }) => {
        const res = (await tool(tools, "note_forget").run({ uuid: "nope" })) as any;
        assert.equal(res.forgotten, false);
        assert.match(res.error, /no note/);
    });
});

test("note_links reads outgoing links with ids and targets", async () => {
    await withTools(async ({ memStore, tools }) => {
        const m = memStore.save(new Memory({ content: "a fact" }));
        const a = (await tool(tools, "note_save").run({ title: "A", content: "x" })) as any;
        const b = (await tool(tools, "note_save").run({ title: "B", content: "y" })) as any;
        await tool(tools, "note_link").run({
            from: a.note.uuid,
            toNote: b.note.uuid,
            kind: "references",
        });
        await tool(tools, "note_link").run({
            from: a.note.uuid,
            toMemory: m.id,
            kind: "derived_from",
        });

        const res = (await tool(tools, "note_links").run({ uuid: a.note.uuid })) as any;
        assert.equal(res.direction, "out");
        assert.equal(res.count, 2);
        const noteLink = res.links.find((l: any) => l.kind === "note");
        assert.equal(noteLink.relation, "references");
        assert.equal(noteLink.toUuid, b.note.uuid); // target resolved to its uuid
        assert.ok(typeof noteLink.id === "number"); // usable with note_unlink
        const memLink = res.links.find((l: any) => l.kind === "memory");
        assert.equal(memLink.toMemory, m.id);
    });
});

test("note_links direction:'in' is the reverse lookup", async () => {
    await withTools(async ({ tools }) => {
        const a = (await tool(tools, "note_save").run({ title: "A", content: "x" })) as any;
        const b = (await tool(tools, "note_save").run({ title: "B", content: "y" })) as any;
        await tool(tools, "note_link").run({ from: a.note.uuid, toNote: b.note.uuid });

        // B has no outgoing links, but one incoming (from A).
        const out = (await tool(tools, "note_links").run({ uuid: b.note.uuid })) as any;
        assert.equal(out.count, 0);
        const incoming = (await tool(tools, "note_links").run({
            uuid: b.note.uuid,
            direction: "in",
        })) as any;
        assert.equal(incoming.direction, "in");
        assert.equal(incoming.count, 1);
        assert.equal(incoming.links[0].toUuid, b.note.uuid);
    });
});

test("note_links on a missing uuid reports cleanly", async () => {
    await withTools(async ({ tools }) => {
        const res = (await tool(tools, "note_links").run({ uuid: "nope" })) as any;
        assert.match(res.error, /no note/);
    });
});

test("note_unlink removes one edge by id without touching the notes", async () => {
    await withTools(async ({ store, tools }) => {
        const a = (await tool(tools, "note_save").run({ title: "A", content: "x" })) as any;
        const b = (await tool(tools, "note_save").run({ title: "B", content: "y" })) as any;
        const linked = (await tool(tools, "note_link").run({
            from: a.note.uuid,
            toNote: b.note.uuid,
        })) as any;

        const res = (await tool(tools, "note_unlink").run({ id: linked.link.id })) as any;
        assert.equal(res.unlinked, true);
        assert.equal(store.linksFrom(a.note.id).length, 0);
        // Both notes survive: only the edge was removed.
        assert.ok(store.getByUuid(a.note.uuid));
        assert.ok(store.getByUuid(b.note.uuid));

        // Unlinking a gone id is a clean false, not an error.
        const again = (await tool(tools, "note_unlink").run({ id: linked.link.id })) as any;
        assert.equal(again.unlinked, false);
    });
});

test("note_unlink rejects a non-numeric id", async () => {
    await withTools(async ({ tools }) => {
        const res = (await tool(tools, "note_unlink").run({ id: "nope" })) as any;
        assert.equal(res.unlinked, false);
        assert.match(res.error, /finite number/);
    });
});

test("the tool set exposes the documented names and default limit", async () => {
    await withTools(async ({ tools }) => {
        assert.deepEqual(tools.map((t) => t.name).sort(), [
            "note_forget",
            "note_link",
            "note_links",
            "note_recall",
            "note_save",
            "note_unlink",
            "note_update",
        ]);
        assert.equal(DEFAULT_NOTE_RECALL_LIMIT, 8);
    });
});
