/**
 * Tests for the knowledge-base HTTP routes ({@link handleNotes} via
 * {@link createHandler}).
 *
 * Like the main server suite, the handler is driven with a fake req/res and a
 * real (temp-dir) NotesService, so a write through the API genuinely lands both
 * a DB row and a markdown file on disk. What these lock in: the five routes
 * return the right shapes and statuses, a create writes a file, an update
 * rewrites it, a delete removes it, and bad input / clashes surface as 4xx
 * rather than a bare 500.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { createHandler } from "../src/server.ts";
import { Session } from "../src/session.ts";
import { MemoryStore } from "../src/memory.ts";
import { EventStore } from "../src/events.ts";
import { NotesStore } from "../src/notes.ts";
import { NotesService } from "../src/notesService.ts";
import { FakeClient } from "./helpers/fakeClient.ts";

interface Captured {
    status: number;
    headers: Record<string, string>;
    body: string;
}

function fakeRes(): { res: any; captured: Captured } {
    const captured: Captured = { status: 0, headers: {}, body: "" };
    const res = {
        headersSent: false,
        setHeader(k: string, v: string) {
            captured.headers[k.toLowerCase()] = v;
        },
        writeHead(status: number, headers?: Record<string, string>) {
            captured.status = status;
            this.headersSent = true;
            if (headers)
                for (const [k, v] of Object.entries(headers)) captured.headers[k.toLowerCase()] = v;
            return this;
        },
        write(chunk: string) {
            this.headersSent = true;
            captured.body += chunk;
            return true;
        },
        end(chunk?: string) {
            this.headersSent = true;
            if (chunk) captured.body += chunk;
        },
    };
    return { res, captured };
}

function req(method: string, url: string, body?: string): any {
    const r: any = new EventEmitter();
    r.method = method;
    r.url = url;
    r.destroy = () => {};
    if (body !== undefined) {
        queueMicrotask(() => {
            r.emit("data", Buffer.from(body, "utf8"));
            r.emit("end");
        });
    } else {
        queueMicrotask(() => r.emit("end"));
    }
    return r;
}

async function withDeps(fn: (deps: any, kb: string) => Promise<void>): Promise<void> {
    const dir = mkdtempSync(join(tmpdir(), "server-notes-"));
    const kb = join(dir, "kb");
    const dbPath = join(dir, "db.sqlite");
    const store = new MemoryStore(dbPath);
    const events = new EventStore(dbPath);
    const notesStore = new NotesStore(dbPath);
    const notes = new NotesService({
        store: notesStore,
        root: kb,
        debounceMs: 20,
        onWarn: () => {},
    });
    await notes.ready();
    const session = new Session({ client: new FakeClient([]), system: "be brief", store, events });
    const deps = {
        store,
        events,
        session,
        notes,
        notesStore,
        close() {
            notes.close();
            events.close();
            notesStore.close();
            store.close();
        },
    };
    try {
        await fn(deps, kb);
    } finally {
        deps.close();
        rmSync(dir, { recursive: true, force: true });
    }
}

function body(captured: Captured): any {
    return JSON.parse(captured.body);
}

// ---------------------------------------------------------------------------

test("POST /api/notes creates a row and a file, returns 201", async () => {
    await withDeps(async (deps, kb) => {
        const handle = createHandler(deps);
        const { res, captured } = fakeRes();
        await handle(
            req("POST", "/api/notes", JSON.stringify({ title: "Hello", content: "world" })),
            res,
        );

        assert.equal(captured.status, 201);
        const note = body(captured).note;
        assert.equal(note.title, "Hello");
        assert.equal(note.content, "world");
        assert.ok(note.uuid);
        assert.ok(existsSync(join(kb, note.path)), "the file was written");
        assert.ok(readFileSync(join(kb, note.path), "utf8").includes("world"));
    });
});

test("POST /api/notes with no title is a 400", async () => {
    await withDeps(async (deps) => {
        const handle = createHandler(deps);
        const { res, captured } = fakeRes();
        await handle(req("POST", "/api/notes", JSON.stringify({ content: "x" })), res);
        assert.equal(captured.status, 400);
    });
});

test("POST /api/notes onto a taken path is a 400 (clash), not a 500", async () => {
    await withDeps(async (deps) => {
        const handle = createHandler(deps);
        const a = fakeRes();
        await handle(
            req("POST", "/api/notes", JSON.stringify({ title: "A", content: "x", path: "dup.md" })),
            a.res,
        );
        assert.equal(a.captured.status, 201);

        const b = fakeRes();
        await handle(
            req("POST", "/api/notes", JSON.stringify({ title: "B", content: "y", path: "dup.md" })),
            b.res,
        );
        assert.equal(b.captured.status, 400);
        assert.match(body(b.captured).error, /already exists/);
    });
});

test("GET /api/notes lists notes (summary shape, no body)", async () => {
    await withDeps(async (deps) => {
        await deps.notes.create({ title: "One", content: "first" });
        await deps.notes.create({ title: "Two", content: "second" });
        const handle = createHandler(deps);
        const { res, captured } = fakeRes();
        await handle(req("GET", "/api/notes"), res);

        assert.equal(captured.status, 200);
        const data = body(captured);
        assert.equal(data.total, 2);
        assert.equal(data.notes.length, 2);
        // Summary shape excludes the body.
        assert.ok(!("content" in data.notes[0]));
        assert.ok(data.notes[0].title);
    });
});

test("GET /api/notes?q= searches; ?prefix= filters by folder", async () => {
    await withDeps(async (deps) => {
        await deps.notes.create({ title: "Deploy", content: "rollout", path: "ops/deploy.md" });
        await deps.notes.create({ title: "Recipe", content: "braise", path: "home/recipe.md" });
        const handle = createHandler(deps);

        const q = fakeRes();
        await handle(req("GET", "/api/notes?q=braise"), q.res);
        assert.equal(body(q.captured).notes.length, 1);
        assert.equal(body(q.captured).notes[0].title, "Recipe");

        const p = fakeRes();
        await handle(req("GET", "/api/notes?prefix=ops/"), p.res);
        assert.equal(body(p.captured).notes.length, 1);
        assert.equal(body(p.captured).notes[0].title, "Deploy");
    });
});

test("GET /api/notes/:uuid returns the note with body and links", async () => {
    await withDeps(async (deps) => {
        const { note } = await deps.notes.create({ title: "T", content: "the body" });
        const handle = createHandler(deps);
        const { res, captured } = fakeRes();
        await handle(req("GET", `/api/notes/${note.uuid}`), res);

        assert.equal(captured.status, 200);
        const got = body(captured).note;
        assert.equal(got.content, "the body");
        assert.deepEqual(got.links, []);
    });
});

test("GET /api/notes/:uuid for a missing note is a 404", async () => {
    await withDeps(async (deps) => {
        const handle = createHandler(deps);
        const { res, captured } = fakeRes();
        await handle(req("GET", "/api/notes/does-not-exist"), res);
        assert.equal(captured.status, 404);
    });
});

test("PUT /api/notes/:uuid updates fields and rewrites the file", async () => {
    await withDeps(async (deps, kb) => {
        const { note } = await deps.notes.create({ title: "T", content: "v1" });
        const handle = createHandler(deps);
        const { res, captured } = fakeRes();
        await handle(req("PUT", `/api/notes/${note.uuid}`, JSON.stringify({ content: "v2" })), res);

        assert.equal(captured.status, 200);
        assert.equal(body(captured).note.content, "v2");
        assert.ok(readFileSync(join(kb, note.path), "utf8").includes("v2"));
    });
});

test("PUT that moves the path relocates the file", async () => {
    await withDeps(async (deps, kb) => {
        const { note } = await deps.notes.create({ title: "T", content: "x", path: "a.md" });
        const handle = createHandler(deps);
        const { res, captured } = fakeRes();
        await handle(req("PUT", `/api/notes/${note.uuid}`, JSON.stringify({ path: "b.md" })), res);

        assert.equal(captured.status, 200);
        assert.equal(existsSync(join(kb, "a.md")), false);
        assert.ok(existsSync(join(kb, "b.md")));
    });
});

test("DELETE /api/notes/:uuid removes the row and the file", async () => {
    await withDeps(async (deps, kb) => {
        const { note } = await deps.notes.create({ title: "T", content: "x" });
        const handle = createHandler(deps);
        const { res, captured } = fakeRes();
        await handle(req("DELETE", `/api/notes/${note.uuid}`), res);

        assert.equal(captured.status, 200);
        assert.equal(body(captured).deleted, true);
        assert.equal(existsSync(join(kb, note.path)), false);
        assert.equal(deps.notesStore.getByUuid(note.uuid), undefined);
    });
});

test("notes routes 503 cleanly when the KB is not configured", async () => {
    // Deps without a notes service (the optional half).
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    const session = new Session({ client: new FakeClient([]), system: "x", store, events });
    const deps = {
        store,
        events,
        session,
        close() {
            events.close();
            store.close();
        },
    };
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(req("GET", "/api/notes"), res);
    deps.close();
    assert.equal(captured.status, 503);
});
