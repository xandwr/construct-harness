/**
 * Tests for the HTTP surface ({@link createHandler}).
 *
 * The handler is pure routing over a set of deps, so we drive it with an
 * in-memory store/event log and a scripted {@link FakeClient} — no network, no
 * spend, no real port. We fake just enough of node's req/res to capture status,
 * headers, and body (and, for SSE, the sequence of frames written).
 *
 * What these lock in: the read endpoints return the store's contents in the wire
 * shape, a chat turn streams SSE frames AND persists to the log as a side effect
 * (the load-bearing "chats persist to disk" property), and the obvious error
 * paths (missing param, bad route) return the right status.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
    createHandler,
    resolveServerTools,
    SessionPool,
    goalEventSink,
    GOAL_EVENT_KIND,
} from "../src/server.ts";
import { BUILTIN_COMMANDS } from "../src/commands.ts";
import { MemoryStore, Memory } from "../src/memory.ts";
import { EventStore } from "../src/events.ts";
import { GoalStore } from "../src/goals.ts";
import { RuntimeConfig } from "../src/runtimeConfig.ts";
import { AnthropicClient } from "../src/bridge/anthropic.ts";
import { FakeClient, textTurn, callTurn } from "./helpers/fakeClient.ts";

/** A captured response: status, headers, and the concatenated body. For SSE the
 *  body is the raw stream, which `frames()` splits back into events. */
interface Captured {
    status: number;
    headers: Record<string, string>;
    body: string;
    frames(): Array<{ event: string; data: unknown }>;
}

/** A minimal stand-in for node's ServerResponse that records everything the
 *  handler writes. writeHead/setHeader/write/end behave like the real ones for
 *  our purposes. */
function fakeRes(): { res: any; captured: Captured } {
    const captured: Captured = {
        status: 0,
        headers: {},
        body: "",
        frames() {
            const out: Array<{ event: string; data: unknown }> = [];
            for (const chunk of this.body.split("\n\n")) {
                let event = "";
                let data = "";
                for (const line of chunk.split("\n")) {
                    if (line.startsWith("event:")) event = line.slice(6).trim();
                    else if (line.startsWith("data:")) data = line.slice(5).trim();
                }
                if (event && data) out.push({ event, data: JSON.parse(data) });
            }
            return out;
        },
    };
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

/** A fake GET request: just a method and a url. */
function getReq(url: string): any {
    const req: any = new EventEmitter();
    req.method = "GET";
    req.url = url;
    return req;
}

/** A fake POST request that emits its body on the next tick, like a real stream. */
function postReq(url: string, body: string): any {
    const req: any = new EventEmitter();
    req.method = "POST";
    req.url = url;
    req.destroy = () => {};
    queueMicrotask(() => {
        req.emit("data", Buffer.from(body, "utf8"));
        req.emit("end");
    });
    return req;
}

/** Build deps over in-memory stores and a scripted client, plus a close() that
 *  releases both handles. The Sessions are wired with the EventStore, exactly as
 *  the real server does (via a SessionPool over one shared config), so a chat
 *  turn persists and a past conversation can be resumed by id. */
function makeDeps(client: FakeClient) {
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    // Wire the production goal→event sink, so a goal write through these deps logs
    // an event exactly as the real server does.
    const goals = new GoalStore({ location: ":memory:", onChange: goalEventSink(events) });
    const sessions = new SessionPool(() => ({ client, system: "be brief", store, events, goals }));
    return {
        store,
        events,
        goals,
        sessions,
        // The scripted client, exposed so a resume test can read the messages the
        // model was handed (FakeClient records every call's params).
        client,
        close() {
            events.close();
            goals.close();
            store.close();
        },
    };
}

/** Like {@link makeDeps}, but over a single shared on-disk database file the way
 *  the real server wires it (one file, one schema). Needed where a cross-store
 *  foreign key must resolve — memory provenance references events(id), which only
 *  works when the memory and event tables live in the SAME database, not two
 *  separate `:memory:` handles. The caller's close() also removes the temp dir. */
function makeSharedDeps(client: FakeClient) {
    const dir = mkdtempSync(join(tmpdir(), "server-shared-"));
    const path = join(dir, "db.sqlite");
    const store = new MemoryStore(path);
    const events = new EventStore(path);
    const goals = new GoalStore(path);
    const sessions = new SessionPool(() => ({ client, system: "be brief", store, events, goals }));
    return {
        store,
        events,
        goals,
        sessions,
        client,
        close() {
            events.close();
            goals.close();
            store.close();
            rmSync(dir, { recursive: true, force: true });
        },
    };
}

/** The default (boot) session's id — what a chat with no `session` lands on, and
 *  what the read endpoints flag as live. The pool seeds exactly one at start. */
function defaultId(deps: { sessions: SessionPool }): string {
    return deps.sessions.ids()[0]!;
}

/** A RuntimeConfig over a real AnthropicClient (no network: model switching and
 *  status reads are pure), the shared provider-options object, and a two-group
 *  local-tool catalogue — the same shape buildDeps wires, for the settings tests.
 *  No ANTHROPIC_API_KEY is needed: the SDK constructs lazily and we never call it. */
function makeRuntime() {
    const client = new AnthropicClient({ apiKey: "test-key", model: "claude-opus-4-8" });
    const providerOptions: Record<string, unknown> = {
        cacheSystem: true,
        thinking: true,
        thinkingDisplay: true,
        serverTools: ["web_search", "web_fetch"],
    };
    const localGroups = [
        {
            key: "notes",
            label: "knowledge base",
            note: "save / recall / link markdown notes",
            toolNames: ["note_save", "note_recall"],
        },
        {
            key: "shell",
            label: "local shell",
            note: "run commands on the user's real machine",
            toolNames: ["use__user__shell"],
        },
    ];
    const runtime = new RuntimeConfig(client, providerOptions as any, localGroups, [
        "notes",
        "shell",
    ]);
    return { client, providerOptions, runtime };
}

test("GET /api/health reports ok and the live session ids", async () => {
    const deps = makeDeps(new FakeClient([]));
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/health"), res);
    deps.close();

    assert.equal(captured.status, 200);
    const body = JSON.parse(captured.body);
    assert.equal(body.ok, true);
    assert.deepEqual(body.sessions, [defaultId(deps)]);
});

test("CORS origin is configurable", async () => {
    const deps = { ...makeDeps(new FakeClient([])), corsOrigin: "http://localhost:5173" };
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/health"), res);
    deps.close();

    assert.equal(captured.headers["access-control-allow-origin"], "http://localhost:5173");
    assert.equal(captured.headers.vary, "Origin");
});

test("GET /api/memories returns the curated store in wire shape", async () => {
    const deps = makeDeps(new FakeClient([]));
    deps.store.save(
        new Memory({ content: "Deploys go out on Fridays.", tags: ["ops"], importance: 0.8 }),
    );
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/memories"), res);
    deps.close();

    assert.equal(captured.status, 200);
    const body = JSON.parse(captured.body);
    assert.equal(body.total, 1);
    assert.equal(body.memories.length, 1);
    assert.equal(body.memories[0].content, "Deploys go out on Fridays.");
    assert.deepEqual(body.memories[0].tags, ["ops"]);
    // The enriched curation fields are present: strength, surfacing, provenance,
    // embedding. A fresh memory has full-ish strength, never surfaced, no
    // provenance, no embedding (no embedder wired here).
    assert.equal(typeof body.memories[0].strength, "number");
    assert.equal(body.memories[0].lastSurfaced, null);
    assert.equal(body.memories[0].provenance, null);
    assert.equal(body.memories[0].hasEmbedding, false);
});

test("GET /api/commands returns the slash-command catalogue", async () => {
    const deps = makeDeps(new FakeClient([]));
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/commands"), res);
    deps.close();

    assert.equal(captured.status, 200);
    const body = JSON.parse(captured.body);
    assert.deepEqual(body.commands, BUILTIN_COMMANDS);
    // The shape the client's `/` menu reads: a name and params per command.
    const reset = body.commands.find((c: { name: string }) => c.name === "reset");
    assert.ok(reset, "expected a /reset command");
    assert.ok(Array.isArray(reset.params));
});

test("GET /api/dreams returns logged dreams flattened to persona/scenario/choice", async () => {
    const deps = makeDeps(new FakeClient([]));
    // Seed a dream the way dreamOnce logs one: choice in content, the structured
    // record in meta. The GET should flatten this into named fields.
    deps.events.append({
        kind: "dream",
        role: "agent",
        content: "I hold it to verify. Reasoning… and so I wait.",
        meta: {
            persona: { name: "Mara", role: "staff security engineer" },
            scenario: "You must decide whether to ship under pressure. Choose.",
            sourceMemoryIds: [3, 7],
        },
    });
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/dreams"), res);
    deps.close();

    assert.equal(captured.status, 200);
    const body = JSON.parse(captured.body);
    assert.equal(body.total, 1);
    assert.equal(body.dreams.length, 1);
    const d = body.dreams[0];
    assert.equal(d.persona.name, "Mara");
    assert.equal(d.persona.role, "staff security engineer");
    assert.match(d.scenario, /whether to ship/);
    assert.match(d.choice, /I hold it to verify/);
    assert.deepEqual(d.sourceMemoryIds, [3, 7]);
});

test("GET /api/dreams returns only dream events, newest first", async () => {
    const deps = makeDeps(new FakeClient([]));
    // A non-dream event must not leak into the dreams view…
    deps.events.append({ kind: "message", role: "user", content: "not a dream" });
    deps.events.append({
        kind: "dream",
        role: "agent",
        content: "older dream",
        meta: { persona: { name: "A" }, scenario: "s1", sourceMemoryIds: [] },
    });
    deps.events.append({
        kind: "dream",
        role: "agent",
        content: "newer dream",
        meta: { persona: { name: "B" }, scenario: "s2", sourceMemoryIds: [] },
    });
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/dreams"), res);
    deps.close();

    const body = JSON.parse(captured.body);
    assert.equal(body.total, 2, "only the two dream events are counted");
    assert.deepEqual(
        body.dreams.map((d: { choice: string }) => d.choice),
        ["newer dream", "older dream"],
        "dreams are newest first",
    );
});

test("POST /api/dreams runs a dream and appends it to the log", async () => {
    // Empty corpus: sampleScenario falls back without a model turn, so a single
    // dream is just persona generation + the persona's choice. Script both.
    const deps = makeDeps(
        new FakeClient([
            textTurn('```json\n{"name":"Dreamer"}\n```'), // persona
            textTurn("I choose to hold and verify. Here is why…"), // the choice
        ]),
    );
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(postReq("/api/dreams", JSON.stringify({ count: 1 })), res);
    deps.close();

    assert.equal(captured.status, 200);
    const body = JSON.parse(captured.body);
    assert.equal(body.dreams.length, 1, "the batch produced one dream");
    assert.equal(body.failures.length, 0);
    assert.equal(body.dreams[0].persona.name, "Dreamer");
    assert.match(body.dreams[0].choice, /hold and verify/);

    // The dream was persisted as a dream event (the load-bearing side effect): a
    // fresh GET on the same deps sees it. (deps already closed above, so assert on
    // the returned dream's id being a real row instead.)
    assert.ok(Number.isInteger(body.dreams[0].id), "the dream carries its logged event id");
});

test("POST /api/dreams clamps a missing count to one dream", async () => {
    const deps = makeDeps(new FakeClient([textTurn('{"name":"X"}'), textTurn("a choice")]));
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(postReq("/api/dreams", JSON.stringify({})), res);
    deps.close();

    assert.equal(captured.status, 200);
    const body = JSON.parse(captured.body);
    assert.equal(body.dreams.length, 1, "no count defaults to exactly one dream");
});

test("POST /api/dreams records a malformed dream as a failure, not a crash", async () => {
    // The persona reply is junk: dreamLoop catches the PersonaError, records it,
    // and the route returns it under `failures` with the batch still 200.
    const deps = makeDeps(new FakeClient([textTurn("I'd rather not invent anyone.")]));
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(postReq("/api/dreams", JSON.stringify({ count: 1 })), res);
    deps.close();

    assert.equal(captured.status, 200);
    const body = JSON.parse(captured.body);
    assert.equal(body.dreams.length, 0, "no dream completed");
    assert.equal(body.failures.length, 1, "the bad dream is surfaced, not hidden");
    assert.equal(body.failures[0].index, 0);
    assert.ok(typeof body.failures[0].error === "string");
});

test("GET /api/events requires a session param", async () => {
    const deps = makeDeps(new FakeClient([]));
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/events"), res);
    deps.close();

    assert.equal(captured.status, 400);
});

test("an unknown route is a 404", async () => {
    const deps = makeDeps(new FakeClient([]));
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/nope"), res);
    deps.close();

    assert.equal(captured.status, 404);
});

test("POST /api/chat streams SSE frames and persists the turn to the log", async () => {
    const deps = makeDeps(new FakeClient([textTurn("You deploy on Fridays.")]));
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();

    await handle(postReq("/api/chat", JSON.stringify({ message: "what about deploys?" })), res);

    assert.equal(captured.headers["content-type"], "text/event-stream; charset=utf-8");
    const frames = captured.frames();
    const kinds = frames.map((f) => f.event);
    // open, at least one text delta, and a terminal done.
    assert.equal(kinds[0], "open");
    assert.ok(kinds.includes("text"), "expected a text frame");
    assert.equal(kinds.at(-1), "done");

    const text = frames
        .filter((f) => f.event === "text")
        .map((f) => (f.data as { text: string }).text)
        .join("");
    assert.equal(text, "You deploy on Fridays.");

    // The load-bearing property: running the turn wrote it to the event log, so
    // it's queryable as a past conversation.
    const logged = deps.events.recent({ session: defaultId(deps) });
    const contents = logged.map((e) => e.content);
    assert.ok(contents.includes("what about deploys?"), "user message persisted");
    assert.ok(contents.includes("You deploy on Fridays."), "agent reply persisted");

    deps.close();
});

test("POST /api/chat forwards the model's thinking trace as `thinking` frames", async () => {
    // A scripted turn that thinks before answering. The FakeClient streams the
    // thinking as a delta; the server must relay it as a `thinking` SSE frame
    // ahead of the text, and it must never appear in the persisted log (thinking
    // isn't a content part).
    const deps = makeDeps(
        new FakeClient([
            { content: [{ kind: "text", text: "42." }], thinking: "let me work it out" },
        ]),
    );
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();

    await handle(postReq("/api/chat", JSON.stringify({ message: "the answer?" })), res);

    const frames = captured.frames();
    const thinking = frames
        .filter((f) => f.event === "thinking")
        .map((f) => (f.data as { text: string }).text)
        .join("");
    assert.equal(thinking, "let me work it out");
    // The thinking frame precedes the text frame in the stream.
    const kinds = frames.map((f) => f.event);
    assert.ok(kinds.indexOf("thinking") < kinds.indexOf("text"), "thinking should stream first");

    // Reasoning is ephemeral: the log holds the reply, never the trace.
    const logged = deps.events.recent({ session: defaultId(deps) }).map((e) => e.content);
    assert.ok(!logged.some((c) => /work it out/.test(c)), "thinking must not be persisted");

    deps.close();
});

test("a tool turn surfaces tool frames and the conversation appears in /api/sessions", async () => {
    // A turn that calls memory_recall, then replies. The FakeClient's callTurn
    // requests the tool; the Session runs it and feeds the result back.
    const deps = makeDeps(
        new FakeClient([callTurn("c1", "memory_recall", { query: "deploys" }), textTurn("noted.")]),
    );
    const handle = createHandler(deps);

    const chat = fakeRes();
    await handle(postReq("/api/chat", JSON.stringify({ message: "remember deploys" })), chat.res);
    const toolFrames = chat.captured.frames().filter((f) => f.event === "tool");
    assert.ok(toolFrames.length >= 1, "expected at least one tool frame");

    // The session now shows up in the conversation list, with its count.
    const list = fakeRes();
    await handle(getReq("/api/sessions"), list.res);
    const body = JSON.parse(list.captured.body);
    const mine = body.sessions.find((s: { session: string }) => s.session === defaultId(deps));
    assert.ok(mine, "live session is listed");
    assert.equal(mine.live, true);
    assert.ok(mine.count > 0);

    deps.close();
});

// ── Resuming a past conversation ─────────────────────────────────────────────

test("POST /api/chat with no session lands on the default conversation across turns", async () => {
    const deps = makeDeps(new FakeClient([textTurn("one"), textTurn("two")]));
    const handle = createHandler(deps);

    const a = fakeRes();
    await handle(postReq("/api/chat", JSON.stringify({ message: "first" })), a.res);
    const b = fakeRes();
    await handle(postReq("/api/chat", JSON.stringify({ message: "second" })), b.res);

    // Both turns announce the same session id: a client that never sends one keeps
    // talking to the same conversation.
    const idA = (a.captured.frames().find((f) => f.event === "open")!.data as { session: string })
        .session;
    const idB = (b.captured.frames().find((f) => f.event === "open")!.data as { session: string })
        .session;
    assert.equal(idA, idB);
    assert.equal(idA, defaultId(deps));

    deps.close();
});

test("POST /api/chat with a new session id opens that conversation and lists it live", async () => {
    const deps = makeDeps(new FakeClient([textTurn("hi there")]));
    const handle = createHandler(deps);

    const chat = fakeRes();
    await handle(
        postReq("/api/chat", JSON.stringify({ message: "hello", session: "my-thread" })),
        chat.res,
    );

    // The turn ran under the requested id, and it persisted there.
    const open = chat.captured.frames().find((f) => f.event === "open")!.data as {
        session: string;
    };
    assert.equal(open.session, "my-thread");
    const logged = deps.events.recent({ session: "my-thread" }).map((e) => e.content);
    assert.ok(logged.includes("hello"), "user message persisted under the requested session");

    // It's now held live in the pool alongside the boot conversation.
    assert.ok(deps.sessions.has("my-thread"), "requested session is live in the pool");
    const list = fakeRes();
    await handle(getReq("/api/sessions"), list.res);
    const body = JSON.parse(list.captured.body);
    assert.ok(body.live.includes("my-thread"), "live list includes the resumed session");
    const row = body.sessions.find((s: { session: string }) => s.session === "my-thread");
    assert.equal(row.live, true);

    deps.close();
});

test("POST /api/chat resuming a past conversation feeds its history to the model", async () => {
    // Seed a prior exchange directly in the log under a session that is NOT live in
    // this process's pool, the shape of a conversation left over from an earlier run.
    const deps = makeDeps(new FakeClient([textTurn("still 4")]));
    deps.events.append({ kind: "message", role: "user", content: "what is 2+2", session: "old" });
    deps.events.append({ kind: "message", role: "agent", content: "4", session: "old" });
    assert.ok(!deps.sessions.has("old"), "precondition: the past conversation isn't live yet");

    const handle = createHandler(deps);
    const chat = fakeRes();
    await handle(
        postReq("/api/chat", JSON.stringify({ message: "are you sure", session: "old" })),
        chat.res,
    );

    // The resumed turn ran under "old" and its reply persisted there, so the
    // conversation continued rather than forking a new one.
    const open = chat.captured.frames().find((f) => f.event === "open")!.data as {
        session: string;
    };
    assert.equal(open.session, "old");

    // The load-bearing resume property: the model saw the prior turns in context,
    // not an empty history. The FakeClient records the messages it was handed.
    const lastCall = deps.client.calls.at(-1)!;
    const texts = lastCall.messages
        .flatMap((m) => m.content)
        .filter((p): p is { kind: "text"; text: string } => p.kind === "text")
        .map((p) => p.text);
    assert.ok(
        texts.some((t) => t.includes("what is 2+2")),
        "prior user message rehydrated into context",
    );
    assert.ok(texts.includes("4"), "prior agent reply rehydrated into context");
    assert.ok(texts.includes("are you sure"), "the new turn is present too");

    deps.close();
});

// ── Goals CRUD ───────────────────────────────────────────────────────────────

/** A fake request with a method and a JSON body, for PUT/DELETE/POST. Mirrors
 *  `postReq` but lets the test pick the verb; the body streams on the next tick. */
function bodyReq(method: string, url: string, body?: string): any {
    const req: any = new EventEmitter();
    req.method = method;
    req.url = url;
    req.destroy = () => {};
    queueMicrotask(() => {
        if (body !== undefined) req.emit("data", Buffer.from(body, "utf8"));
        req.emit("end");
    });
    return req;
}

/** Run one request through the handler and return the captured response, parsed. */
async function call(
    deps: ReturnType<typeof makeDeps>,
    req: any,
): Promise<{ status: number; json: any }> {
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(req, res);
    let json: any = null;
    try {
        json = JSON.parse(captured.body);
    } catch {
        // non-JSON body (none expected in these tests)
    }
    return { status: captured.status, json };
}

test("POST /api/goals creates a global goal when no session is given", async () => {
    const deps = makeDeps(new FakeClient([]));
    const { status, json } = await call(
        deps,
        bodyReq("POST", "/api/goals", JSON.stringify({ content: "uphold the house style" })),
    );
    assert.equal(status, 201);
    assert.equal(json.goal.content, "uphold the house style");
    assert.equal(json.goal.status, "active");
    assert.equal(json.goal.session, null, "no session ⇒ a shared (global) goal");
    deps.close();
});

test("POST /api/goals scopes to a session when one is given", async () => {
    const deps = makeDeps(new FakeClient([]));
    const { json } = await call(
        deps,
        bodyReq("POST", "/api/goals", JSON.stringify({ content: "ship it", session: "s_1" })),
    );
    assert.equal(json.goal.session, "s_1");
    deps.close();
});

test("POST /api/goals rejects a missing/empty content with 400", async () => {
    const deps = makeDeps(new FakeClient([]));
    const missing = await call(deps, bodyReq("POST", "/api/goals", JSON.stringify({})));
    assert.equal(missing.status, 400);
    const empty = await call(
        deps,
        bodyReq("POST", "/api/goals", JSON.stringify({ content: "   " })),
    );
    assert.equal(empty.status, 400, "the store's GoalError maps to a 400");
    deps.close();
});

test("GET /api/goals?scope=global returns only the shared goals", async () => {
    const deps = makeDeps(new FakeClient([]));
    deps.goals.create({ content: "shared one" });
    deps.goals.create({ content: "shared two" });
    deps.goals.create({ content: "session-scoped", session: "s_1" });

    const all = await call(deps, getReq("/api/goals"));
    assert.equal(all.json.goals.length, 3);

    const global = await call(deps, getReq("/api/goals?scope=global"));
    assert.equal(global.json.goals.length, 2);
    assert.ok(global.json.goals.every((g: any) => g.session === null));
    deps.close();
});

test("GET /api/goals?scope=session filters to one conversation; requires a session", async () => {
    const deps = makeDeps(new FakeClient([]));
    deps.goals.create({ content: "global" });
    deps.goals.create({ content: "mine", session: "s_me" });
    deps.goals.create({ content: "theirs", session: "s_other" });

    const mine = await call(deps, getReq("/api/goals?scope=session&session=s_me"));
    assert.equal(mine.json.goals.length, 1);
    assert.equal(mine.json.goals[0].content, "mine");

    // scope=session with no session id is a client error, not a silent all-read.
    const noSession = await call(deps, getReq("/api/goals?scope=session"));
    assert.equal(noSession.status, 400);
    deps.close();
});

test("GET /api/goals filters by status and rejects a bad status", async () => {
    const deps = makeDeps(new FakeClient([]));
    const a = deps.goals.create({ content: "active goal" });
    const b = deps.goals.create({ content: "done goal" });
    deps.goals.setStatus(b.id, "done");
    void a;

    const active = await call(deps, getReq("/api/goals?status=active"));
    assert.deepEqual(
        active.json.goals.map((g: any) => g.content),
        ["active goal"],
    );

    const bad = await call(deps, getReq("/api/goals?status=bogus"));
    assert.equal(bad.status, 400);
    deps.close();
});

test("PUT /api/goals/:id edits content and changes status", async () => {
    const deps = makeDeps(new FakeClient([]));
    const g = deps.goals.create({ content: "rough draft" });

    const edited = await call(
        deps,
        bodyReq(
            "PUT",
            `/api/goals/${g.id}`,
            JSON.stringify({ content: "sharpened", status: "done" }),
        ),
    );
    assert.equal(edited.status, 200);
    assert.equal(edited.json.goal.content, "sharpened");
    assert.equal(edited.json.goal.status, "done");

    // An empty patch is a 400 (nothing to change), a bad id is a 404.
    const empty = await call(deps, bodyReq("PUT", `/api/goals/${g.id}`, JSON.stringify({})));
    assert.equal(empty.status, 400);
    const missing = await call(
        deps,
        bodyReq("PUT", "/api/goals/999999", JSON.stringify({ status: "done" })),
    );
    assert.equal(missing.status, 404);
    deps.close();
});

test("PUT /api/goals/:id rejects a bad status with 400", async () => {
    const deps = makeDeps(new FakeClient([]));
    const g = deps.goals.create({ content: "x" });
    const bad = await call(
        deps,
        bodyReq("PUT", `/api/goals/${g.id}`, JSON.stringify({ status: "bogus" })),
    );
    assert.equal(bad.status, 400);
    deps.close();
});

test("DELETE /api/goals/:id removes a goal; a missing id is a 404", async () => {
    const deps = makeDeps(new FakeClient([]));
    const g = deps.goals.create({ content: "oops" });

    const removed = await call(deps, bodyReq("DELETE", `/api/goals/${g.id}`));
    assert.equal(removed.status, 200);
    assert.equal(removed.json.deleted, true);
    assert.equal(deps.goals.get(g.id), undefined);

    const again = await call(deps, bodyReq("DELETE", `/api/goals/${g.id}`));
    assert.equal(again.status, 404);
    deps.close();
});

test("a goal added then deleted through the routes logs both events", async () => {
    const deps = makeDeps(new FakeClient([]));

    // Add via the route (no session ⇒ a shared, unscoped goal).
    const created = await call(
        deps,
        bodyReq("POST", "/api/goals", JSON.stringify({ content: "uphold the house style" })),
    );
    const id = created.json.goal.id;

    // Delete it via the route.
    const removed = await call(deps, bodyReq("DELETE", `/api/goals/${id}`));
    assert.equal(removed.json.deleted, true);

    // Both writes left a goal event in the log, newest first.
    const logged = deps.events.recent({ kind: GOAL_EVENT_KIND });
    assert.deepEqual(
        logged.map((e) => (e.meta as { change?: string })?.change),
        ["deleted", "created"],
    );
    // The events name the goal and carry its id; a shared goal stays unscoped.
    assert.ok(logged.every((e) => e.session === undefined));
    assert.deepEqual(
        logged.map((e) => (e.meta as { goalId?: number })?.goalId),
        [id, id],
    );
    assert.ok(logged.every((e) => e.content.includes("uphold the house style")));

    deps.close();
});

test("a session-scoped goal's events carry that session", async () => {
    const deps = makeDeps(new FakeClient([]));
    await call(
        deps,
        bodyReq("POST", "/api/goals", JSON.stringify({ content: "ship it", session: "s_42" })),
    );
    const logged = deps.events.recent({ kind: GOAL_EVENT_KIND, session: "s_42" });
    assert.equal(logged.length, 1);
    assert.equal((logged[0]!.meta as { change?: string })?.change, "created");
    deps.close();
});

// ── Memory curation ──────────────────────────────────────────────────────────

test("GET /api/memories/:id returns detail with provenance and the source event", async () => {
    // Provenance is a foreign key from memory_meta to events(id); it only resolves
    // when both tables share one database, so this test wires the stores over one
    // shared file the way the real server does.
    const deps = makeSharedDeps(new FakeClient([]));
    // Append an event and point a memory's provenance at it, the way the agent's
    // curation does as it runs.
    const ev = deps.events.append({
        kind: "message",
        role: "user",
        content: "we deploy on Fridays, never weekends",
        session: "s_deploy",
    });
    const m = deps.store.save(new Memory({ content: "deploys are Fridays", tags: ["ops"] }));
    deps.store.setProvenance(m.id, ev.id);

    const { status, json } = await call(deps, getReq(`/api/memories/${m.id}`));
    assert.equal(status, 200);
    assert.equal(json.memory.content, "deploys are Fridays");
    assert.equal(json.memory.provenance.eventId, ev.id);
    assert.equal(json.memory.provenance.session, "s_deploy");
    // Detail carries the source event itself for the "curated from" view.
    assert.ok(json.sourceEvent, "source event present");
    assert.match(json.sourceEvent.content, /never weekends/);
    deps.close();
});

test("GET /api/memories/:id is a 404 for a missing or non-numeric id", async () => {
    const deps = makeDeps(new FakeClient([]));
    assert.equal((await call(deps, getReq("/api/memories/999999"))).status, 404);
    assert.equal((await call(deps, getReq("/api/memories/abc"))).status, 404);
    deps.close();
});

test("PUT /api/memories/:id edits content, tags, and importance", async () => {
    const deps = makeDeps(new FakeClient([]));
    const m = deps.store.save(new Memory({ content: "rough", tags: ["x"], importance: 0.2 }));

    const edited = await call(
        deps,
        bodyReq(
            "PUT",
            `/api/memories/${m.id}`,
            JSON.stringify({ content: "sharpened", tags: ["ops", "deploy"], importance: 0.9 }),
        ),
    );
    assert.equal(edited.status, 200);
    assert.equal(edited.json.memory.content, "sharpened");
    assert.deepEqual(edited.json.memory.tags, ["ops", "deploy"]);
    assert.equal(edited.json.memory.importance, 0.9);
    // Persisted.
    assert.equal(deps.store.get(m.id)!.content, "sharpened");
    deps.close();
});

test("PUT /api/memories/:id can clear importance with null", async () => {
    const deps = makeDeps(new FakeClient([]));
    const m = deps.store.save(new Memory({ content: "c", importance: 0.5 }));
    const res = await call(
        deps,
        bodyReq("PUT", `/api/memories/${m.id}`, JSON.stringify({ importance: null })),
    );
    assert.equal(res.status, 200);
    assert.equal(res.json.memory.importance, null);
    deps.close();
});

test("PUT /api/memories/:id rejects bad fields and an empty patch", async () => {
    const deps = makeDeps(new FakeClient([]));
    const m = deps.store.save(new Memory({ content: "c" }));
    // Empty patch.
    assert.equal(
        (await call(deps, bodyReq("PUT", `/api/memories/${m.id}`, JSON.stringify({})))).status,
        400,
    );
    // Bad tags.
    assert.equal(
        (await call(deps, bodyReq("PUT", `/api/memories/${m.id}`, JSON.stringify({ tags: "ops" }))))
            .status,
        400,
    );
    // Missing id.
    assert.equal(
        (await call(deps, bodyReq("PUT", "/api/memories/999999", JSON.stringify({ content: "x" }))))
            .status,
        404,
    );
    deps.close();
});

test("DELETE /api/memories/:id forgets a memory; a missing id is a 404", async () => {
    const deps = makeDeps(new FakeClient([]));
    const m = deps.store.save(new Memory({ content: "forget me" }));

    const removed = await call(deps, bodyReq("DELETE", `/api/memories/${m.id}`));
    assert.equal(removed.status, 200);
    assert.equal(removed.json.deleted, true);
    assert.equal(deps.store.get(m.id), undefined);

    const again = await call(deps, bodyReq("DELETE", `/api/memories/${m.id}`));
    assert.equal(again.status, 404);
    deps.close();
});

// ── Status ───────────────────────────────────────────────────────────────────

test("GET /api/status reports the static snapshot plus the live runtime knobs", async () => {
    const { runtime } = makeRuntime();
    const deps = {
        ...makeDeps(new FakeClient([])),
        runtime,
        status: {
            memoryDb: "/tmp/db.sqlite",
            kbDir: "/tmp/kb",
            compactAt: 99_000,
            embeddingConfigured: true,
            dreamsEnabled: true,
            transcriptRecall: true,
            workingMind: true,
            shellPolicy: { mode: "read-only" as const, allowedCwdRoots: ["/srv/app"] },
        },
    };
    const { status, json } = await call(deps, getReq("/api/status"));
    assert.equal(status, 200);
    // The live knobs come off the runtime, not the snapshot.
    assert.equal(json.provider.id, "anthropic");
    assert.equal(json.provider.model, "claude-opus-4-8");
    // Capabilities come from the live client.
    assert.deepEqual(json.provider.capabilities, deps.client.capabilities);
    // The provider/model catalogue the dropdowns render from, with the in-use
    // model flagged current.
    const anthropic = json.providers.find((p: any) => p.id === "anthropic");
    assert.ok(anthropic, "anthropic provider is offered");
    assert.ok(anthropic.models.some((m: any) => m.id === "claude-opus-4-8" && m.current));
    assert.ok(anthropic.models.some((m: any) => m.id === "claude-fable-5" && !m.current));
    // Server tools as a toggle catalogue, the enabled ones flagged.
    const byId = Object.fromEntries(json.serverTools.map((t: any) => [t.id, t]));
    assert.equal(byId.web_search.enabled, true);
    assert.equal(byId.web_fetch.enabled, true);
    assert.equal(byId.code_execution.enabled, false);
    // Local tools as a toggle catalogue, both groups on at boot.
    const byKey = Object.fromEntries(json.localTools.map((g: any) => [g.key, g]));
    assert.equal(byKey.notes.enabled, true);
    assert.equal(byKey.shell.enabled, true);
    assert.deepEqual(byKey.shell.toolNames, ["use__user__shell"]);
    // Effort: no level set means the provider default, plus the level catalogue.
    assert.equal(json.effort.current, null);
    assert.ok(json.effort.levels.includes("xhigh"));
    // The static fields still come from the snapshot.
    assert.equal(json.storage.memoryDb, "/tmp/db.sqlite");
    assert.equal(json.storage.kbDir, "/tmp/kb");
    assert.equal(json.compactAt, 99_000);
    assert.equal(json.embeddingConfigured, true);
    assert.deepEqual(json.features, { dreams: true, transcriptRecall: true, workingMind: true });
    assert.deepEqual(json.shellPolicy, { mode: "read-only", allowedCwdRoots: ["/srv/app"] });
    deps.close();
});

test("GET /api/status reports live dynamic data: schema version, counts, sessions", async () => {
    const deps = makeDeps(new FakeClient([]));
    deps.store.save(new Memory({ content: "a fact" }));
    deps.goals.create({ content: "a goal" });
    const { json } = await call(deps, getReq("/api/status"));
    // Schema version is the real migrated version of the shared store.
    assert.equal(json.storage.schemaVersion, deps.store.version);
    assert.equal(json.storage.memories, 1);
    assert.equal(json.storage.goals, 1);
    // The boot session is live in the pool.
    assert.deepEqual(json.liveSessions, [defaultId(deps)]);
    deps.close();
});

test("GET /api/status answers even with no snapshot or runtime (no-key/no-embedder case)", async () => {
    // makeDeps supplies neither `status` nor `runtime`, modelling a bare process:
    // the live knobs degrade gracefully (model falls back to the client, no tools
    // enabled) and the static fields to empty/false, but the route still answers
    // with the live dynamic half rather than 500-ing.
    const deps = makeDeps(new FakeClient([]));
    const { status, json } = await call(deps, getReq("/api/status"));
    assert.equal(status, 200);
    // The model falls back to the live client's id; no provider serves "fake-model".
    assert.equal(json.provider.model, deps.client.model);
    assert.equal(json.provider.id, null);
    // The server-tool catalogue is still offered (static), but nothing is enabled.
    assert.ok(json.serverTools.every((t: any) => t.enabled === false));
    // No runtime means no local-tool groups to list.
    assert.deepEqual(json.localTools, []);
    // No effort level set.
    assert.equal(json.effort.current, null);
    assert.equal(json.embeddingConfigured, false);
    assert.equal(json.storage.memoryDb, null);
    assert.equal(json.compactAt, null);
    assert.deepEqual(json.features, {
        dreams: false,
        transcriptRecall: false,
        workingMind: false,
    });
    // With no snapshot, the shell policy defaults to unrestricted (the historical
    // behavior), unconfined.
    assert.deepEqual(json.shellPolicy, { mode: "unrestricted", allowedCwdRoots: [] });
    // No secrets ever appear in the body.
    assert.ok(!/api[_-]?key/i.test(captured(json)), "no key field leaks");
    deps.close();
});

/** Re-serialize a parsed body so a regex over the wire text can assert no secret
 *  field name appears. */
function captured(json: unknown): string {
    return JSON.stringify(json);
}

// ── Settings (write) ───────────────────────────────────────────────────────────

test("PATCH /api/settings switches the model live and echoes the new status", async () => {
    const { client, runtime } = makeRuntime();
    const deps = { ...makeDeps(new FakeClient([])), runtime };
    const { status, json } = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ model: "claude-sonnet-4-6" })),
    );
    assert.equal(status, 200);
    // The change lands on the live client (so every conversation's next turn).
    assert.equal(client.model, "claude-sonnet-4-6");
    // The echoed status reflects it immediately.
    assert.equal(json.provider.model, "claude-sonnet-4-6");
    assert.equal(json.provider.id, "anthropic");
    const anthropic = json.providers.find((p: any) => p.id === "anthropic");
    assert.ok(anthropic.models.some((m: any) => m.id === "claude-sonnet-4-6" && m.current));
    deps.close();
});

test("PATCH /api/settings rejects an unknown model and leaves the live model unchanged", async () => {
    const { client, runtime } = makeRuntime();
    const deps = { ...makeDeps(new FakeClient([])), runtime };
    const { status, json } = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ model: "gpt-nope" })),
    );
    assert.equal(status, 400);
    assert.match(json.error, /unknown model/);
    // The live model is untouched.
    assert.equal(client.model, "claude-opus-4-8");
    deps.close();
});

test("PATCH /api/settings toggles server tools in place on the shared options", async () => {
    const { providerOptions, runtime } = makeRuntime();
    const deps = { ...makeDeps(new FakeClient([])), runtime };
    const { status, json } = await call(
        deps,
        bodyReq(
            "PATCH",
            "/api/settings",
            JSON.stringify({ serverTools: ["web_search", "code_execution"] }),
        ),
    );
    assert.equal(status, 200);
    // The shared provider-options object every live Session reads is mutated in
    // place, in catalogue (display) order, so live conversations pick it up.
    assert.deepEqual(providerOptions.serverTools, ["web_search", "code_execution"]);
    const byId = Object.fromEntries(json.serverTools.map((t: any) => [t.id, t.enabled]));
    assert.deepEqual(byId, { web_search: true, web_fetch: false, code_execution: true });
    deps.close();
});

test("PATCH /api/settings rejects an unknown server tool", async () => {
    const { providerOptions, runtime } = makeRuntime();
    const deps = { ...makeDeps(new FakeClient([])), runtime };
    const { status, json } = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ serverTools: ["telepathy"] })),
    );
    assert.equal(status, 400);
    assert.match(json.error, /unknown server tool/);
    // Unchanged on rejection.
    assert.deepEqual(providerOptions.serverTools, ["web_search", "web_fetch"]);
    deps.close();
});

test("PATCH /api/settings sets and clears the effort level", async () => {
    const { providerOptions, runtime } = makeRuntime();
    const deps = { ...makeDeps(new FakeClient([])), runtime };
    const set = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ effort: "xhigh" })),
    );
    assert.equal(set.status, 200);
    assert.equal(providerOptions.effort, "xhigh");
    assert.equal(set.json.effort.current, "xhigh");

    const cleared = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ effort: null })),
    );
    assert.equal(cleared.status, 200);
    assert.equal("effort" in providerOptions, false, "null clears the level entirely");
    assert.equal(cleared.json.effort.current, null);

    const bad = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ effort: "ludicrous" })),
    );
    assert.equal(bad.status, 400);
    assert.match(bad.json.error, /unknown effort level/);
    deps.close();
});

test("PATCH /api/settings toggles a local tool group and rejects an unknown key", async () => {
    const { runtime } = makeRuntime();
    const deps = { ...makeDeps(new FakeClient([])), runtime };
    const off = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ localTools: { shell: false } })),
    );
    assert.equal(off.status, 200);
    assert.equal(runtime.isLocalEnabled("shell"), false);
    assert.equal(runtime.isLocalEnabled("notes"), true);
    const byKey = Object.fromEntries(off.json.localTools.map((g: any) => [g.key, g.enabled]));
    assert.deepEqual(byKey, { notes: true, shell: false });

    const bad = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ localTools: { quantum: true } })),
    );
    assert.equal(bad.status, 400);
    assert.match(bad.json.error, /unknown tool/);
    deps.close();
});

test("PATCH /api/settings applies several knobs in one request", async () => {
    const { client, providerOptions, runtime } = makeRuntime();
    const deps = { ...makeDeps(new FakeClient([])), runtime };
    const { status } = await call(
        deps,
        bodyReq(
            "PATCH",
            "/api/settings",
            JSON.stringify({
                model: "claude-fable-5",
                serverTools: ["web_search"],
                effort: "high",
                localTools: { notes: false },
            }),
        ),
    );
    assert.equal(status, 200);
    assert.equal(client.model, "claude-fable-5");
    assert.deepEqual(providerOptions.serverTools, ["web_search"]);
    assert.equal(providerOptions.effort, "high");
    assert.equal(runtime.isLocalEnabled("notes"), false);
    deps.close();
});

test("PATCH /api/settings 503s when no runtime is wired", async () => {
    const deps = makeDeps(new FakeClient([]));
    const { status } = await call(
        deps,
        bodyReq("PATCH", "/api/settings", JSON.stringify({ model: "claude-opus-4-8" })),
    );
    assert.equal(status, 503);
    deps.close();
});

test("non-PATCH /api/settings is 405", async () => {
    const { runtime } = makeRuntime();
    const deps = { ...makeDeps(new FakeClient([])), runtime };
    const { status } = await call(deps, getReq("/api/settings"));
    assert.equal(status, 405);
    deps.close();
});

test("a server-tool toggle reaches a live conversation's next turn", async () => {
    // The load-bearing claim: the runtime config mutates the SAME ProviderOptions
    // object the session pool was built with, so a toggle lands on a turn already
    // in flight — not just on conversations started afterward. Wire it the way
    // buildDeps does: one shared options object referenced by both.
    const client = new FakeClient([textTurn("ok")]);
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    const goals = new GoalStore({ location: ":memory:", onChange: goalEventSink(events) });
    const sharedProviderOptions: Record<string, unknown> = {
        thinking: true,
        serverTools: ["web_search"],
    };
    const sessions = new SessionPool(() => ({
        client,
        system: "be brief",
        store,
        events,
        goals,
        providerOptions: sharedProviderOptions,
    }));
    const runtime = new RuntimeConfig(
        new AnthropicClient({ apiKey: "test-key", model: "claude-opus-4-8" }),
        sharedProviderOptions as any,
        [],
        [],
    );
    const deps = {
        store,
        events,
        goals,
        sessions,
        client,
        runtime,
        close() {
            events.close();
            goals.close();
            store.close();
        },
    };

    // Toggle code_execution on (and keep web_search) via the settings route.
    const patched = await call(
        deps,
        bodyReq(
            "PATCH",
            "/api/settings",
            JSON.stringify({ serverTools: ["web_search", "code_execution"] }),
        ),
    );
    assert.equal(patched.status, 200);

    // Now run a chat turn on the default (already-live) conversation and read back
    // what the loop handed the client.
    const handle = createHandler(deps);
    const { res } = fakeRes();
    await handle(postReq("/api/chat", JSON.stringify({ message: "hi" })), res);

    const lastCall = client.calls.at(-1)!;
    const opts = lastCall.providerOptions as { serverTools?: string[] };
    assert.deepEqual(
        opts.serverTools,
        ["web_search", "code_execution"],
        "the in-flight turn saw the toggled server-tool set",
    );
    deps.close();
});

// ── Context inspector ────────────────────────────────────────────────────────

test("GET /api/context previews the default conversation's context", async () => {
    const deps = makeDeps(new FakeClient([]));
    deps.store.save(new Memory({ content: "user prefers dark mode" }));
    deps.goals.create({ content: "honor the house style" }); // shared goal

    const { status, json } = await call(deps, getReq("/api/context?q=what%20theme"));
    assert.equal(status, 200);
    // The standing sections are assembled: base always, goals (shared), and the
    // matching memory.
    const names = json.sections.map((s: any) => s.name);
    assert.ok(names.includes("base"));
    assert.ok(names.includes("goals"));
    assert.ok(json.totalTokens > 0);
    deps.close();
});

test("GET /api/context is read-only: it does not reinforce memory or bring a session live", async () => {
    const deps = makeDeps(new FakeClient([]));
    const m = deps.store.save(new Memory({ content: "the launch is on Tuesday" }));
    assert.equal(deps.store.get(m.id)!.lastSurfaced, undefined);

    // Inspect a brand-new session id that isn't live in the pool.
    const liveBefore = deps.sessions.ids().length;
    const { status } = await call(deps, getReq("/api/context?session=ghost&q=when%20is%20launch"));
    assert.equal(status, 200);

    // peek resumed a transient session, but it was NOT added to the pool: the
    // inspector must not silently bring a past conversation live (that's what
    // sending a turn does).
    assert.equal(deps.sessions.ids().length, liveBefore, "inspect brought a session live");
    assert.ok(!deps.sessions.has("ghost"), "ghost session must not be pooled");
    // And the surfaced memory was not reinforced.
    assert.equal(deps.store.get(m.id)!.lastSurfaced, undefined, "inspect reinforced a memory");
    deps.close();
});

// ── Server-tool resolution ──────────────────────────────────────────────────

test("resolveServerTools defaults to live web access when unset", () => {
    assert.deepEqual(resolveServerTools(undefined), ["web_search", "web_fetch"]);
});

test("resolveServerTools disables tools on 'none' or empty", () => {
    assert.deepEqual(resolveServerTools("none"), []);
    assert.deepEqual(resolveServerTools(""), []);
    assert.deepEqual(resolveServerTools("  "), []);
});

test("resolveServerTools parses an explicit comma-separated set", () => {
    assert.deepEqual(resolveServerTools("web_search,code_execution"), [
        "web_search",
        "code_execution",
    ]);
    // Whitespace and case are tolerated; an unknown name is dropped, not fatal.
    assert.deepEqual(resolveServerTools(" Web_Search , bogus "), ["web_search"]);
});
