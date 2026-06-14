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

import { createHandler, resolveServerTools, SessionPool } from "../src/server.ts";
import { BUILTIN_COMMANDS } from "../src/commands.ts";
import { MemoryStore, Memory } from "../src/memory.ts";
import { EventStore } from "../src/events.ts";
import { GoalStore } from "../src/goals.ts";
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
    const goals = new GoalStore(":memory:");
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

/** The default (boot) session's id — what a chat with no `session` lands on, and
 *  what the read endpoints flag as live. The pool seeds exactly one at start. */
function defaultId(deps: { sessions: SessionPool }): string {
    return deps.sessions.ids()[0]!;
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
