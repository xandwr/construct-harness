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

import { createHandler } from "../src/server.ts";
import { Session } from "../src/session.ts";
import { MemoryStore, Memory } from "../src/memory.ts";
import { EventStore } from "../src/events.ts";
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
 *  releases both handles. The Session is wired with the EventStore, exactly as
 *  the real server does, so a chat turn persists. */
function makeDeps(client: FakeClient) {
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    const session = new Session({ client, system: "be brief", store, events });
    return {
        store,
        events,
        session,
        close() {
            events.close();
            store.close();
        },
    };
}

test("GET /api/health reports ok and the live session id", async () => {
    const deps = makeDeps(new FakeClient([]));
    const handle = createHandler(deps);
    const { res, captured } = fakeRes();
    await handle(getReq("/api/health"), res);
    deps.close();

    assert.equal(captured.status, 200);
    const body = JSON.parse(captured.body);
    assert.equal(body.ok, true);
    assert.equal(body.session, deps.session.id);
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
    const logged = deps.events.recent({ session: deps.session.id });
    const contents = logged.map((e) => e.content);
    assert.ok(contents.includes("what about deploys?"), "user message persisted");
    assert.ok(contents.includes("You deploy on Fridays."), "agent reply persisted");

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
    const mine = body.sessions.find((s: { session: string }) => s.session === deps.session.id);
    assert.ok(mine, "live session is listed");
    assert.equal(mine.live, true);
    assert.ok(mine.count > 0);

    deps.close();
});
