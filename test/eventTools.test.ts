/**
 * Tests for the transcript tool bridge ({@link eventTools}).
 *
 * Mirrors the memory-tool suite: every store is an in-memory SQLite db so the
 * suite never touches disk, `ts` is injected wherever ordering matters, and the
 * tool is exercised two ways — directly (calling `run` with an args bag, the way
 * the loop would) and through {@link runLoop} driven by the scripted
 * {@link FakeClient}, which proves it's actually dispatchable and that results
 * flow back as a `tool_result` part.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { EventStore } from "../src/events.ts";
import {
    eventTools,
    embedEventIfPossible,
    backfillEventEmbeddings,
    DEFAULT_TRANSCRIPT_LIMIT,
} from "../src/eventTools.ts";
import { EmbeddingError, type Embedder } from "../src/embeddings.ts";
import { runLoop } from "../src/bridge/loop.ts";
import { RoleType } from "../src/types.ts";
import type { Message, ToolDef, ToolResultPart } from "../src/types.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

function freshStore(): EventStore {
    return new EventStore(":memory:");
}

/**
 * A deterministic, offline {@link Embedder}: maps each text to a normalized 2-D
 * vector via a caller-supplied table, so semantic ranking is fully predictable
 * with no network. Unknown texts get a fixed "far" vector. Mirrors the helper in
 * memoryTools.test.ts.
 */
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
            if (opts.fail) throw new EmbeddingError("embedding service down");
            return texts.map((t) => norm(table[t] ?? [-1, -1]));
        },
    };
}

/** Find a tool by name from the factory output. */
function tool(tools: ToolDef[], name: string): ToolDef {
    const t = tools.find((x) => x.name === name);
    assert.ok(t, `expected a tool named ${name}`);
    return t;
}

const user = (text: string): Message => ({
    sender: { role: RoleType.User },
    timestamp: 0,
    content: [{ kind: "text", text }],
});

// ---------------------------------------------------------------------------
// Shape & registration
// ---------------------------------------------------------------------------

test("eventTools exposes exactly the read-only transcript_recall tool", () => {
    const store = freshStore();
    const tools = eventTools(store);
    assert.deepEqual(
        tools.map((t) => t.name),
        ["transcript_recall"],
    );
    // Read-only by contract: no forget/delete counterpart.
    assert.ok(!tools.some((t) => /forget|delete/.test(t.name)));
    store.close();
});

// ---------------------------------------------------------------------------
// Recency (no query)
// ---------------------------------------------------------------------------

test("recall with no query returns recent events, newest first", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "first", ts: 1 });
    store.append({ kind: "message", role: "agent", content: "second", ts: 2 });
    store.append({ kind: "message", role: "user", content: "third", ts: 3 });

    const recall = tool(eventTools(store), "transcript_recall");
    const res = (await recall.run({})) as { count: number; events: { content: string }[] };

    assert.equal(res.count, 3);
    assert.deepEqual(
        res.events.map((e) => e.content),
        ["third", "second", "first"],
    );
    store.close();
});

test("limit bounds the result; default applies when omitted", async () => {
    const store = freshStore();
    for (let i = 0; i < DEFAULT_TRANSCRIPT_LIMIT + 5; i++) {
        store.append({ kind: "message", role: "user", content: `m${i}`, ts: i + 1 });
    }
    const recall = tool(eventTools(store), "transcript_recall");

    const dflt = (await recall.run({})) as { count: number };
    assert.equal(dflt.count, DEFAULT_TRANSCRIPT_LIMIT);

    const three = (await recall.run({ limit: 3 })) as { count: number };
    assert.equal(three.count, 3);
    store.close();
});

// ---------------------------------------------------------------------------
// Session scoping
// ---------------------------------------------------------------------------

test("recall is scoped to the configured session by default", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "mine", session: "s_me", ts: 1 });
    store.append({ kind: "message", role: "user", content: "theirs", session: "s_other", ts: 2 });

    const recall = tool(eventTools(store, { sessionId: "s_me" }), "transcript_recall");
    const res = (await recall.run({})) as { count: number; events: { content: string }[] };

    assert.equal(res.count, 1);
    assert.equal(res.events[0].content, "mine");
    store.close();
});

test("all_sessions widens recall past the configured session", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "mine", session: "s_me", ts: 1 });
    store.append({ kind: "message", role: "user", content: "theirs", session: "s_other", ts: 2 });

    const recall = tool(eventTools(store, { sessionId: "s_me" }), "transcript_recall");
    const res = (await recall.run({ all_sessions: true })) as {
        count: number;
        events: { content: string }[];
    };

    assert.equal(res.count, 2);
    assert.deepEqual(new Set(res.events.map((e) => e.content)), new Set(["mine", "theirs"]));
    store.close();
});

test("a bare log (no sessionId) recalls across everything", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "a", session: "s1", ts: 1 });
    store.append({ kind: "message", role: "user", content: "b", session: "s2", ts: 2 });

    const recall = tool(eventTools(store), "transcript_recall");
    const res = (await recall.run({})) as { count: number };
    assert.equal(res.count, 2);
    store.close();
});

// ---------------------------------------------------------------------------
// Filters: kind, since/until
// ---------------------------------------------------------------------------

test("kind filter narrows to one event kind", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "a question", ts: 1 });
    store.append({ kind: "tool_call", role: "agent", content: "memory_save", ts: 2 });
    store.append({ kind: "tool_result", role: "tool", content: "{saved:true}", ts: 3 });

    const recall = tool(eventTools(store), "transcript_recall");
    const res = (await recall.run({ kind: "tool_call" })) as {
        count: number;
        events: { kind: string; content: string }[];
    };
    assert.equal(res.count, 1);
    assert.equal(res.events[0].kind, "tool_call");
    assert.equal(res.events[0].content, "memory_save");
    store.close();
});

test("since/until bound the recall to a time window", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "old", ts: 10 });
    store.append({ kind: "message", role: "user", content: "mid", ts: 20 });
    store.append({ kind: "message", role: "user", content: "new", ts: 30 });

    const recall = tool(eventTools(store), "transcript_recall");
    const res = (await recall.run({ since: 15, until: 25 })) as {
        count: number;
        events: { content: string }[];
    };
    assert.equal(res.count, 1);
    assert.equal(res.events[0].content, "mid");
    store.close();
});

test("a non-finite time bound is ignored rather than failing the call", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "x", ts: 1 });
    const recall = tool(eventTools(store), "transcript_recall");
    // NaN/Infinity slip past JSON as numbers a careless model might send; asNumber
    // drops them so the query runs unfiltered instead of throwing EventError.
    const res = (await recall.run({ since: Number.NaN })) as { count: number; error?: string };
    assert.equal(res.error, undefined);
    assert.equal(res.count, 1);
    store.close();
});

// ---------------------------------------------------------------------------
// Lexical recall (query, no embedder)
// ---------------------------------------------------------------------------

test("a query ranks by lexical (FTS) relevance", async () => {
    const store = freshStore();
    store.append({
        kind: "message",
        role: "user",
        content: "the deployment pipeline failed",
        ts: 1,
    });
    store.append({ kind: "message", role: "user", content: "lunch plans for friday", ts: 2 });

    const recall = tool(eventTools(store), "transcript_recall");
    const res = (await recall.run({ query: "deployment" })) as {
        count: number;
        events: { content: string }[];
    };
    assert.equal(res.count, 1);
    assert.match(res.events[0].content, /deployment pipeline/);
    store.close();
});

test("a query with no lexical hits falls back to recency", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "alpha", ts: 1 });
    store.append({ kind: "message", role: "user", content: "beta", ts: 2 });

    const recall = tool(eventTools(store), "transcript_recall");
    // "zeta" matches nothing; recall degrades to recent rather than empty.
    const res = (await recall.run({ query: "zeta" })) as {
        count: number;
        events: { content: string }[];
    };
    assert.equal(res.count, 2);
    assert.equal(res.events[0].content, "beta"); // newest first
    store.close();
});

// ---------------------------------------------------------------------------
// Semantic recall (query + embedder)
// ---------------------------------------------------------------------------

test("with an embedder, recall ranks embedded events by meaning", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({
        "shipping a release": [1, 0],
        "what's for dinner": [0, 1],
        deploy: [1, 0.05], // close to the release event, far from dinner
    });

    const a = store.append({ kind: "message", role: "user", content: "shipping a release", ts: 1 });
    const b = store.append({ kind: "message", role: "user", content: "what's for dinner", ts: 2 });
    // Embed both so they're visible to semantic search (the index is selective).
    const [va, vb] = await embedder.embed(["shipping a release", "what's for dinner"]);
    store.setEmbedding(a.id, va);
    store.setEmbedding(b.id, vb);

    const recall = tool(eventTools(store, { embedder }), "transcript_recall");
    const res = (await recall.run({ query: "deploy" })) as {
        count: number;
        events: { content: string }[];
    };
    assert.equal(res.events[0].content, "shipping a release");
    store.close();
});

test("a failing embedder degrades to lexical recall, never erroring", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({}, { fail: true });
    store.append({ kind: "message", role: "user", content: "deployment notes", ts: 1 });
    store.append({ kind: "message", role: "user", content: "grocery list", ts: 2 });

    const recall = tool(eventTools(store, { embedder }), "transcript_recall");
    const res = (await recall.run({ query: "deployment" })) as {
        count: number;
        error?: string;
        events: { content: string }[];
    };
    assert.equal(res.error, undefined);
    assert.equal(res.count, 1);
    assert.match(res.events[0].content, /deployment/);
    store.close();
});

// ---------------------------------------------------------------------------
// Content cap
// ---------------------------------------------------------------------------

test("an oversized event content is truncated in the view", async () => {
    const store = freshStore();
    const huge = "x".repeat(5_000);
    store.append({ kind: "tool_result", role: "tool", content: huge, ts: 1 });

    const recall = tool(eventTools(store), "transcript_recall");
    const res = (await recall.run({})) as { events: { content: string }[] };
    const got = res.events[0].content;
    assert.ok(got.length < huge.length, "content should be trimmed");
    assert.match(got, /…\[truncated\]$/);
    store.close();
});

// ---------------------------------------------------------------------------
// Dispatchable through the loop
// ---------------------------------------------------------------------------

test("transcript_recall is dispatchable through runLoop and results flow back", async () => {
    const store = freshStore();
    store.append({ kind: "message", role: "user", content: "we chose sqlite for storage", ts: 1 });

    const tools = eventTools(store);
    // The model calls the tool, then (next turn) replies with text.
    const client = new FakeClient([
        callTurn("c1", "transcript_recall", { query: "storage" }),
        textTurn("we used sqlite"),
    ]);

    const result = await runLoop(client, {
        messages: [user("what storage did we pick?")],
        tools,
    });

    // The tool result turn carries our recalled event back to the model.
    const toolResults = result.messages
        .flatMap((m) => m.content)
        .filter((p): p is ToolResultPart => p.kind === "tool_result");
    assert.equal(toolResults.length, 1);
    const payload = toolResults[0].result as { count: number; events: { content: string }[] };
    assert.equal(payload.count, 1);
    assert.match(payload.events[0].content, /sqlite/);
    store.close();
});

// ---------------------------------------------------------------------------
// embedEventIfPossible: embed one event on append
// ---------------------------------------------------------------------------

test("embedEventIfPossible stores a vector and makes the event semantically visible", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({ "shipping a release": [1, 0], deploy: [1, 0.05] });
    const e = store.append({ kind: "message", role: "user", content: "shipping a release", ts: 1 });

    assert.ok(!store.hasEmbedding(e.id), "should start with no vector");
    const ok = await embedEventIfPossible(store, embedder, e);
    assert.equal(ok, true);
    assert.ok(store.hasEmbedding(e.id), "vector was not stored");

    const [qv] = await embedder.embed(["deploy"]);
    const hits = store.semanticSearch(qv);
    assert.equal(hits[0]?.event.content, "shipping a release");
    store.close();
});

test("embedEventIfPossible is a no-op without an embedder", async () => {
    const store = freshStore();
    const e = store.append({ kind: "message", role: "user", content: "x", ts: 1 });
    const ok = await embedEventIfPossible(store, undefined, e);
    assert.equal(ok, false);
    assert.ok(!store.hasEmbedding(e.id));
    store.close();
});

test("embedEventIfPossible swallows an embedding outage rather than throwing", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({}, { fail: true });
    const e = store.append({ kind: "message", role: "user", content: "x", ts: 1 });
    const ok = await embedEventIfPossible(store, embedder, e);
    assert.equal(ok, false);
    assert.ok(!store.hasEmbedding(e.id));
    store.close();
});

test("embedEventIfPossible swallows a write to a closed store", async () => {
    const store = freshStore();
    const e = store.append({ kind: "message", role: "user", content: "x", ts: 1 });
    // A slow embed whose store closes before it resolves: the setEmbedding write
    // throws EventError, which the helper swallows so a background embed can never
    // crash a turn.
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    const slow: Embedder = {
        provider: "fake",
        model: "fake-2d",
        dimensions: 2,
        async embed(texts) {
            await gate;
            return texts.map(() => Float32Array.from([1, 0]));
        },
    };
    const pending = embedEventIfPossible(store, slow, e);
    store.close();
    release();
    const ok = await pending;
    assert.equal(ok, false);
});

// ---------------------------------------------------------------------------
// backfillEventEmbeddings: catch up a log that predates embed-on-append
// ---------------------------------------------------------------------------

test("backfillEventEmbeddings embeds every event missing a vector", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({
        "shipping a release": [1, 0],
        "what's for dinner": [0, 1],
        deploy: [1, 0.05],
    });
    store.append({ kind: "message", role: "user", content: "shipping a release", ts: 1 });
    store.append({ kind: "message", role: "user", content: "what's for dinner", ts: 2 });

    const n = await backfillEventEmbeddings(store, embedder);
    assert.equal(n, 2);

    // Both are now semantically searchable; "deploy" ranks the release first.
    const [qv] = await embedder.embed(["deploy"]);
    const hits = store.semanticSearch(qv);
    assert.equal(hits.length, 2);
    assert.equal(hits[0]?.event.content, "shipping a release");
    store.close();
});

test("backfillEventEmbeddings skips events that already have a vector", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({ a: [1, 0], b: [0, 1] });
    const a = store.append({ kind: "message", role: "user", content: "a", ts: 1 });
    store.append({ kind: "message", role: "user", content: "b", ts: 2 });
    // Pre-embed `a`, so only `b` is on the missing-vector work-list.
    const [va] = await embedder.embed(["a"]);
    store.setEmbedding(a.id, va);

    const n = await backfillEventEmbeddings(store, embedder);
    assert.equal(n, 1, "should embed only the one missing vector");
    store.close();
});

test("backfillEventEmbeddings returns 0 on an embedding outage, leaving rows lexical", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({}, { fail: true });
    const e = store.append({ kind: "message", role: "user", content: "x", ts: 1 });
    const n = await backfillEventEmbeddings(store, embedder);
    assert.equal(n, 0);
    assert.ok(!store.hasEmbedding(e.id));
    store.close();
});

test("backfillEventEmbeddings returns 0 on an empty log", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({});
    assert.equal(await backfillEventEmbeddings(store, embedder), 0);
    store.close();
});

test("backfillEventEmbeddings honors the limit, newest first", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({ old: [1, 0], mid: [0, 1], new: [1, 1] });
    store.append({ kind: "message", role: "user", content: "old", ts: 1 });
    store.append({ kind: "message", role: "user", content: "mid", ts: 2 });
    const newest = store.append({ kind: "message", role: "user", content: "new", ts: 3 });

    const n = await backfillEventEmbeddings(store, embedder, 1);
    assert.equal(n, 1);
    // Newest first: only the most recent event got a vector this pass.
    assert.ok(store.hasEmbedding(newest.id), "newest should be embedded first");
    store.close();
});
