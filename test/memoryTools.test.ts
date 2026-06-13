/**
 * Tests for the memory tool bridge ({@link memoryTools}, {@link recallContext}).
 *
 * These exercise the tools two ways: directly (calling each `ToolDef.run` with
 * an args bag, the way the loop would) and through {@link runLoop} driven by the
 * scripted {@link FakeClient}, which proves they're actually dispatchable and
 * that results flow back as `tool_result` parts.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { MemoryStore, Memory } from "../src/memory.ts";
import { memoryTools, recallContext, DEFAULT_RECALL_LIMIT } from "../src/memoryTools.ts";
import { EmbeddingError, type Embedder } from "../src/embeddings.ts";
import { runLoop } from "../src/bridge/loop.ts";
import { RoleType } from "../src/types.ts";
import type { Message, ToolDef, ToolResultPart } from "../src/types.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

function freshStore(): MemoryStore {
    return new MemoryStore(":memory:");
}

/**
 * A deterministic, offline {@link Embedder}: it maps each text to a normalized
 * vector via a caller-supplied table, so semantic ranking is fully predictable
 * with no network. Unknown texts get a fixed "far" vector.
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

function toolResults(messages: Message[]): ToolResultPart[] {
    return messages
        .flatMap((m) => m.content)
        .filter((p): p is ToolResultPart => p.kind === "tool_result");
}

// ---------------------------------------------------------------------------
// memory_save
// ---------------------------------------------------------------------------

test("memory_save persists a memory and returns its view", async () => {
    const store = freshStore();
    const save = tool(memoryTools(store), "memory_save");
    const res = (await save.run({
        content: "user prefers dark mode",
        tags: ["pref"],
        importance: 0.8,
    })) as { saved: boolean; memory: { id: number; content: string } };

    assert.equal(res.saved, true);
    assert.ok(res.memory.id > 0);
    assert.equal(store.count(), 1);
    assert.equal(store.get(res.memory.id)?.content, "user prefers dark mode");
    store.close();
});

test("memory_save reports validation failures instead of throwing", async () => {
    const store = freshStore();
    const save = tool(memoryTools(store), "memory_save");
    const res = (await save.run({ content: "   " })) as { saved: boolean; error?: string };
    assert.equal(res.saved, false);
    assert.match(res.error ?? "", /empty/);
    assert.equal(store.count(), 0);
    store.close();
});

test("memory_save ignores a non-array tags value rather than crashing", async () => {
    const store = freshStore();
    const save = tool(memoryTools(store), "memory_save");
    const res = (await save.run({ content: "hi", tags: "not-an-array" })) as { saved: boolean };
    assert.equal(res.saved, true);
    assert.deepEqual(store.all()[0]?.tags, []);
    store.close();
});

// ---------------------------------------------------------------------------
// memory_recall
// ---------------------------------------------------------------------------

test("memory_recall searches by query, most relevant first", async () => {
    const store = freshStore();
    store.save(new Memory({ content: "likes oat milk", importance: 0.2 }));
    store.save(new Memory({ content: "allergic to oat", importance: 0.9 }));
    store.save(new Memory({ content: "unrelated" }));

    const recall = tool(memoryTools(store), "memory_recall");
    const res = (await recall.run({ query: "oat" })) as {
        count: number;
        memories: { content: string }[];
    };
    assert.equal(res.count, 2);
    assert.equal(res.memories[0].content, "allergic to oat"); // higher importance first
    store.close();
});

test("memory_recall with no query lists recent/important memories", async () => {
    const store = freshStore();
    store.save(new Memory({ content: "a" }));
    store.save(new Memory({ content: "b" }));
    const recall = tool(memoryTools(store), "memory_recall");
    const res = (await recall.run({})) as { count: number };
    assert.equal(res.count, 2);
    store.close();
});

test("memory_recall filters by tags", async () => {
    const store = freshStore();
    store.save(new Memory({ content: "work thing", tags: ["work"] }));
    store.save(new Memory({ content: "home thing", tags: ["home"] }));
    const recall = tool(memoryTools(store), "memory_recall");
    const res = (await recall.run({ tags: ["work"] })) as { memories: { content: string }[] };
    assert.deepEqual(
        res.memories.map((m) => m.content),
        ["work thing"],
    );
    store.close();
});

// ---------------------------------------------------------------------------
// memory_forget
// ---------------------------------------------------------------------------

test("memory_forget deletes by id and reports the outcome", async () => {
    const store = freshStore();
    const saved = store.save(new Memory({ content: "delete me" }));
    const forget = tool(memoryTools(store), "memory_forget");

    const hit = (await forget.run({ id: saved.id })) as { forgotten: boolean };
    assert.equal(hit.forgotten, true);
    assert.equal(store.count(), 0);

    const miss = (await forget.run({ id: saved.id })) as { forgotten: boolean };
    assert.equal(miss.forgotten, false);
    store.close();
});

test("memory_forget rejects a non-numeric id", async () => {
    const store = freshStore();
    const forget = tool(memoryTools(store), "memory_forget");
    const res = (await forget.run({ id: "nope" })) as { forgotten: boolean; error?: string };
    assert.equal(res.forgotten, false);
    assert.match(res.error ?? "", /finite number/);
    store.close();
});

// ---------------------------------------------------------------------------
// recallContext
// ---------------------------------------------------------------------------

test("recallContext returns null for an empty store", async () => {
    const store = freshStore();
    assert.equal(await recallContext(store), null);
    store.close();
});

test("recallContext renders memories with id and tags", async () => {
    const store = freshStore();
    const saved = store.save(new Memory({ content: "remember this", tags: ["x", "y"] }));
    const text = await recallContext(store);
    assert.ok(text);
    assert.match(text, /Relevant things you remember:/);
    assert.match(text, new RegExp(`#${saved.id}`));
    assert.match(text, /\[x, y\]/);
    assert.match(text, /remember this/);
    store.close();
});

test("recallContext honors its limit (bare-number back-compat form)", async () => {
    const store = freshStore();
    for (let i = 0; i < DEFAULT_RECALL_LIMIT + 5; i++) {
        store.save(new Memory({ content: `m${i}`, created: i }));
    }
    const text = await recallContext(store, 3);
    assert.ok(text);
    assert.equal(text.split("\n").length - 1, 3); // 1 header line + 3 bullets
    store.close();
});

test("recallContext with a query surfaces turn-relevant, not just important, memories", async () => {
    const store = freshStore();
    // Higher importance, but unrelated to the turn.
    store.save(new Memory({ content: "the office wifi password is hunter2", importance: 0.95 }));
    // Lower importance, but exactly what the turn is about. Porter stemming
    // links "allergies" here to "allergy" in the query.
    store.save(new Memory({ content: "user has several food allergies", importance: 0.2 }));

    // Relevance must win over importance: the allergy note ranks above the more
    // "important" wifi note. (Stopword-ish shared tokens may still pull the wifi
    // note in; the contract is ordering, so assert the allergy line comes first.)
    const text = await recallContext(store, { query: "what food allergy should I cook around?" });
    assert.ok(text);
    assert.match(text, /food allergies/);
    const allergyAt = text.indexOf("allergies");
    const wifiAt = text.indexOf("wifi password");
    assert.ok(wifiAt === -1 || allergyAt < wifiAt, "allergy note should rank before wifi note");
    store.close();
});

test("recallContext falls back to importance order when the query matches nothing", async () => {
    const store = freshStore();
    store.save(new Memory({ content: "alpha", importance: 0.3 }));
    store.save(new Memory({ content: "beta", importance: 0.9 }));

    // Query shares no token with any memory: don't return empty: fall back.
    const text = await recallContext(store, { query: "zzz nonexistent terms" });
    assert.ok(text);
    assert.match(text, /beta/); // most important first, since relevance was a wash
    assert.match(text, /alpha/);
    store.close();
});

test("recallContext with no query keeps the old importance-ordered behavior", async () => {
    const store = freshStore();
    store.save(new Memory({ content: "low", importance: 0.1 }));
    store.save(new Memory({ content: "high", importance: 0.9 }));
    const text = await recallContext(store, { limit: 1 });
    assert.ok(text);
    assert.match(text, /high/);
    assert.doesNotMatch(text, /low/);
    store.close();
});

// ---------------------------------------------------------------------------
// Semantic recall (with an embedder)
// ---------------------------------------------------------------------------

test("memory_save embeds the saved memory when an embedder is configured", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({ "user loves espresso": [1, 0] });
    const save = tool(memoryTools(store, embedder), "memory_save");
    const res = (await save.run({ content: "user loves espresso" })) as { memory: { id: number } };
    assert.equal(store.hasEmbedding(res.memory.id), true);
    store.close();
});

test("memory_recall ranks by meaning, beating a lexically-closer but unrelated note", async () => {
    const store = freshStore();
    // Query "what caffeine does the user drink" shares the word "user" with the
    // car note, but *means* the same as the coffee note. Vectors encode that:
    // coffee is near the query direction, car is far.
    const embedder = fakeEmbedder({
        "user enjoys a strong coffee": [1, 0],
        "user drives a fast car": [-1, 0.2],
        "what caffeine does the user drink": [1, 0.05],
    });
    const tools = memoryTools(store, embedder);
    const save = tool(tools, "memory_save");
    await save.run({ content: "user enjoys a strong coffee" });
    await save.run({ content: "user drives a fast car" });

    const recall = tool(tools, "memory_recall");
    const res = (await recall.run({ query: "what caffeine does the user drink" })) as {
        memories: { content: string }[];
    };
    assert.equal(res.memories[0].content, "user enjoys a strong coffee");
    store.close();
});

test("memory_recall falls back to lexical when embedding fails", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({}, { fail: true });
    const tools = memoryTools(store, embedder);
    // Save with a failing embedder: the memory still persists (no vector).
    const save = tool(tools, "memory_save");
    const saved = (await save.run({ content: "the deploy runs on fridays" })) as {
        saved: boolean;
        memory: { id: number };
    };
    assert.equal(saved.saved, true);
    assert.equal(store.hasEmbedding(saved.memory.id), false);

    // Recall still works via FTS even though the embedder throws.
    const recall = tool(tools, "memory_recall");
    const res = (await recall.run({ query: "when does the deploy happen" })) as {
        memories: { content: string }[];
    };
    assert.equal(res.memories[0]?.content, "the deploy runs on fridays");
    store.close();
});

test("recallContext uses semantic ranking when given an embedder", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({
        "user is vegetarian": [1, 0],
        "user owns a sailboat": [-1, 0],
        "what can I cook for the user": [1, 0.1],
    });
    const a = store.save(new Memory({ content: "user is vegetarian" }));
    const b = store.save(new Memory({ content: "user owns a sailboat" }));
    store.setEmbedding(a.id, (await embedder.embed(["user is vegetarian"]))[0]);
    store.setEmbedding(b.id, (await embedder.embed(["user owns a sailboat"]))[0]);

    const text = await recallContext(store, {
        query: "what can I cook for the user",
        embedder,
    });
    assert.ok(text);
    const vegAt = text.indexOf("vegetarian");
    const boatAt = text.indexOf("sailboat");
    assert.ok(vegAt !== -1);
    assert.ok(boatAt === -1 || vegAt < boatAt, "vegetarian note should rank first");
    store.close();
});

// ---------------------------------------------------------------------------
// Integration through the loop
// ---------------------------------------------------------------------------

test("the loop can dispatch memory_save and feed the result back", async () => {
    const store = freshStore();
    const tools = memoryTools(store);
    const client = new FakeClient([
        callTurn("c1", "memory_save", { content: "loop-saved fact", importance: 0.5 }),
        textTurn("done"),
    ]);

    const res = await runLoop(client, { messages: [user("remember something")], tools });

    assert.equal(res.turns, 2);
    assert.equal(store.count(), 1);
    assert.equal(store.all()[0]?.content, "loop-saved fact");

    // The save result came back to the model as a non-error tool_result.
    const [result] = toolResults(res.messages);
    assert.equal(result.callId, "c1");
    assert.notEqual(result.isError, true);
    assert.equal((result.result as { saved: boolean }).saved, true);
    store.close();
});

test("the loop surfaces a save validation failure as a non-error result", async () => {
    const store = freshStore();
    const tools = memoryTools(store);
    const client = new FakeClient([
        callTurn("c1", "memory_save", { content: "  " }),
        textTurn("ok"),
    ]);

    const res = await runLoop(client, { messages: [user("save blank")], tools });

    // The tool handled the bad input itself, so it's a normal result the model
    // can read: not a thrown error that aborts the call.
    const [result] = toolResults(res.messages);
    assert.equal(result.callId, "c1");
    assert.notEqual(result.isError, true);
    assert.equal((result.result as { saved: boolean }).saved, false);
    assert.equal(store.count(), 0);
    store.close();
});
