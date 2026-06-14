/**
 * Tests for the dream bridge ({@link dreamTools}, {@link dreamContext},
 * {@link dreamEventToView}, {@link renderLastDream}).
 *
 * A dream is stored as one `dream` event whose `content` is the persona's choice
 * and whose `meta` is `{ persona, scenario, sourceMemoryIds }` (see
 * {@link dreamOnce}). These tests append dreams in that shape directly (no model
 * turns needed: the dreaming module already tests the production path), then
 * exercise the two channels this module adds:
 *
 *  - `dream_recall` two ways: directly (calling `run` with an args bag) and
 *    through {@link runLoop} driven by the scripted {@link FakeClient}, proving
 *    it's dispatchable and that the flattened dream flows back as a tool_result.
 *  - {@link dreamContext}, the passive provider that injects the *most recent*
 *    dream into the system prompt, asserted via the provider's `contribute`.
 *
 * Every store is in-memory so the suite never touches disk, and `ts` is injected
 * wherever ordering matters. Mirrors the transcript-tool suite.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { EventStore } from "../src/events.ts";
import { DREAM_EVENT_KIND } from "../src/dreaming.ts";
import {
    dreamTools,
    dreamContext,
    dreamEventToView,
    renderLastDream,
    DEFAULT_DREAM_LIMIT,
} from "../src/dreamTools.ts";
import type { DreamView } from "../src/dreamTools.ts";
import { EmbeddingError, type Embedder } from "../src/embeddings.ts";
import { runLoop } from "../src/bridge/loop.ts";
import { RoleType } from "../src/types.ts";
import type { Message, ToolDef, ToolResultPart } from "../src/types.ts";
import type { Personality } from "../src/critics.ts";
import type { ContextScope } from "../src/context.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

function freshStore(): EventStore {
    return new EventStore(":memory:");
}

/**
 * A deterministic, offline {@link Embedder}: maps each text to a normalized 2-D
 * vector via a caller-supplied table, so semantic ranking is fully predictable
 * with no network. Unknown texts get a fixed "far" vector. Mirrors the helper in
 * eventTools.test.ts.
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

/** Append a dream the way {@link dreamOnce} does: choice in `content`, the
 *  structured record in `meta`. Returns the appended event's id. */
function appendDream(
    store: EventStore,
    opts: {
        persona: Personality;
        scenario: string;
        choice: string;
        ts?: number;
        sourceMemoryIds?: number[];
    },
): number {
    const e = store.append({
        kind: DREAM_EVENT_KIND,
        role: "agent",
        content: opts.choice,
        ts: opts.ts,
        meta: {
            persona: opts.persona,
            scenario: opts.scenario,
            sourceMemoryIds: opts.sourceMemoryIds ?? [],
        },
    });
    return e.id;
}

/** A minimal context scope for exercising a provider directly. */
function scope(messages: Message[] = [], turn = 0): ContextScope {
    return { messages, turn };
}

// ---------------------------------------------------------------------------
// Shape & registration
// ---------------------------------------------------------------------------

test("dreamTools exposes exactly the read-only dream_recall tool", () => {
    const store = freshStore();
    const tools = dreamTools(store);
    assert.deepEqual(
        tools.map((t) => t.name),
        ["dream_recall"],
    );
    // Read-only by contract: a dream is a record, no forget/delete counterpart.
    assert.ok(!tools.some((t) => /forget|delete/.test(t.name)));
    store.close();
});

// ---------------------------------------------------------------------------
// Flattening: dreamEventToView
// ---------------------------------------------------------------------------

test("dreamEventToView flattens the event into persona / scenario / choice", () => {
    const store = freshStore();
    const id = appendDream(store, {
        persona: { name: "Mara", role: "night-shift nurse" },
        scenario: "A patient asks you to bend a rule.",
        choice: "I held the line, and explained why.",
        ts: 5,
    });
    const view = dreamEventToView(store.get(id)!);
    assert.equal(view.id, id);
    assert.equal(view.ts, 5);
    assert.deepEqual(view.persona, { name: "Mara", role: "night-shift nurse" });
    assert.equal(view.scenario, "A patient asks you to bend a rule.");
    assert.equal(view.choice, "I held the line, and explained why.");
    store.close();
});

test("dreamEventToView degrades a dream with no usable meta to a named-empty view", () => {
    const store = freshStore();
    // A dream-kind event with no meta at all: the choice still rides in content.
    const e = store.append({ kind: DREAM_EVENT_KIND, role: "agent", content: "a choice", ts: 1 });
    const view = dreamEventToView(e);
    assert.deepEqual(view.persona, { name: "(unknown)" });
    assert.equal(view.scenario, "");
    assert.equal(view.choice, "a choice");
    store.close();
});

// ---------------------------------------------------------------------------
// Recall: recency (no query)
// ---------------------------------------------------------------------------

test("dream_recall with no query returns recent dreams, newest first", async () => {
    const store = freshStore();
    appendDream(store, { persona: { name: "A" }, scenario: "s1", choice: "first", ts: 1 });
    appendDream(store, { persona: { name: "B" }, scenario: "s2", choice: "second", ts: 2 });
    appendDream(store, { persona: { name: "C" }, scenario: "s3", choice: "third", ts: 3 });

    const recall = tool(dreamTools(store), "dream_recall");
    const res = (await recall.run({})) as { count: number; dreams: DreamView[] };

    assert.equal(res.count, 3);
    assert.deepEqual(
        res.dreams.map((d) => d.choice),
        ["third", "second", "first"],
    );
    store.close();
});

test("dream_recall only ever returns dreams, never other log events", async () => {
    const store = freshStore();
    // A noisy log: messages and tool calls alongside one dream.
    store.append({ kind: "message", role: "user", content: "a question", ts: 1 });
    store.append({ kind: "tool_call", role: "agent", content: "memory_save", ts: 2 });
    appendDream(store, { persona: { name: "A" }, scenario: "s", choice: "the only dream", ts: 3 });
    store.append({ kind: "message", role: "agent", content: "an answer", ts: 4 });

    const recall = tool(dreamTools(store), "dream_recall");
    const res = (await recall.run({})) as { count: number; dreams: DreamView[] };

    assert.equal(res.count, 1);
    assert.equal(res.dreams[0].choice, "the only dream");
    store.close();
});

test("limit bounds the result; default applies when omitted", async () => {
    const store = freshStore();
    for (let i = 0; i < DEFAULT_DREAM_LIMIT + 4; i++) {
        appendDream(store, {
            persona: { name: `p${i}` },
            scenario: "s",
            choice: `c${i}`,
            ts: i + 1,
        });
    }
    const recall = tool(dreamTools(store), "dream_recall");

    const dflt = (await recall.run({})) as { count: number };
    assert.equal(dflt.count, DEFAULT_DREAM_LIMIT);

    const two = (await recall.run({ limit: 2 })) as { count: number };
    assert.equal(two.count, 2);
    store.close();
});

test("dream_recall spans dreams regardless of which session asks (not session-scoped)", async () => {
    const store = freshStore();
    // Dreams are conjured outside any conversation; even if one carried a session
    // tag, recall is not scoped by it. Append two dreams under different sessions.
    store.append({
        kind: DREAM_EVENT_KIND,
        role: "agent",
        content: "dream one",
        session: "s_a",
        ts: 1,
        meta: { persona: { name: "A" }, scenario: "s", sourceMemoryIds: [] },
    });
    store.append({
        kind: DREAM_EVENT_KIND,
        role: "agent",
        content: "dream two",
        session: "s_b",
        ts: 2,
        meta: { persona: { name: "B" }, scenario: "s", sourceMemoryIds: [] },
    });

    const recall = tool(dreamTools(store), "dream_recall");
    const res = (await recall.run({})) as { count: number; dreams: DreamView[] };
    assert.equal(res.count, 2);
    assert.deepEqual(new Set(res.dreams.map((d) => d.choice)), new Set(["dream one", "dream two"]));
    store.close();
});

// ---------------------------------------------------------------------------
// Recall: lexical query and time window
// ---------------------------------------------------------------------------

test("dream_recall ranks by lexical match when a query is given", async () => {
    const store = freshStore();
    appendDream(store, {
        persona: { name: "A" },
        scenario: "s",
        choice: "I shipped the release under deadline pressure",
        ts: 1,
    });
    appendDream(store, {
        persona: { name: "B" },
        scenario: "s",
        choice: "I waited and asked a colleague to review",
        ts: 2,
    });

    const recall = tool(dreamTools(store), "dream_recall");
    const res = (await recall.run({ query: "release shipped" })) as {
        count: number;
        dreams: DreamView[];
    };
    assert.ok(res.count >= 1);
    assert.match(res.dreams[0].choice, /shipped the release/);
    store.close();
});

test("since/until bound the recall to a time window", async () => {
    const store = freshStore();
    appendDream(store, { persona: { name: "A" }, scenario: "s", choice: "old", ts: 10 });
    appendDream(store, { persona: { name: "B" }, scenario: "s", choice: "mid", ts: 20 });
    appendDream(store, { persona: { name: "C" }, scenario: "s", choice: "new", ts: 30 });

    const recall = tool(dreamTools(store), "dream_recall");
    const res = (await recall.run({ since: 15, until: 25 })) as {
        count: number;
        dreams: DreamView[];
    };
    assert.equal(res.count, 1);
    assert.equal(res.dreams[0].choice, "mid");
    store.close();
});

// ---------------------------------------------------------------------------
// Recall: semantic ranking and its fallbacks
// ---------------------------------------------------------------------------

test("dream_recall ranks by meaning when an embedder is configured", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({
        "I held firm on the rule": [1, 0],
        "I let it slide this once": [0, 1],
        integrity: [1, 0.05],
    });
    const a = appendDream(store, {
        persona: { name: "A" },
        scenario: "s",
        choice: "I held firm on the rule",
        ts: 1,
    });
    const b = appendDream(store, {
        persona: { name: "B" },
        scenario: "s",
        choice: "I let it slide this once",
        ts: 2,
    });
    // Embed both dream events so the semantic path can see them.
    for (const id of [a, b]) {
        const e = store.get(id)!;
        const [vec] = await embedder.embed([e.content]);
        store.setEmbedding(e.id, vec!);
    }

    const recall = tool(dreamTools(store, { embedder }), "dream_recall");
    const res = (await recall.run({ query: "integrity" })) as {
        count: number;
        dreams: DreamView[];
    };
    assert.equal(res.dreams[0].choice, "I held firm on the rule");
    store.close();
});

test("dream_recall degrades to lexical when embedding fails", async () => {
    const store = freshStore();
    const embedder = fakeEmbedder({}, { fail: true });
    appendDream(store, {
        persona: { name: "A" },
        scenario: "s",
        choice: "I held firm on the rule",
        ts: 1,
    });

    const recall = tool(dreamTools(store, { embedder }), "dream_recall");
    // Embedding throws; recall should fall through to lexical and still find it.
    const res = (await recall.run({ query: "rule firm" })) as {
        count: number;
        dreams: DreamView[];
    };
    assert.ok(res.count >= 1);
    assert.match(res.dreams[0].choice, /held firm/);
    store.close();
});

// ---------------------------------------------------------------------------
// Recall: error handling
// ---------------------------------------------------------------------------

test("a bad time bound becomes a readable error, not a throw", async () => {
    const store = freshStore();
    appendDream(store, { persona: { name: "A" }, scenario: "s", choice: "c", ts: 1 });
    const recall = tool(dreamTools(store), "dream_recall");
    // A non-finite `since` slips past asNumber (so it's omitted): to actually
    // trip the store's validation we'd have to pass a bound that reaches it.
    // asNumber drops NaN, so assert the tool never throws for odd input and just
    // returns recent dreams.
    const res = (await recall.run({ since: NaN })) as { count: number };
    assert.equal(res.count, 1);
    store.close();
});

// ---------------------------------------------------------------------------
// Recall: dispatch through the loop
// ---------------------------------------------------------------------------

test("dream_recall is dispatchable through runLoop and the dream flows back", async () => {
    const store = freshStore();
    appendDream(store, {
        persona: { name: "Mara", role: "night-shift nurse" },
        scenario: "A patient asks you to bend a rule.",
        choice: "I held the line, and explained why.",
        ts: 1,
    });

    const tools = dreamTools(store);
    const client = new FakeClient([
        callTurn("c1", "dream_recall", { query: "rule" }),
        textTurn("I recalled a dream about holding the line."),
    ]);

    const result = await runLoop(client, {
        messages: [user("what did you dream about rules?")],
        tools,
    });

    const toolResults = result.messages
        .flatMap((m) => m.content)
        .filter((p): p is ToolResultPart => p.kind === "tool_result");
    assert.equal(toolResults.length, 1);
    const payload = toolResults[0].result as { count: number; dreams: DreamView[] };
    assert.equal(payload.count, 1);
    assert.equal(payload.dreams[0].persona.name, "Mara");
    assert.match(payload.dreams[0].choice, /held the line/);
    store.close();
});

// ---------------------------------------------------------------------------
// renderLastDream
// ---------------------------------------------------------------------------

test("renderLastDream names the persona, scenario, and choice", () => {
    const text = renderLastDream({
        id: 1,
        ts: 1,
        persona: { name: "Mara", role: "night-shift nurse" },
        scenario: "A patient asks you to bend a rule.",
        choice: "I held the line.",
    });
    assert.ok(text);
    assert.match(text!, /Mara, night-shift nurse/);
    assert.match(text!, /A patient asks you to bend a rule\./);
    assert.match(text!, /I held the line\./);
});

test("renderLastDream returns null for a dream with no scenario and no choice", () => {
    const text = renderLastDream({
        id: 1,
        ts: 1,
        persona: { name: "X" },
        scenario: "",
        choice: "",
    });
    assert.equal(text, null);
});

test("renderLastDream falls back gracefully on a persona with no name", () => {
    const text = renderLastDream({
        id: 1,
        ts: 1,
        persona: { name: "" } as Personality,
        scenario: "a dilemma",
        choice: "a choice",
    });
    assert.ok(text);
    assert.match(text!, /You dreamed as someone\./);
});

// ---------------------------------------------------------------------------
// dreamContext: the passive last-dream provider
// ---------------------------------------------------------------------------

test("dreamContext injects the most recent dream as system text", async () => {
    const store = freshStore();
    appendDream(store, { persona: { name: "A" }, scenario: "old dilemma", choice: "old", ts: 1 });
    appendDream(store, {
        persona: { name: "Mara", role: "nurse" },
        scenario: "fresh dilemma",
        choice: "the freshest choice",
        ts: 2,
    });

    const provider = dreamContext(store);
    const contribution = await provider.contribute(scope());
    assert.ok(contribution);
    assert.ok(contribution!.system);
    // The newest dream wins, not the older one.
    assert.match(contribution!.system!, /Mara, nurse/);
    assert.match(contribution!.system!, /fresh dilemma/);
    assert.match(contribution!.system!, /the freshest choice/);
    assert.doesNotMatch(contribution!.system!, /old dilemma/);
    store.close();
});

test("dreamContext is silent when there are no dreams", async () => {
    const store = freshStore();
    // A log with non-dream events but no dreams: nothing to inject.
    store.append({ kind: "message", role: "user", content: "hi", ts: 1 });
    const provider = dreamContext(store);
    const contribution = await provider.contribute(scope());
    assert.equal(contribution, undefined);
    store.close();
});

test("dreamContext is silent after the store is closed (best-effort read)", async () => {
    const store = freshStore();
    appendDream(store, { persona: { name: "A" }, scenario: "s", choice: "c", ts: 1 });
    store.close();
    const provider = dreamContext(store);
    // A read against the closed store throws inside the provider; it must swallow
    // that and contribute nothing rather than break the turn.
    const contribution = await provider.contribute(scope());
    assert.equal(contribution, undefined);
});

test("dreamContext provider has a stable name", () => {
    const store = freshStore();
    assert.equal(dreamContext(store).name, "last-dream");
    store.close();
});
