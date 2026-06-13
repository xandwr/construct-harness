/**
 * Tests for usage accounting and the token estimator ({@link UsageTracker},
 * {@link estimateTokens}).
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { UsageTracker, estimateTokens } from "../src/usage.ts";
import { RoleType } from "../src/types.ts";
import type { Message } from "../src/types.ts";

const textMsg = (role: RoleType, text: string): Message => ({
    sender: { role },
    timestamp: 0,
    content: [{ kind: "text", text }],
});

test("UsageTracker sums input/output/cache across turns", () => {
    const t = new UsageTracker();
    t.add({ inputTokens: 10, outputTokens: 5, cacheReadTokens: 2 });
    t.add({ inputTokens: 3, outputTokens: 7 });

    assert.deepEqual(t.totals(), {
        inputTokens: 13,
        outputTokens: 12,
        cacheReadTokens: 2,
        turns: 2,
    });
});

test("UsageTracker treats missing fields as zero", () => {
    const t = new UsageTracker();
    t.add({}); // a provider that reported nothing
    assert.deepEqual(t.totals(), {
        inputTokens: 0,
        outputTokens: 0,
        cacheReadTokens: 0,
        turns: 1,
    });
});

test("totals() returns a snapshot that can't mutate tracker state", () => {
    const t = new UsageTracker();
    t.add({ inputTokens: 1, outputTokens: 1 });
    const snap = t.totals();
    snap.inputTokens = 999;
    assert.equal(t.totals().inputTokens, 1, "mutating the snapshot changed the tracker");
});

test("estimateTokens grows with conversation size", () => {
    const small = [textMsg(RoleType.User, "hi")];
    const big = [textMsg(RoleType.User, "x".repeat(4000))];
    assert.ok(estimateTokens(big) > estimateTokens(small));
});

test("estimateTokens counts tool-call and tool-result parts", () => {
    const withTool: Message[] = [
        {
            sender: { role: RoleType.Agent },
            timestamp: 0,
            content: [{ kind: "tool_call", id: "c1", name: "search", args: { q: "weather" } }],
        },
        {
            sender: { role: RoleType.User },
            timestamp: 0,
            content: [{ kind: "tool_result", callId: "c1", result: { temp: 14 } }],
        },
    ];
    // Both parts contribute beyond bare per-message overhead.
    assert.ok(estimateTokens(withTool) > 2 * 4);
});

test("estimateTokens errs high: over-counts relative to chars/4", () => {
    // The gate wants to fire early, so the estimate should exceed the common
    // ~4-chars/token rule of thumb for plain prose.
    const text = "a".repeat(400);
    const est = estimateTokens([textMsg(RoleType.User, text)]);
    assert.ok(est >= 400 / 4, `estimate ${est} should be >= ${400 / 4}`);
});

test("estimateTokens tolerates an unserializable tool result", () => {
    const circular: { self?: unknown } = {};
    circular.self = circular;
    const msgs: Message[] = [
        {
            sender: { role: RoleType.User },
            timestamp: 0,
            content: [{ kind: "tool_result", callId: "c1", result: circular }],
        },
    ];
    assert.doesNotThrow(() => estimateTokens(msgs));
});
