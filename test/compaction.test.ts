/**
 * Tests for context compaction ({@link compactConversation}).
 *
 * Compaction summarizes older turns through the model itself, so we drive it
 * with the scripted {@link FakeClient}: the next scripted turn is the "summary"
 * the fake model returns. We assert on what gets summarized vs. kept, and — most
 * importantly — that a tool_call is never split from its tool_result.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { compactConversation } from "../src/compaction.ts";
import { RoleType } from "../src/types.ts";
import type { Message } from "../src/types.ts";
import { FakeClient, textTurn } from "./helpers/fakeClient.ts";

const user = (text: string): Message => ({
    sender: { role: RoleType.User },
    timestamp: 0,
    content: [{ kind: "text", text }],
});
const agent = (text: string): Message => ({
    sender: { role: RoleType.Agent },
    timestamp: 0,
    content: [{ kind: "text", text }],
});
const system = (text: string): Message => ({
    sender: { role: RoleType.System },
    timestamp: 0,
    content: [{ kind: "text", text }],
});

/** A back-and-forth of n user/agent message pairs. */
function conversation(n: number): Message[] {
    const msgs: Message[] = [];
    for (let i = 0; i < n; i++) {
        msgs.push(user(`u${i}`), agent(`a${i}`));
    }
    return msgs;
}

test("does nothing when history already fits within keepRecent", async () => {
    const client = new FakeClient([textTurn("SUMMARY")]);
    const msgs = conversation(2); // 4 messages
    const res = await compactConversation(client, msgs, { keepRecent: 8 });

    assert.equal(res.compacted, false);
    assert.equal(res.summarizedCount, 0);
    assert.equal(res.messages, msgs, "input returned unchanged");
    assert.equal(client.calls.length, 0, "no summarization call made");
});

test("summarizes older turns and keeps the recent tail verbatim", async () => {
    const client = new FakeClient([textTurn("CONDENSED HISTORY")]);
    const msgs = conversation(6); // 12 messages
    const res = await compactConversation(client, msgs, { keepRecent: 4 });

    assert.equal(res.compacted, true);
    assert.equal(res.summarizedCount, 8); // 12 - 4 kept
    // Result = summary message + last 4 originals.
    assert.equal(res.messages.length, 5);
    assert.match(textOf(res.messages[0]!), /CONDENSED HISTORY/);
    assert.match(textOf(res.messages[0]!), /Summary of 8 earlier message/);
    // The kept tail is the last 4 messages, untouched.
    assert.deepEqual(res.messages.slice(1), msgs.slice(8));
    assert.ok(res.usage, "summarizer usage reported");
});

test("system messages are preserved and never summarized", async () => {
    const client = new FakeClient([textTurn("SUMMARY")]);
    const msgs = [system("you are X"), ...conversation(6)];
    const res = await compactConversation(client, msgs, { keepRecent: 4 });

    assert.equal(res.compacted, true);
    // System stays at the front, ahead of the summary.
    assert.equal(res.messages[0]!.sender.role, RoleType.System);
    assert.match(textOf(res.messages[0]!), /you are X/);
    assert.match(textOf(res.messages[1]!), /SUMMARY/);

    // The transcript handed to the summarizer must not contain the system text.
    const prompt = client.calls[0]!.messages;
    assert.doesNotMatch(
        prompt.map(textOf).join("\n"),
        /you are X/,
        "system guidance leaked into the summary transcript",
    );
});

test("never orphans a tool_result from its tool_call across the boundary", async () => {
    // Build: u0,a0, then an assistant tool_call, then its tool_result, then
    // a recent pair. With keepRecent=2 the naive split would keep the
    // tool_result but summarize its tool_call — an orphan. The boundary must
    // snap back to keep the call too.
    const msgs: Message[] = [
        user("u0"),
        agent("a0"),
        {
            sender: { role: RoleType.Agent },
            timestamp: 0,
            content: [{ kind: "tool_call", id: "c1", name: "f", args: {} }],
        },
        {
            sender: { role: RoleType.User },
            timestamp: 0,
            content: [{ kind: "tool_result", callId: "c1", result: "ok" }],
        },
        user("u-recent"),
    ];
    const client = new FakeClient([textTurn("SUMMARY")]);
    const res = await compactConversation(client, msgs, { keepRecent: 2 });

    assert.equal(res.compacted, true);
    // The kept tail must not begin with an orphaned tool_result.
    const kept = res.messages.slice(1); // drop the summary message
    const first = kept[0]!;
    const startsWithResult = first.content.some((p) => p.kind === "tool_result");
    assert.equal(startsWithResult, false, "kept tail starts with an orphaned tool_result");

    // Verify no tool_result in the kept tail lacks its tool_call in the tail.
    const callIds = new Set(
        kept
            .flatMap((m) => m.content)
            .filter((p) => p.kind === "tool_call")
            .map((p) => p.id),
    );
    for (const m of kept) {
        for (const p of m.content) {
            if (p.kind === "tool_result") {
                assert.ok(callIds.has(p.callId), `orphaned result for ${p.callId}`);
            }
        }
    }
});

test("bails (keeps originals) when the summarizer returns nothing usable", async () => {
    const client = new FakeClient([textTurn("   ")]); // whitespace-only summary
    const msgs = conversation(6);
    const res = await compactConversation(client, msgs, { keepRecent: 4 });

    assert.equal(res.compacted, false);
    assert.equal(res.messages, msgs, "history preserved when summary is empty");
    assert.ok(res.usage, "usage still reported for the wasted call");
});

function textOf(m: Message): string {
    return m.content
        .filter(
            (p): p is Extract<(typeof m.content)[number], { kind: "text" }> => p.kind === "text",
        )
        .map((p) => p.text)
        .join(" ");
}
