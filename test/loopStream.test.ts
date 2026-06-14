/**
 * Tests for the streaming loop driver ({@link runLoopStream}).
 *
 * It must reproduce runLoop's control flow exactly: tool dispatch, max-turns
 * cut-off, compaction, usage: while *observing* it through {@link LoopEvent}s.
 * We drive it with the streaming {@link FakeClient} so it's deterministic.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { runLoopStream } from "../src/bridge/loop.ts";
import type { LoopEvent } from "../src/bridge/loop.ts";
import { RoleType } from "../src/types.ts";
import type { Message, ToolDef } from "../src/types.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

const user = (text: string): Message => ({
    sender: { role: RoleType.User },
    timestamp: 0,
    content: [{ kind: "text", text }],
});

function spyTool(name = "echo"): ToolDef & { invocations: unknown[] } {
    const invocations: unknown[] = [];
    return {
        name,
        description: "echoes its input",
        parameters: { type: "object" },
        invocations,
        async run(args) {
            invocations.push(args);
            return { echoed: args };
        },
    };
}

/** Drain a LoopEvent stream into the event list and the terminal result. */
async function drain(
    gen: AsyncGenerator<LoopEvent, void, void>,
): Promise<{ events: LoopEvent[]; done: Extract<LoopEvent, { kind: "loop_done" }> }> {
    const events: LoopEvent[] = [];
    let done: Extract<LoopEvent, { kind: "loop_done" }> | undefined;
    for await (const e of gen) {
        events.push(e);
        if (e.kind === "loop_done") done = e;
    }
    assert.ok(done, "stream must end with loop_done");
    return { events, done };
}

test("streams text deltas and ends with loop_done", async () => {
    const client = new FakeClient([textTurn("hello world")]);
    const { events, done } = await drain(runLoopStream(client, { messages: [user("hi")] }));

    const text = events
        .filter((e): e is Extract<LoopEvent, { kind: "text" }> => e.kind === "text")
        .map((e) => e.text)
        .join("");
    assert.equal(text, "hello world");
    assert.equal(done.result.turns, 1);
    assert.equal(done.result.final.stopReason, "end_turn");
});

test("separates prose that resumes across a tool turn with a blank line", async () => {
    // The model says something, calls a tool, then says more on the next turn.
    // Without a boundary the two prose runs concatenate into one unspaced wall;
    // the loop injects a "\n\n" text delta so any consumer that joins `text`
    // deltas (the web client, the session's assistantText) gets paragraphs.
    const tool = spyTool();
    // Text and a tool call in one assistant turn (like a real reply that says
    // something then acts), followed by a turn of more prose.
    const client = new FakeClient([
        {
            content: [
                { kind: "text", text: "Looking now." },
                { kind: "tool_call", id: "c1", name: "echo", args: {} },
            ],
            stopReason: "tool_use",
        },
        textTurn("Here's what I found."),
    ]);
    const { events } = await drain(
        runLoopStream(client, { messages: [user("go")], tools: [tool] }),
    );

    const joined = events
        .filter((e): e is Extract<LoopEvent, { kind: "text" }> => e.kind === "text")
        .map((e) => e.text)
        .join("");
    assert.equal(
        joined,
        "Looking now.\n\nHere's what I found.",
        "resumed prose must be separated from the earlier prose by a blank line",
    );
});

test("does not insert a break before the very first prose of a run", async () => {
    // A single text-only turn: no earlier prose exists, so no separator.
    const client = new FakeClient([textTurn("hello world")]);
    const { events } = await drain(runLoopStream(client, { messages: [user("hi")] }));
    const firstText = events.find((e) => e.kind === "text");
    assert.ok(firstText && firstText.kind === "text");
    assert.equal(firstText.text, "hello world", "first prose must not be prefixed with a break");
});

test("a tool-only turn between prose does not stack extra breaks", async () => {
    // Turn 1 prose, turn 2 is a pure tool call (no text), turn 3 prose. Only one
    // boundary separates the two prose runs: the break is keyed on text
    // resuming, not on every turn, so an intervening text-less turn adds nothing.
    const tool = spyTool();
    const client = new FakeClient([
        {
            content: [
                { kind: "text", text: "First." },
                { kind: "tool_call", id: "c1", name: "echo", args: {} },
            ],
            stopReason: "tool_use",
        },
        callTurn("c2", "echo", {}),
        textTurn("Second."),
    ]);
    const { events } = await drain(
        runLoopStream(client, { messages: [user("go")], tools: [tool] }),
    );
    const joined = events
        .filter((e): e is Extract<LoopEvent, { kind: "text" }> => e.kind === "text")
        .map((e) => e.text)
        .join("");
    assert.equal(joined, "First.\n\nSecond.");
});

test("runs a tool and emits tool_start/tool_end around it", async () => {
    const tool = spyTool();
    const client = new FakeClient([callTurn("c1", "echo", { x: 1 }), textTurn("done")]);
    const { events, done } = await drain(
        runLoopStream(client, { messages: [user("go")], tools: [tool] }),
    );

    assert.deepEqual(tool.invocations, [{ x: 1 }]);
    const start = events.find((e) => e.kind === "tool_start");
    const end = events.find((e) => e.kind === "tool_end");
    assert.ok(start && start.kind === "tool_start" && start.name === "echo");
    assert.ok(end && end.kind === "tool_end" && end.isError === false);
    assert.equal(done.result.turns, 2);

    // tool_start must precede tool_end in the event order.
    assert.ok(
        events.indexOf(start) < events.indexOf(end),
        "tool_start should be emitted before tool_end",
    );
});

test("a tool error is reported as tool_end with isError", async () => {
    const boom: ToolDef = {
        name: "boom",
        description: "throws",
        parameters: { type: "object" },
        async run() {
            throw new Error("kaboom");
        },
    };
    const client = new FakeClient([callTurn("c1", "boom", {}), textTurn("recovered")]);
    const { events } = await drain(
        runLoopStream(client, { messages: [user("go")], tools: [boom] }),
    );
    const end = events.find((e) => e.kind === "tool_end");
    assert.ok(end && end.kind === "tool_end" && end.isError === true);
});

test("accumulates usage and flags maxTurns cut-off", async () => {
    const tool = spyTool();
    const client = new FakeClient([callTurn("c1", "echo", {}), callTurn("c2", "echo", {})]);
    const { done } = await drain(
        runLoopStream(client, { messages: [user("go")], tools: [tool], maxTurns: 2 }),
    );
    assert.equal(done.result.turns, 2);
    assert.equal(done.result.usage.turns, 2);
    assert.equal(done.result.stoppedAtMaxTurns, true);
});

test("streaming loop clamps maxTurns to at least one model turn", async () => {
    const client = new FakeClient([textTurn("hello")]);
    const { done } = await drain(runLoopStream(client, { messages: [user("hi")], maxTurns: 0 }));

    assert.equal(done.result.turns, 1);
    assert.equal(client.calls.length, 1);
    assert.equal(done.result.final.stopReason, "end_turn");
});

test("streaming loop floors fractional maxTurns", async () => {
    const tool = spyTool();
    const client = new FakeClient([
        callTurn("c1", "echo", {}),
        callTurn("c2", "echo", {}),
        textTurn("unreached"),
    ]);
    const { done } = await drain(
        runLoopStream(client, { messages: [user("go")], tools: [tool], maxTurns: 1.9 }),
    );

    assert.equal(done.result.turns, 1);
    assert.equal(client.calls.length, 1);
    assert.equal(done.result.stoppedAtMaxTurns, true);
});

test("compacts mid-stream and emits a compacted event", async () => {
    const big = user("x".repeat(5000));
    const client = new FakeClient([textTurn("SUMMARY"), textTurn("answer")]);
    const { events, done } = await drain(
        runLoopStream(client, {
            messages: [big, user("u1"), user("u2"), user("now")],
            compaction: { thresholdTokens: 100, keepRecent: 2 },
        }),
    );
    assert.equal(done.result.compactions, 1);
    assert.ok(events.some((e) => e.kind === "compacted"));
});

test("does not surface the model's done delta to the consumer", async () => {
    // The model stream's own `done` is consumed internally; the only terminal
    // event the consumer sees is loop_done.
    const client = new FakeClient([textTurn("hi")]);
    const { events } = await drain(runLoopStream(client, { messages: [user("go")] }));
    const dones = events.filter((e) => e.kind === "done");
    assert.equal(dones.length, 0, "raw model done must not leak to the consumer");
});
