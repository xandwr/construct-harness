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

// ── Cancellation: saving the partial stream ──────────────────────────────────

test("a mid-stream abort keeps the partial prose and ends cleanly", async () => {
    // The model streams two text parts, then the user presses "stop" before the
    // turn finishes. The loop must keep the prose already streamed, mark the run
    // cancelled, and still end with exactly one loop_done (not a thrown error).
    const client = new FakeClient([
        {
            content: [
                { kind: "text", text: "First part. " },
                { kind: "text", text: "Second part. " },
                { kind: "text", text: "Third part never streams." },
            ],
            // Abort after the first two text parts, as if "stop" was pressed then.
            abortAfterTextParts: 2,
        },
    ]);
    const { events, done } = await drain(runLoopStream(client, { messages: [user("go")] }));

    // The prose streamed before the abort is delivered as normal text deltas.
    const streamed = events
        .filter((e): e is Extract<LoopEvent, { kind: "text" }> => e.kind === "text")
        .map((e) => e.text)
        .join("");
    assert.equal(streamed, "First part. Second part. ", "partial prose must be kept");

    // Exactly one cancelled event, carrying the same partial text.
    const cancelledEvents = events.filter((e) => e.kind === "cancelled");
    assert.equal(cancelledEvents.length, 1, "one cancelled event");
    assert.equal(
        (cancelledEvents[0] as Extract<LoopEvent, { kind: "cancelled" }>).text,
        "First part. Second part. ",
    );

    // The run reports cancelled and the partial turn is the final result.
    assert.equal(done.result.cancelled, true);
    assert.equal(done.result.final.stopReason, "canceled");
    assert.equal(done.result.turns, 1);
});

test("the cancelled partial turn is committed to the conversation as an agent message", async () => {
    const client = new FakeClient([
        { content: [{ kind: "text", text: "kept prose" }], abortAfterTextParts: 1 },
    ]);
    const { done } = await drain(runLoopStream(client, { messages: [user("go")] }));

    // The last message in the returned conversation is the partial assistant turn,
    // carrying the prose that streamed before the abort (not discarded).
    const last = done.result.messages.at(-1)!;
    assert.equal(last.sender.role, RoleType.Agent);
    const text = last.content.find((p) => p.kind === "text");
    assert.ok(text && text.kind === "text");
    assert.equal(text.text, "kept prose");
});

test("aborting before any prose marks cancelled with empty text", async () => {
    // An already-aborted signal: the stream throws before emitting any delta. The
    // run is still cancelled, with an empty partial reply (nothing to save).
    const controller = new AbortController();
    controller.abort();
    const client = new FakeClient([textTurn("never streams")]);
    const { events, done } = await drain(
        runLoopStream(client, { messages: [user("go")], signal: controller.signal }),
    );

    const cancelledEvents = events.filter((e) => e.kind === "cancelled");
    assert.equal(cancelledEvents.length, 1);
    assert.equal((cancelledEvents[0] as Extract<LoopEvent, { kind: "cancelled" }>).text, "");
    assert.equal(done.result.cancelled, true);
    // No text content on the empty partial turn.
    const last = done.result.messages.at(-1)!;
    assert.equal(last.content.length, 0);
});

test("the abort signal threads through to the model stream", async () => {
    // The loop must forward params.signal to client.stream so a real provider can
    // honor it. FakeClient throws `canceled` when its signal is already aborted;
    // observing the cancelled outcome proves the signal reached the stream call.
    const controller = new AbortController();
    controller.abort();
    const client = new FakeClient([textTurn("x")]);
    const { done } = await drain(
        runLoopStream(client, { messages: [user("go")], signal: controller.signal }),
    );
    assert.equal(done.result.cancelled, true);
});

test("a non-cancel stream error still propagates (not swallowed as a cancel)", async () => {
    // Only a `canceled` HarnessError becomes a partial completion; any other
    // failure must still throw to the caller rather than masquerade as a stop.
    const failing = {
        provider: "boom",
        model: "boom",
        capabilities: {
            thinking: false,
            effort: false,
            promptCaching: false,
            serverTools: false,
            streaming: true as const,
        },
        async generate() {
            throw new Error("unused");
        },
        // eslint-disable-next-line require-yield
        async *stream() {
            throw new Error("network exploded");
        },
    };
    await assert.rejects(
        () => drain(runLoopStream(failing as never, { messages: [user("go")] })),
        /network exploded/,
    );
});
