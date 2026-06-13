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
