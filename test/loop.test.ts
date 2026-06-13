/**
 * Tests for the agentic loop driver ({@link runLoop}).
 *
 * The loop is the heart of the "agent tool stuff": it decides when to run tools,
 * how to feed results back, and when to stop. We drive it with a scripted
 * {@link FakeClient} so every test is deterministic and network-free.
 *
 * Some tests below document *current* behavior that we consider a robustness gap
 * (marked GAP:); they're written so that hardening the loop later is a tightening
 * of the assertion, not a surprise.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { runLoop } from "../src/bridge/loop.ts";
import type { ContextProvider } from "../src/context.ts";
import { RoleType } from "../src/types.ts";
import type { Message, ToolDef, ToolResultPart, ContentPart } from "../src/types.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

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

/** A tool that echoes its args and records every invocation. */
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

/** Pull the tool_result parts out of the appended user turns. */
function toolResults(messages: Message[]): ToolResultPart[] {
    return messages
        .flatMap((m) => m.content)
        .filter((p): p is ToolResultPart => p.kind === "tool_result");
}

test("returns immediately when the model emits no tool calls", async () => {
    const client = new FakeClient([textTurn("hello")]);
    const res = await runLoop(client, { messages: [user("hi")] });

    assert.equal(res.turns, 1);
    assert.equal(res.final.stopReason, "end_turn");
    assert.equal(client.calls.length, 1);
    // Conversation = original user turn + the one assistant turn.
    assert.equal(res.messages.length, 2);
});

test("runs a tool, feeds the result back, and finishes on the next turn", async () => {
    const tool = spyTool();
    const client = new FakeClient([callTurn("c1", "echo", { x: 1 }), textTurn("done")]);

    const res = await runLoop(client, { messages: [user("go")], tools: [tool] });

    assert.equal(res.turns, 2);
    assert.deepEqual(tool.invocations, [{ x: 1 }]);

    // The second generate call must include the tool_result the loop appended.
    const secondCallMessages = client.calls[1]!.messages;
    const results = toolResults(secondCallMessages);
    assert.equal(results.length, 1);
    assert.equal(results[0]!.callId, "c1");
    assert.deepEqual(results[0]!.result, { echoed: { x: 1 } });
    assert.notEqual(results[0]!.isError, true);
});

test("runs multiple tool calls from one turn in parallel and matches ids", async () => {
    const a = spyTool("a");
    const b = spyTool("b");
    const client = new FakeClient([
        {
            content: [
                { kind: "tool_call", id: "ca", name: "a", args: { n: 1 } },
                { kind: "tool_call", id: "cb", name: "b", args: { n: 2 } },
            ],
            stopReason: "tool_use",
        },
        textTurn("ok"),
    ]);

    const res = await runLoop(client, { messages: [user("go")], tools: [a, b] });

    const results = toolResults(res.messages);
    assert.equal(results.length, 2);
    const byId = new Map(results.map((r) => [r.callId, r.result]));
    assert.deepEqual(byId.get("ca"), { echoed: { n: 1 } });
    assert.deepEqual(byId.get("cb"), { echoed: { n: 2 } });
});

test("an unknown tool name becomes an error result, not a crash", async () => {
    const client = new FakeClient([callTurn("c1", "does_not_exist", {}), textTurn("recovered")]);

    const res = await runLoop(client, { messages: [user("go")], tools: [] });

    const results = toolResults(res.messages);
    assert.equal(results.length, 1);
    assert.equal(results[0]!.isError, true);
    assert.match(String(results[0]!.result), /No such tool: does_not_exist/);
    // The loop kept going and the model recovered.
    assert.equal(res.final.stopReason, "end_turn");
});

test("a throwing tool is captured as an error result the model can see", async () => {
    const boom: ToolDef = {
        name: "boom",
        description: "always throws",
        parameters: { type: "object" },
        async run() {
            throw new Error("kaboom");
        },
    };
    const client = new FakeClient([callTurn("c1", "boom", {}), textTurn("ok")]);

    const res = await runLoop(client, { messages: [user("go")], tools: [boom] });

    const results = toolResults(res.messages);
    assert.equal(results[0]!.isError, true);
    assert.equal(results[0]!.result, "kaboom");
    assert.equal(res.turns, 2); // loop survived the throw
});

test("a non-Error thrown value is stringified", async () => {
    const tool: ToolDef = {
        name: "weird",
        description: "throws a string",
        parameters: {},
        async run() {
            throw "just a string"; // eslint-disable-line no-throw-literal
        },
    };
    const client = new FakeClient([callTurn("c1", "weird", {}), textTurn("ok")]);

    const res = await runLoop(client, { messages: [user("go")], tools: [tool] });
    const results = toolResults(res.messages);
    assert.equal(results[0]!.isError, true);
    assert.equal(results[0]!.result, "just a string");
});

test("stops at maxTurns even if the model keeps requesting tools", async () => {
    const tool = spyTool();
    // Script more tool-calling turns than maxTurns allows.
    const client = new FakeClient([
        callTurn("c1", "echo", {}),
        callTurn("c2", "echo", {}),
        callTurn("c3", "echo", {}),
    ]);

    const res = await runLoop(client, {
        messages: [user("go")],
        tools: [tool],
        maxTurns: 2,
    });

    assert.equal(res.turns, 2);
    assert.equal(client.calls.length, 2);
    // final is the 2nd turn, which still requested a tool: caller can detect
    // the runaway via stopReason === "tool_use".
    assert.equal(res.final.stopReason, "tool_use");
});

test("does not mutate the caller's messages array", async () => {
    const tool = spyTool();
    const messages = [user("go")];
    const before = messages.length;
    const client = new FakeClient([callTurn("c1", "echo", {}), textTurn("done")]);

    await runLoop(client, { messages, tools: [tool] });

    assert.equal(messages.length, before, "caller's array was appended to");
});

test("passes tools through to every generate call", async () => {
    const tool = spyTool();
    const client = new FakeClient([callTurn("c1", "echo", {}), textTurn("done")]);

    await runLoop(client, { messages: [user("go")], tools: [tool] });

    for (const call of client.calls) {
        assert.equal(call.tools?.length, 1);
        assert.equal(call.tools?.[0]!.name, "echo");
    }
});

// ── Passive context ──────────────────────────────────────────────────────────

/** Concatenate the system text seen by a given generate call. */
function systemTextOf(messages: Message[]): string {
    return messages
        .filter((m) => m.sender.role === RoleType.System)
        .flatMap((m) => m.content)
        .filter((p): p is Extract<ContentPart, { kind: "text" }> => p.kind === "text")
        .map((p) => p.text)
        .join("\n\n");
}

test("context providers are folded into the system prompt every turn", async () => {
    const tool = spyTool();
    let calls = 0;
    // A provider whose output changes per turn, so we can prove it's recomputed.
    const ticking: ContextProvider = {
        name: "ticking",
        contribute: () => ({ system: `tick ${calls++}` }),
    };
    const client = new FakeClient([callTurn("c1", "echo", {}), textTurn("done")]);

    await runLoop(client, { messages: [user("go")], tools: [tool], context: [ticking] });

    // Two generate calls; each saw a freshly-recomputed system contribution.
    assert.equal(client.calls.length, 2);
    assert.match(systemTextOf(client.calls[0]!.messages), /tick 0/);
    assert.match(systemTextOf(client.calls[1]!.messages), /tick 1/);
});

test("folded context does not leak into the returned conversation", async () => {
    const provider: ContextProvider = {
        name: "p",
        contribute: () => ({ system: "EPHEMERAL-CONTEXT" }),
    };
    const client = new FakeClient([textTurn("hi")]);

    const res = await runLoop(client, { messages: [user("go")], context: [provider] });

    // The wire saw it…
    assert.match(systemTextOf(client.calls[0]!.messages), /EPHEMERAL-CONTEXT/);
    // …but the conversation we return is clean.
    assert.doesNotMatch(systemTextOf(res.messages), /EPHEMERAL-CONTEXT/);
});

test("no context providers leaves the messages untouched", async () => {
    const client = new FakeClient([textTurn("hi")]);
    await runLoop(client, { messages: [user("go")] });
    assert.equal(systemTextOf(client.calls[0]!.messages), "");
});

// ── Truncation handling ─────────────────────────────────────────────────────

test("a max_tokens turn is terminal: its partial tool_call is NOT executed", async () => {
    // A turn truncated by max_tokens may carry a half-emitted tool_call whose
    // args were cut off. The loop must not dispatch it; it stops and lets the
    // caller see stopReason === "max_tokens".
    const tool = spyTool();
    const client = new FakeClient([
        {
            content: [{ kind: "tool_call", id: "c1", name: "echo", args: {} }],
            stopReason: "max_tokens",
        },
        textTurn("unreached"),
    ]);

    const res = await runLoop(client, { messages: [user("go")], tools: [tool] });
    assert.equal(tool.invocations.length, 0, "tool must not run on a truncated turn");
    assert.equal(res.turns, 1);
    assert.equal(res.final.stopReason, "max_tokens");
});

// ── Duplicate tool names ─────────────────────────────────────────────────────

test("duplicate tool names fail loudly at setup", async () => {
    const make = (run: () => Promise<unknown>): ToolDef => ({
        name: "dup",
        description: "d",
        parameters: {},
        run,
    });
    const client = new FakeClient([callTurn("c1", "dup", {}), textTurn("ok")]);

    await assert.rejects(
        runLoop(client, {
            messages: [user("go")],
            tools: [make(async () => "FIRST"), make(async () => "SECOND")],
        }),
        /duplicate tool name "dup"/,
    );
});

// ── Argument validation ──────────────────────────────────────────────────────

test("a call missing a required argument becomes an error result", async () => {
    const tool: ToolDef & { invocations: unknown[] } = {
        name: "needs_city",
        description: "d",
        parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
        },
        invocations: [],
        async run(args) {
            this.invocations.push(args);
            return "ok";
        },
    };
    const client = new FakeClient([callTurn("c1", "needs_city", {}), textTurn("recovered")]);

    const res = await runLoop(client, { messages: [user("go")], tools: [tool] });
    const result = toolResults(res.messages)[0]!;
    assert.equal(result.isError, true);
    assert.match(String(result.result), /missing required argument\(s\): city/);
    assert.equal(tool.invocations.length, 0, "run must not be called with invalid args");
    assert.equal(res.final.stopReason, "end_turn");
});

test("a non-object arg for an object-typed schema is rejected", async () => {
    const tool = spyTool("obj");
    tool.parameters = { type: "object", properties: {} };
    const client = new FakeClient([callTurn("c1", "obj", "not an object"), textTurn("ok")]);

    const res = await runLoop(client, { messages: [user("go")], tools: [tool] });
    const result = toolResults(res.messages)[0]!;
    assert.equal(result.isError, true);
    assert.match(String(result.result), /expected an object of arguments, got string/);
    assert.equal(tool.invocations.length, 0);
});

test("validation lets valid args through to the tool", async () => {
    const tool: ToolDef & { invocations: unknown[] } = {
        name: "needs_city",
        description: "d",
        parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
        },
        invocations: [],
        async run(args) {
            this.invocations.push(args);
            return "ok";
        },
    };
    const client = new FakeClient([
        callTurn("c1", "needs_city", { city: "Dublin" }),
        textTurn("done"),
    ]);

    await runLoop(client, { messages: [user("go")], tools: [tool] });
    assert.deepEqual(tool.invocations, [{ city: "Dublin" }]);
});

test("a tool with no object schema skips validation entirely", async () => {
    const tool = spyTool("loose");
    tool.parameters = {}; // no type:object → nothing to enforce
    const client = new FakeClient([callTurn("c1", "loose", 42), textTurn("done")]);

    await runLoop(client, { messages: [user("go")], tools: [tool] });
    assert.deepEqual(tool.invocations, [42], "non-object schema passes args through unchecked");
});

// ── Cumulative usage & maxTurns signalling ───────────────────────────────────

test("accumulates usage across every turn", async () => {
    const tool = spyTool();
    // Each FakeClient turn reports {input:1, output:1}; two turns → totals of 2.
    const client = new FakeClient([callTurn("c1", "echo", {}), textTurn("done")]);

    const res = await runLoop(client, { messages: [user("go")], tools: [tool] });

    assert.equal(res.usage.turns, 2);
    assert.equal(res.usage.inputTokens, 2);
    assert.equal(res.usage.outputTokens, 2);
});

test("a clean completion does not set stoppedAtMaxTurns", async () => {
    const client = new FakeClient([textTurn("done")]);
    const res = await runLoop(client, { messages: [user("go")] });
    assert.equal(res.stoppedAtMaxTurns, false);
    assert.equal(res.compactions, 0);
});

test("stoppedAtMaxTurns is true when cut off mid tool loop", async () => {
    const tool = spyTool();
    const client = new FakeClient([callTurn("c1", "echo", {}), callTurn("c2", "echo", {})]);
    const res = await runLoop(client, {
        messages: [user("go")],
        tools: [tool],
        maxTurns: 2,
    });
    assert.equal(res.stoppedAtMaxTurns, true, "runaway tool loop should be flagged");
    assert.equal(res.final.stopReason, "tool_use");
});

// ── Auto-compaction gate ─────────────────────────────────────────────────────

test("compacts the conversation before a turn when it exceeds the threshold", async () => {
    // A bulky opening user turn pushes the estimate over a tiny threshold, so
    // the gate fires before the first generate. The first scripted turn is the
    // summary; the second is the real model reply.
    const big = user("x".repeat(5000));
    const client = new FakeClient([textTurn("SUMMARY"), textTurn("answer")]);

    const res = await runLoop(client, {
        messages: [big, user("u1"), agent("a1"), user("u2"), agent("a2"), user("now answer")],
        compaction: { thresholdTokens: 100, keepRecent: 2 },
    });

    assert.equal(res.compactions, 1, "should have compacted once");
    // The summarizer turn + the real reply both count toward usage.
    assert.ok(res.usage.turns >= 2);
    // The conversation the loop carries forward contains the summary, not the
    // bulky original.
    const joined = res.messages
        .flatMap((m) => m.content)
        .filter((p): p is Extract<ContentPart, { kind: "text" }> => p.kind === "text")
        .map((p) => p.text)
        .join(" ");
    assert.match(joined, /SUMMARY/);
    assert.doesNotMatch(joined, /x{5000}/);
});

test("does not compact when under the threshold", async () => {
    const client = new FakeClient([textTurn("done")]);
    const res = await runLoop(client, {
        messages: [user("short")],
        compaction: { thresholdTokens: 1_000_000 },
    });
    assert.equal(res.compactions, 0);
    assert.equal(client.calls.length, 1, "only the real generate, no summarizer call");
});
