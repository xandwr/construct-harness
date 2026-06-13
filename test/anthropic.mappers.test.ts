/**
 * Tests for the pure Core ↔ Anthropic mappers in src/bridge/anthropic.ts.
 *
 * These cross the wire format both ways and are the second half of the "agent
 * tool stuff": if a tool_call or tool_result is mapped wrong, the loop's
 * id-correlation breaks even though the loop logic is fine. No SDK calls here:
 * every function under test is pure and exported for exactly this reason.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
    toAnthropicMessages,
    toAnthropicTools,
    fromAnthropicMessage,
    toStopReason,
    stringifyResult,
    ANTHROPIC_CAPABILITIES,
} from "../src/bridge/anthropic.ts";
import { RoleType } from "../src/types.ts";
import type { Message, ToolDef } from "../src/types.ts";

// ── Core → Anthropic: messages ──────────────────────────────────────────────

test("lifts system turns out of the message array and concatenates them", () => {
    const messages: Message[] = [
        { sender: { role: RoleType.System }, timestamp: 0, content: [{ kind: "text", text: "A" }] },
        { sender: { role: RoleType.System }, timestamp: 0, content: [{ kind: "text", text: "B" }] },
        { sender: { role: RoleType.User }, timestamp: 0, content: [{ kind: "text", text: "hi" }] },
    ];

    const { system, messages: out } = toAnthropicMessages(messages);

    assert.equal(system, "A\n\nB");
    assert.equal(out.length, 1, "system turns must not appear in the message array");
    assert.equal(out[0]!.role, "user");
});

test("system is undefined when there are no system turns", () => {
    const { system } = toAnthropicMessages([
        { sender: { role: RoleType.User }, timestamp: 0, content: [{ kind: "text", text: "hi" }] },
    ]);
    assert.equal(system, undefined);
});

test("maps the agent role to assistant and everything else to user", () => {
    const { messages } = toAnthropicMessages([
        { sender: { role: RoleType.Agent }, timestamp: 0, content: [{ kind: "text", text: "a" }] },
        { sender: { role: RoleType.User }, timestamp: 0, content: [{ kind: "text", text: "u" }] },
        { sender: { role: RoleType.Tool }, timestamp: 0, content: [{ kind: "text", text: "t" }] },
    ]);
    assert.deepEqual(
        messages.map((m) => m.role),
        ["assistant", "user", "user"],
    );
});

test("maps a tool_call part to an Anthropic tool_use block", () => {
    const { messages } = toAnthropicMessages([
        {
            sender: { role: RoleType.Agent },
            timestamp: 0,
            content: [
                { kind: "tool_call", id: "c1", name: "get_weather", args: { city: "Dublin" } },
            ],
        },
    ]);
    const block = (messages[0]!.content as any[])[0];
    assert.equal(block.type, "tool_use");
    assert.equal(block.id, "c1");
    assert.equal(block.name, "get_weather");
    assert.deepEqual(block.input, { city: "Dublin" });
});

test("tool_call with null/undefined args becomes an empty input object", () => {
    const { messages } = toAnthropicMessages([
        {
            sender: { role: RoleType.Agent },
            timestamp: 0,
            content: [{ kind: "tool_call", id: "c1", name: "noop", args: undefined }],
        },
    ]);
    const block = (messages[0]!.content as any[])[0];
    assert.deepEqual(block.input, {});
});

test("maps a tool_result with object payload to a JSON-stringified block", () => {
    const { messages } = toAnthropicMessages([
        {
            sender: { role: RoleType.User },
            timestamp: 0,
            content: [{ kind: "tool_result", callId: "c1", result: { tempC: 14 } }],
        },
    ]);
    const block = (messages[0]!.content as any[])[0];
    assert.equal(block.type, "tool_result");
    assert.equal(block.tool_use_id, "c1");
    assert.equal(block.content, '{"tempC":14}');
});

test("maps a string tool_result without re-stringifying it", () => {
    const { messages } = toAnthropicMessages([
        {
            sender: { role: RoleType.User },
            timestamp: 0,
            content: [{ kind: "tool_result", callId: "c1", result: "plain text", isError: true }],
        },
    ]);
    const block = (messages[0]!.content as any[])[0];
    assert.equal(block.content, "plain text", "string results must not be JSON-quoted");
    assert.equal(block.is_error, true);
});

// ── Core → Anthropic: tools ─────────────────────────────────────────────────

test("toAnthropicTools maps name/description/schema and drops run", () => {
    const tool: ToolDef = {
        name: "t",
        description: "d",
        parameters: { type: "object", properties: { x: { type: "number" } } },
        async run() {
            return null;
        },
    };
    const out = toAnthropicTools([tool])!;
    assert.equal(out.length, 1);
    assert.equal(out[0]!.name, "t");
    assert.equal(out[0]!.description, "d");
    assert.deepEqual(out[0]!.input_schema, tool.parameters);
    assert.equal("run" in out[0]!, false, "the harness-owned run must never cross the wire");
});

test("toAnthropicTools returns undefined for empty or missing tool lists", () => {
    assert.equal(toAnthropicTools(undefined), undefined);
    assert.equal(toAnthropicTools([]), undefined);
});

// ── stringifyResult ─────────────────────────────────────────────────────────

test("stringifyResult passes strings through and JSON-encodes objects", () => {
    assert.equal(stringifyResult("hi"), "hi");
    assert.equal(stringifyResult({ a: 1 }), '{"a":1}');
    assert.equal(stringifyResult([1, 2]), "[1,2]");
});

test("stringifyResult survives a circular object instead of throwing", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    const out = stringifyResult(circular);
    assert.match(out, /\[unserializable tool result:/);
});

test("stringifyResult survives a throwing toJSON", () => {
    const hostile = {
        toJSON() {
            throw new Error("nope");
        },
    };
    assert.match(stringifyResult(hostile), /\[unserializable tool result: nope\]/);
});

test("stringifyResult falls back to String() for undefined", () => {
    assert.equal(stringifyResult(undefined), "undefined");
});

// ── Anthropic → Core ────────────────────────────────────────────────────────

test("fromAnthropicMessage maps text and tool_use blocks, drops thinking", () => {
    const msg: any = {
        model: "claude-test",
        content: [
            { type: "thinking", thinking: "hmm" },
            { type: "text", text: "hello" },
            { type: "tool_use", id: "c1", name: "echo", input: { a: 1 } },
        ],
    };
    const core = fromAnthropicMessage(msg);
    assert.equal(core.sender.role, RoleType.Agent);
    assert.equal(core.sender.name, "claude-test");
    assert.equal(core.content.length, 2, "thinking block must be dropped");
    assert.deepEqual(core.content[0], { kind: "text", text: "hello" });
    assert.deepEqual(core.content[1], {
        kind: "tool_call",
        id: "c1",
        name: "echo",
        args: { a: 1 },
    });
});

test("toStopReason normalizes every known reason and falls back to other", () => {
    assert.equal(toStopReason("end_turn"), "end_turn");
    assert.equal(toStopReason("stop_sequence"), "end_turn");
    assert.equal(toStopReason("tool_use"), "tool_use");
    assert.equal(toStopReason("max_tokens"), "max_tokens");
    assert.equal(toStopReason("refusal"), "refusal");
    assert.equal(toStopReason("pause_turn" as any), "other");
    assert.equal(toStopReason(null), "other");
});

// ── Round trip ──────────────────────────────────────────────────────────────

test("a tool_call round-trips Anthropic → core → Anthropic with id intact", () => {
    const anthropicMsg: any = {
        model: "m",
        content: [{ type: "tool_use", id: "call_xyz", name: "echo", input: { k: "v" } }],
    };
    const core = fromAnthropicMessage(anthropicMsg);
    const { messages } = toAnthropicMessages([core]);
    const block = (messages[0]!.content as any[])[0];
    assert.equal(block.type, "tool_use");
    assert.equal(block.id, "call_xyz", "id must survive the round trip for result correlation");
    assert.deepEqual(block.input, { k: "v" });
});

// ── Capability honesty ──────────────────────────────────────────────────────

test("advertised serverTools capability matches what toAnthropicTools emits", () => {
    const tools: ToolDef[] = [
        {
            name: "echo",
            description: "echoes",
            parameters: { type: "object", properties: {} },
            run: async (x) => x,
        },
    ];
    const mapped = toAnthropicTools(tools) ?? [];

    // A server tool is a typed block (e.g. { type: "web_search_20250305" });
    // a custom tool has no `type`. The client only ever emits custom tools, so
    // the capability flag must not claim otherwise.
    const emitsServerTool = mapped.some((t) => "type" in (t as Record<string, unknown>));
    assert.equal(emitsServerTool, false, "no server-tool block is emitted");
    assert.equal(
        ANTHROPIC_CAPABILITIES.serverTools,
        false,
        "serverTools must stay false while only custom tools are emitted",
    );
});
