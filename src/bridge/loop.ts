/**
 * The agentic loop — a provider-neutral driver over {@link ModelClient}.
 *
 * Given a conversation and a tool set, this runs turns until the model stops
 * requesting tools: each time the model emits `tool_call` parts, the loop
 * executes the matching {@link ToolDef.run}, appends `tool_result` parts as a
 * new user turn, and asks again. It speaks only core types and only the bridge
 * interface, so it works unchanged against any provider.
 */

import { RoleType } from "../types.ts";
import type { ContentPart, Message, ToolDef, ToolCallPart } from "../types.ts";
import type { GenerateParams, GenerateResult, ModelClient } from "./types.ts";

export interface RunLoopParams extends GenerateParams {
    /** Hard cap on model turns, to bound runaway tool loops. Default 10. */
    maxTurns?: number;
}

export interface RunLoopResult {
    /** The full conversation, including every assistant turn and tool result. */
    messages: Message[];
    /** The final model result (the turn that stopped without a tool call). */
    final: GenerateResult;
    /** Number of model turns actually taken. */
    turns: number;
}

/** Index tools by name for O(1) dispatch, rejecting duplicate names.
 *
 *  Two tools sharing a name is a caller bug, not a model bug: a `Map` would
 *  silently keep whichever came last, so the model could call `foo` and reach a
 *  different implementation than the author intended. Fail loudly at setup
 *  instead of mis-dispatching at runtime. */
function indexTools(tools: ToolDef[] | undefined): Map<string, ToolDef> {
    const index = new Map<string, ToolDef>();
    for (const tool of tools ?? []) {
        if (index.has(tool.name)) {
            throw new Error(`runLoop: duplicate tool name "${tool.name}"`);
        }
        index.set(tool.name, tool);
    }
    return index;
}

/** Pull the tool-call parts out of an assistant message, if any. */
function toolCalls(message: Message): ToolCallPart[] {
    return message.content.filter((p): p is ToolCallPart => p.kind === "tool_call");
}

/** Build an error `tool_result` for a call. Centralizes the shape so every
 *  failure path — unknown tool, bad args, thrown tool — looks the same to the
 *  model. */
function errorResult(callId: string, message: string): ContentPart {
    return { kind: "tool_result", callId, result: message, isError: true };
}

/** A lightweight, dependency-free check of a tool call's args against the
 *  top-level shape its JSON Schema declares.
 *
 *  This is deliberately *not* a full JSON Schema validator: it catches the
 *  malformed calls a model actually produces — non-object args where an object
 *  is required, and missing `required` properties — without pulling in a schema
 *  library. Anything it can't reason about (nested types, formats) it lets
 *  through to the tool, which remains the final authority on its own input.
 *  Returns an error string, or `null` when the args are acceptable. */
function validateArgs(schema: unknown, args: unknown): string | null {
    if (typeof schema !== "object" || schema === null) return null;
    const s = schema as { type?: unknown; required?: unknown };
    if (s.type !== "object") return null;

    if (typeof args !== "object" || args === null || Array.isArray(args)) {
        return `expected an object of arguments, got ${args === null ? "null" : Array.isArray(args) ? "array" : typeof args}`;
    }

    if (Array.isArray(s.required)) {
        const present = args as Record<string, unknown>;
        const missing = s.required.filter(
            (key): key is string => typeof key === "string" && !(key in present),
        );
        if (missing.length) {
            return `missing required argument(s): ${missing.join(", ")}`;
        }
    }
    return null;
}

/** Execute one tool call, capturing bad args and thrown errors as a
 *  `tool_result` part. A rejected or throwing tool becomes an error result the
 *  model can see and react to — it never crashes the loop. */
async function runTool(call: ToolCallPart, tools: Map<string, ToolDef>): Promise<ContentPart> {
    const def = tools.get(call.name);
    if (!def) {
        return errorResult(call.id, `No such tool: ${call.name}`);
    }
    const invalid = validateArgs(def.parameters, call.args);
    if (invalid) {
        return errorResult(call.id, `Invalid arguments for ${call.name}: ${invalid}`);
    }
    try {
        const result = await def.run(call.args);
        return { kind: "tool_result", callId: call.id, result };
    } catch (err) {
        return errorResult(call.id, err instanceof Error ? err.message : String(err));
    }
}

/**
 * Drive a model + tools to completion.
 *
 * Each iteration calls {@link ModelClient.generate}, appends the assistant
 * turn, and — if it requested tools — runs them all (in parallel) and appends
 * one user turn of `tool_result`s before looping. Stops when the model returns
 * without tool calls or {@link RunLoopParams.maxTurns} is hit.
 */
export async function runLoop(client: ModelClient, params: RunLoopParams): Promise<RunLoopResult> {
    const maxTurns = params.maxTurns ?? 10;
    const toolIndex = indexTools(params.tools);

    const messages = [...params.messages];
    let final: GenerateResult | undefined;
    let turns = 0;

    while (turns < maxTurns) {
        const result = await client.generate({ ...params, messages });
        turns++;
        final = result;
        messages.push(result.message);

        // Only act on tool calls the provider actually signalled with
        // `tool_use`. A turn truncated by `max_tokens` may *contain* a
        // half-emitted tool_call part whose args were cut off; running it would
        // dispatch a malformed call. Treat anything that isn't a clean
        // `tool_use` stop as a terminal turn and let the caller inspect
        // `final.stopReason`.
        if (result.stopReason !== "tool_use") break;

        const calls = toolCalls(result.message);
        if (calls.length === 0) break;

        const results = await Promise.all(calls.map((c) => runTool(c, toolIndex)));
        messages.push({
            sender: { role: RoleType.User },
            timestamp: Date.now(),
            content: results,
        });
    }

    // `final` is always set: the loop runs at least once (maxTurns ≥ 1).
    return { messages, final: final!, turns };
}
