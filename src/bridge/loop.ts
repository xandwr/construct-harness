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

/** Pull the tool-call parts out of an assistant message, if any. */
function toolCalls(message: Message): ToolCallPart[] {
    return message.content.filter(
        (p): p is ToolCallPart => p.kind === "tool_call",
    );
}

/** Execute one tool call, capturing both success and thrown errors as a
 *  `tool_result` part. A throwing tool becomes an error result the model can
 *  see and react to — it never crashes the loop. */
async function runTool(call: ToolCallPart, tools: Map<string, ToolDef>): Promise<ContentPart> {
    const def = tools.get(call.name);
    if (!def) {
        return {
            kind: "tool_result",
            callId: call.id,
            result: `No such tool: ${call.name}`,
            isError: true,
        };
    }
    try {
        const result = await def.run(call.args);
        return { kind: "tool_result", callId: call.id, result };
    } catch (err) {
        return {
            kind: "tool_result",
            callId: call.id,
            result: err instanceof Error ? err.message : String(err),
            isError: true,
        };
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
export async function runLoop(
    client: ModelClient,
    params: RunLoopParams,
): Promise<RunLoopResult> {
    const maxTurns = params.maxTurns ?? 10;
    const toolIndex = new Map((params.tools ?? []).map((t) => [t.name, t]));

    const messages = [...params.messages];
    let final: GenerateResult | undefined;
    let turns = 0;

    while (turns < maxTurns) {
        const result = await client.generate({ ...params, messages });
        turns++;
        final = result;
        messages.push(result.message);

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
