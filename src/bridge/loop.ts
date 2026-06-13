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
import { applyContext } from "../context.ts";
import type { ContextProvider } from "../context.ts";
import { compactConversation } from "../compaction.ts";
import type { CompactionOptions } from "../compaction.ts";
import { UsageTracker, estimateTokens } from "../usage.ts";
import type { CumulativeUsage } from "../usage.ts";
import type { GenerateParams, GenerateResult, ModelClient } from "./types.ts";

export interface RunLoopParams extends GenerateParams {
    /** Hard cap on model turns, to bound runaway tool loops. Default 10. */
    maxTurns?: number;
    /**
     * Passive context providers, evaluated just before every `generate` call.
     * Their contributions (current date/time, standing reminders, …) are folded
     * onto the outgoing messages for that turn only — recomputed each turn so
     * temporal values stay current and never leaking into the conversation
     * history this returns. See {@link applyContext}.
     */
    context?: ContextProvider[];
    /**
     * Auto-compaction: keep a long-lived conversation under the context window
     * by summarizing older turns once the estimated size crosses a threshold.
     * Omit to disable (the loop never compacts on its own). See
     * {@link CompactionConfig}.
     *
     * The gate runs *before* each `generate`, on the persistent conversation
     * (not the per-turn context fold), so a compacted history carries forward to
     * every subsequent turn — which is what an interactive session needs.
     */
    compaction?: CompactionConfig;
}

/** Controls the loop's auto-compaction gate. */
export interface CompactionConfig extends CompactionOptions {
    /**
     * Estimated-token threshold that triggers compaction before a turn. When the
     * running estimate of the conversation (see {@link estimateTokens}) exceeds
     * this, the loop summarizes older turns before calling the model. There is
     * no default — providing this object is what turns compaction on, and the
     * threshold should be set below the model's real context window with headroom
     * for the next turn's output.
     */
    thresholdTokens: number;
}

export interface RunLoopResult {
    /** The full conversation, including every assistant turn and tool result.
     *  If compaction ran, older turns appear here as their summary. */
    messages: Message[];
    /** The final model result (the turn that stopped without a tool call). */
    final: GenerateResult;
    /** Number of model turns actually taken. */
    turns: number;
    /** Token totals across every turn of this run, including any summarization
     *  turns compaction performed. */
    usage: CumulativeUsage;
    /**
     * True when the loop stopped because it hit {@link RunLoopParams.maxTurns}
     * while the model was still requesting tools — i.e. it was cut off, not done.
     * Lets a caller distinguish a completed run from a runaway one without
     * re-deriving it from `final.stopReason`.
     */
    stoppedAtMaxTurns: boolean;
    /** How many times the loop compacted the conversation during this run. */
    compactions: number;
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
    const context = params.context ?? [];

    let messages = [...params.messages];
    const usage = new UsageTracker();
    let final: GenerateResult | undefined;
    let turns = 0;
    let stoppedAtMaxTurns = false;
    let compactions = 0;

    while (turns < maxTurns) {
        // Gate: before sending, compact the persistent conversation if our
        // estimate says it's grown past the threshold. We mutate `messages`
        // itself (not just the outgoing copy) so the compaction carries forward
        // to every later turn — the whole point for a long-lived session. The
        // summarizer's own turn is counted toward this run's usage.
        if (params.compaction && estimateTokens(messages) > params.compaction.thresholdTokens) {
            const compacted = await compactConversation(client, messages, params.compaction);
            if (compacted.usage) usage.add(compacted.usage);
            if (compacted.compacted) {
                messages = compacted.messages;
                compactions++;
            }
        }

        // Fold this turn's passive context (e.g. the current time) onto the
        // outgoing messages only — `messages` itself, and so the conversation we
        // return, stays free of per-turn injected content.
        const outgoing = applyContext(messages, context, turns);
        const result = await client.generate({ ...params, messages: outgoing });
        turns++;
        final = result;
        usage.add(result.usage);
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

        // If that was the last turn we're allowed and the model still wanted
        // tools, we're cutting it off rather than letting it finish. Record that
        // so the caller can tell a runaway from a clean completion.
        if (turns >= maxTurns) stoppedAtMaxTurns = true;
    }

    // `final` is always set: the loop runs at least once (maxTurns ≥ 1).
    return {
        messages,
        final: final!,
        turns,
        usage: usage.totals(),
        stoppedAtMaxTurns,
        compactions,
    };
}
