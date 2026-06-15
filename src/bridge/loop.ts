/**
 * The agentic loop: a provider-neutral driver over {@link ModelClient}.
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
import { HarnessError } from "./errors.ts";
import type { CoreDelta, GenerateParams, GenerateResult, ModelClient } from "./types.ts";

export interface RunLoopParams extends GenerateParams {
    /** Hard cap on model turns, to bound runaway tool loops. Default 10. */
    maxTurns?: number;
    /**
     * Passive context providers, evaluated just before every `generate` call.
     * Their contributions (current date/time, standing reminders, …) are folded
     * onto the outgoing messages for that turn only: recomputed each turn so
     * temporal values stay current and never leaking into the conversation
     * history this returns. See {@link applyContext}.
     */
    context?: ContextProvider[];
    /**
     * Epoch-ms the conversation began, forwarded to {@link ContextProvider}s via
     * {@link ContextScope.sessionStart} so a temporal provider can report how long
     * the session has run. Omit for a one-shot run with no session notion.
     */
    sessionStart?: number;
    /**
     * Auto-compaction: keep a long-lived conversation under the context window
     * by summarizing older turns once the estimated size crosses a threshold.
     * Omit to disable (the loop never compacts on its own). See
     * {@link CompactionConfig}.
     *
     * The gate runs *before* each `generate`, on the persistent conversation
     * (not the per-turn context fold), so a compacted history carries forward to
     * every subsequent turn: which is what an interactive session needs.
     */
    compaction?: CompactionConfig;
}

/** Controls the loop's auto-compaction gate. */
export interface CompactionConfig extends CompactionOptions {
    /**
     * Estimated-token threshold that triggers compaction before a turn. When the
     * running estimate of the conversation (see {@link estimateTokens}) exceeds
     * this, the loop summarizes older turns before calling the model. There is
     * no default: providing this object is what turns compaction on, and the
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
     * while the model was still requesting tools: i.e. it was cut off, not done.
     * Lets a caller distinguish a completed run from a runaway one without
     * re-deriving it from `final.stopReason`.
     */
    stoppedAtMaxTurns: boolean;
    /** How many times the loop compacted the conversation during this run. */
    compactions: number;
    /**
     * True when the run ended because the caller aborted {@link RunLoopParams.signal}
     * mid-stream (a user pressing "stop"), rather than the model finishing. The
     * partial assistant turn produced up to the abort is still committed to
     * {@link messages} and is {@link final} (its `stopReason` is `"canceled"`), so
     * the output the model had already streamed is kept, not discarded — the
     * cancellation just ends the turn early. Only the streaming loop sets this; the
     * buffered {@link runLoop} has no mid-flight abort to honor.
     */
    cancelled: boolean;
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
 *  failure path: unknown tool, bad args, thrown tool: looks the same to the
 *  model. */
function errorResult(callId: string, message: string): ContentPart {
    return { kind: "tool_result", callId, result: message, isError: true };
}

/** A lightweight, dependency-free check of a tool call's args against the
 *  top-level shape its JSON Schema declares.
 *
 *  This is deliberately *not* a full JSON Schema validator: it catches the
 *  malformed calls a model actually produces: non-object args where an object
 *  is required, and missing `required` properties: without pulling in a schema
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

/** Normalize the caller's turn cap to the loop invariant: at least one model
 *  turn, whole-numbered. */
function normalizeMaxTurns(maxTurns: number | undefined): number {
    if (maxTurns === undefined || !Number.isFinite(maxTurns)) return 10;
    return Math.max(1, Math.floor(maxTurns));
}

/** Execute one tool call, capturing bad args and thrown errors as a
 *  `tool_result` part. A rejected or throwing tool becomes an error result the
 *  model can see and react to: it never crashes the loop. */
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
 * Run the compaction gate for one iteration: if compaction is configured and the
 * estimate exceeds the threshold, summarize older turns. Returns the (possibly
 * rewritten) message list and whether it actually compacted, folding the
 * summarizer's usage into `usage`. Shared by both the buffered and streaming
 * loops so the gate behaves identically in each.
 */
async function compactGate(
    client: ModelClient,
    messages: Message[],
    params: RunLoopParams,
    usage: UsageTracker,
): Promise<{ messages: Message[]; compacted: boolean }> {
    if (!params.compaction || estimateTokens(messages) <= params.compaction.thresholdTokens) {
        return { messages, compacted: false };
    }
    const compacted = await compactConversation(client, messages, params.compaction);
    if (compacted.usage) usage.add(compacted.usage);
    return {
        messages: compacted.compacted ? compacted.messages : messages,
        compacted: compacted.compacted,
    };
}

/**
 * Run the tools a completed assistant turn requested and build the `user` turn
 * of results to append. Returns null when the turn isn't a clean `tool_use` stop
 * or carries no calls: in which case the loop should terminate. A `max_tokens`
 * turn may carry a half-emitted tool_call whose args were cut off; treating only
 * a clean `tool_use` stop as actionable keeps us from dispatching it. Shared by
 * both loops.
 */
async function runToolTurn(
    result: GenerateResult,
    toolIndex: Map<string, ToolDef>,
): Promise<Message | null> {
    if (result.stopReason !== "tool_use") return null;
    const calls = toolCalls(result.message);
    if (calls.length === 0) return null;

    const results = await Promise.all(calls.map((c) => runTool(c, toolIndex)));
    return {
        sender: { role: RoleType.User },
        timestamp: Date.now(),
        content: results,
    };
}

/**
 * Drive a model + tools to completion.
 *
 * Each iteration calls {@link ModelClient.generate}, appends the assistant
 * turn, and: if it requested tools: runs them all (in parallel) and appends
 * one user turn of `tool_result`s before looping. Stops when the model returns
 * without tool calls or {@link RunLoopParams.maxTurns} is hit.
 */
export async function runLoop(client: ModelClient, params: RunLoopParams): Promise<RunLoopResult> {
    const maxTurns = normalizeMaxTurns(params.maxTurns);
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
        // estimate says it's grown past the threshold. We reassign `messages`
        // itself (not just the outgoing copy) so the compaction carries forward
        // to every later turn: the whole point for a long-lived session.
        const gated = await compactGate(client, messages, params, usage);
        messages = gated.messages;
        if (gated.compacted) compactions++;

        // Fold this turn's passive context (e.g. the current time) onto the
        // outgoing messages only: `messages` itself, and so the conversation we
        // return, stays free of per-turn injected content.
        const outgoing = await applyContext(messages, context, turns, params.sessionStart);
        const result = await client.generate({ ...params, messages: outgoing });
        turns++;
        final = result;
        usage.add(result.usage);
        messages.push(result.message);

        const toolTurn = await runToolTurn(result, toolIndex);
        if (!toolTurn) break;
        messages.push(toolTurn);

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
        // The buffered path has no mid-stream abort to honor: `generate` either
        // returns a whole turn or rejects (a `canceled` rejection propagates to the
        // caller, it doesn't become a partial result here the way streaming does).
        cancelled: false,
    };
}

// ── Streaming loop ────────────────────────────────────────────────────────────

/**
 * Events a streaming run emits, in core vocabulary. A superset of the bridge's
 * {@link CoreDelta}: the model's own deltas pass through unchanged (so a consumer
 * can print text as it arrives), and the loop adds events for the things the
 * bridge can't know about: compaction and the tool lifecycle. Every run ends
 * with exactly one `loop_done` carrying the full {@link RunLoopResult}.
 */
export type LoopEvent =
    | CoreDelta
    | { kind: "compacted"; summarizedInto: number; turn: number }
    | { kind: "tool_start"; id: string; name: string; args: unknown }
    | { kind: "tool_end"; id: string; name: string; result: unknown; isError: boolean }
    | { kind: "turn_start"; turn: number }
    // The caller aborted the stream (a user pressing "stop"). Carries the partial
    // assistant text streamed up to the abort, so a consumer can mark the reply
    // cancelled while keeping what was already shown. Emitted at most once, and
    // always immediately before the terminal `loop_done` (whose result has
    // `cancelled: true`). `text` is "" when the abort landed before any prose.
    | { kind: "cancelled"; text: string }
    | { kind: "loop_done"; result: RunLoopResult };

/**
 * The streaming counterpart to {@link runLoop}.
 *
 * Drives {@link ModelClient.stream} instead of `generate`, yielding
 * {@link LoopEvent}s as they happen so a caller (e.g. an interactive REPL) can
 * render text token-by-token and show tool activity live. Compaction, usage
 * accounting, the tool loop, and the max-turns cut-off all behave exactly as in
 * {@link runLoop}: this is the same control flow, observed.
 *
 * The model's `stream` ends each turn with a `done` delta carrying the assembled
 * {@link GenerateResult}; we use that as the turn result (it never crosses to the
 * consumer as a `done`: the consumer gets per-turn text deltas and a single
 * terminal `loop_done`). Tools requested by a turn are run between turns, with
 * `tool_start`/`tool_end` bracketing each.
 */
export async function* runLoopStream(
    client: ModelClient,
    params: RunLoopParams,
): AsyncGenerator<LoopEvent, void, void> {
    const maxTurns = normalizeMaxTurns(params.maxTurns);
    const toolIndex = indexTools(params.tools);
    const context = params.context ?? [];

    let messages = [...params.messages];
    const usage = new UsageTracker();
    let final: GenerateResult | undefined;
    let turns = 0;
    let stoppedAtMaxTurns = false;
    let compactions = 0;
    // Set when the caller aborts mid-stream: the turn loop breaks, the partial
    // assistant turn is committed, and `loop_done` reports cancelled: true.
    let cancelled = false;

    // Track text across turns so we can separate it. A multi-turn run emits the
    // model's prose in pieces: the model says something, calls a tool, then on
    // the next turn says more. The model's own newlines live *inside* a turn's
    // text; nothing sits between one turn's last text delta and the next turn's
    // first. A consumer that simply concatenates `text` deltas (the web client's
    // `reply.text +=`, the session's `assistantText`) would glue those pieces
    // into one unspaced wall. We own the turn boundary here: when text resumes in
    // a later turn, prepend a paragraph break to its first delta so every
    // consumer sees the segments as distinct paragraphs without re-deriving the
    // boundary itself. `sawText` is true once any turn has emitted text;
    // `turnHadText` resets each turn and gates "this is the first text since the
    // boundary".
    let sawText = false;
    let turnHadText = false;

    while (turns < maxTurns) {
        const gated = await compactGate(client, messages, params, usage);
        messages = gated.messages;
        if (gated.compacted) {
            compactions++;
            yield { kind: "compacted", summarizedInto: messages.length, turn: turns };
        }

        yield { kind: "turn_start", turn: turns };

        const outgoing = await applyContext(messages, context, turns, params.sessionStart);

        // Consume the model stream, passing its deltas straight through and
        // capturing the terminal `done` as this turn's result. We also accumulate
        // this turn's prose locally so that if the caller aborts mid-stream we can
        // assemble a partial result from what was already streamed rather than
        // discarding it — saving the partial stream is the whole point of an
        // honest cancel.
        let result: GenerateResult | undefined;
        let turnText = "";
        turnHadText = false;
        try {
            for await (const delta of client.stream({ ...params, messages: outgoing })) {
                if (delta.kind === "done") {
                    result = delta.result;
                } else if (delta.kind === "text") {
                    // First non-empty text of a turn that follows an earlier turn's
                    // text: insert the paragraph break that the cross-turn boundary
                    // otherwise lacks. Guard on `delta.text` so an empty delta never
                    // triggers (or absorbs) the separator.
                    if (delta.text.length > 0 && !turnHadText) {
                        turnHadText = true;
                        if (sawText) yield { kind: "text", text: "\n\n" };
                        sawText = true;
                    }
                    turnText += delta.text;
                    yield delta;
                } else {
                    yield delta;
                }
            }
        } catch (err) {
            // A mid-stream abort (the user pressed "stop") is the one failure we
            // turn into a clean, partial completion instead of propagating: the
            // text already streamed is real output we keep. Any other error
            // (network, server, a malformed stream) still throws to the caller.
            if (!isCanceled(err)) throw err;
            cancelled = true;
            turns++;
            // Commit the partial turn as the run's final result: a real assistant
            // message carrying the prose streamed so far, stamped `canceled` so a
            // reader can tell a stopped turn from a finished one. No `done` delta
            // arrived, so its usage is unknown (zeros) — the partial text is the
            // signal that matters, not the token count of an interrupted turn.
            const partial = partialResult(turnText, client.model);
            final = partial;
            messages.push(partial.message);
            // Tell the consumer the turn was cancelled (carrying the partial text),
            // then fall through to the single terminal `loop_done` below.
            yield { kind: "cancelled", text: turnText };
            break;
        }
        if (!result) {
            throw new Error("runLoopStream: model stream ended without a done delta");
        }

        turns++;
        final = result;
        usage.add(result.usage);
        messages.push(result.message);

        if (result.stopReason !== "tool_use") break;
        const calls = toolCalls(result.message);
        if (calls.length === 0) break;

        // Run the requested tools, bracketing each with start/end events so the
        // consumer can show activity. Announce all calls first, then dispatch
        // them in parallel (matching runLoop), reporting each as it settles.
        for (const call of calls) {
            yield { kind: "tool_start", id: call.id, name: call.name, args: call.args };
        }
        const settled = await Promise.all(
            calls.map(async (call) => ({ call, part: await runTool(call, toolIndex) })),
        );
        const resultParts: ContentPart[] = [];
        for (const { call, part } of settled) {
            resultParts.push(part);
            const isError = part.kind === "tool_result" && part.isError === true;
            const value = part.kind === "tool_result" ? part.result : undefined;
            yield { kind: "tool_end", id: call.id, name: call.name, result: value, isError };
        }
        messages.push({
            sender: { role: RoleType.User },
            timestamp: Date.now(),
            content: resultParts,
        });

        if (turns >= maxTurns) stoppedAtMaxTurns = true;
    }

    yield {
        kind: "loop_done",
        result: {
            messages,
            final: final!,
            turns,
            usage: usage.totals(),
            stoppedAtMaxTurns,
            compactions,
            cancelled,
        },
    };
}

/** Whether a thrown value is the bridge's cancellation signal: the caller aborted
 *  the request (a user pressing "stop"). This is the one stream failure the loop
 *  converts into a partial completion rather than rethrowing. */
function isCanceled(err: unknown): boolean {
    return err instanceof HarnessError && err.kind === "canceled";
}

/**
 * Assemble the partial {@link GenerateResult} for a turn the caller aborted, from
 * the prose streamed up to the abort. This is what makes a cancel non-lossy: the
 * text the model had already produced becomes a real assistant message in the
 * conversation, stamped `stopReason: "canceled"` so a reader can distinguish it
 * from a finished turn. Usage is zeroed (no `done` delta carried the real counts;
 * an interrupted turn's token total isn't meaningful), and `content` is empty when
 * the abort landed before any text — a cancelled turn with nothing to save.
 */
function partialResult(text: string, model: string): GenerateResult {
    const content: ContentPart[] = text ? [{ kind: "text", text }] : [];
    return {
        message: { sender: { role: RoleType.Agent, name: model }, timestamp: Date.now(), content },
        stopReason: "canceled",
        usage: {},
        model,
        raw: null,
    };
}
