/**
 * Usage accounting and a cheap token estimator.
 *
 * Two jobs the harness needs but the bridge deliberately doesn't do:
 *
 *  1. The bridge reports {@link Usage} for a *single* turn. A long-lived session
 *     runs many turns, so something has to *accumulate* them: total input,
 *     output, and cache-read tokens across the whole run, for cost control and
 *     telemetry. {@link UsageTracker} is that accumulator.
 *
 *  2. To decide *when* to compact, the loop needs a running estimate of how big
 *     the conversation has grown: before it sends it and gets a real
 *     `input_tokens` back (by which point an over-limit request has already
 *     failed). {@link estimateTokens} is a deliberately crude, synchronous,
 *     dependency-free heuristic for that gate. It is NOT a tokenizer and never
 *     claims to be: it errs toward over-counting so the compaction trigger fires
 *     early rather than late.
 *
 * This module speaks only core types and has no provider or I/O dependency, in
 * keeping with the rest of `src/`.
 */

import type { ContentPart, Message } from "./types.ts";
import type { Usage } from "./bridge/types.ts";

/** Cumulative token counts across every turn of a run. */
export interface CumulativeUsage {
    inputTokens: number;
    outputTokens: number;
    /** Tokens served from a cached prefix, summed across turns. */
    cacheReadTokens: number;
    /** How many model turns contributed to these totals. */
    turns: number;
}

/**
 * Accumulates per-turn {@link Usage} into running totals.
 *
 * The bridge omits a field when a provider doesn't report it; we treat a missing
 * field as zero so the totals stay coherent across providers that report
 * different subsets. `add` is the only mutator; {@link totals} returns a
 * snapshot copy so callers can't mutate our state through it.
 */
export class UsageTracker {
    private input = 0;
    private output = 0;
    private cacheRead = 0;
    private turnCount = 0;

    /** Fold one turn's usage into the running totals. */
    add(usage: Usage): void {
        this.input += usage.inputTokens ?? 0;
        this.output += usage.outputTokens ?? 0;
        this.cacheRead += usage.cacheReadTokens ?? 0;
        this.turnCount += 1;
    }

    /** A snapshot of the totals so far. */
    totals(): CumulativeUsage {
        return {
            inputTokens: this.input,
            outputTokens: this.output,
            cacheReadTokens: this.cacheRead,
            turns: this.turnCount,
        };
    }
}

/**
 * Roughly how many characters one token is worth, for the estimator.
 *
 * English prose averages ~4 chars/token across modern BPE tokenizers; we use a
 * slightly *lower* divisor so the estimate runs high. Over-counting is the safe
 * direction for a compaction gate: triggering a little early wastes a summary,
 * triggering late means an over-limit request that the API rejects outright.
 */
const CHARS_PER_TOKEN = 3.5;

/** Fixed per-message overhead (role markers, block framing) the char count
 *  alone misses, so many tiny messages don't read as ~free. Approximate. */
const PER_MESSAGE_TOKENS = 4;
/** Extra overhead a tool-call/tool-result part carries (id, name, JSON
 *  framing) beyond its serialized text. Approximate. */
const PER_TOOL_PART_TOKENS = 8;

/** Estimate the token cost of a single content part. */
function estimatePart(part: ContentPart): number {
    switch (part.kind) {
        case "text":
            return Math.ceil(part.text.length / CHARS_PER_TOKEN);
        case "tool_call":
            return (
                PER_TOOL_PART_TOKENS +
                Math.ceil((part.name.length + serializedLength(part.args)) / CHARS_PER_TOKEN)
            );
        case "tool_result":
            return (
                PER_TOOL_PART_TOKENS + Math.ceil(serializedLength(part.result) / CHARS_PER_TOKEN)
            );
    }
}

/** Length of a value once serialized for the wire. A string is itself; anything
 *  else is JSON-stringified (with a length fallback for unserializable values,
 *  mirroring the bridge's tolerance for them). */
function serializedLength(value: unknown): number {
    if (typeof value === "string") return value.length;
    try {
        return JSON.stringify(value)?.length ?? String(value).length;
    } catch {
        return String(value).length;
    }
}

/**
 * Estimate the total token cost of a conversation.
 *
 * Crude on purpose (see the module note): a synchronous heuristic for the
 * compaction gate, not a tokenizer. Sums each part's estimate plus a small
 * per-message overhead.
 */
export function estimateTokens(messages: readonly Message[]): number {
    let total = 0;
    for (const msg of messages) {
        total += PER_MESSAGE_TOKENS;
        for (const part of msg.content) total += estimatePart(part);
    }
    return total;
}

/**
 * Estimate the token cost of a bare string, using the same crude chars-per-token
 * heuristic {@link estimateTokens} applies to text parts. For per-section
 * accounting (the context inspector) where there's no surrounding `Message`
 * framing to charge for, so the per-message overhead is deliberately excluded.
 */
export function estimateTextTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN);
}
