/**
 * Context compaction — keep a long-lived conversation under the model's context
 * window by summarizing its older turns.
 *
 * An interactive Construct accumulates turns and tool results without bound;
 * left alone, a session eventually exceeds the context window and every further
 * request fails. Compaction replaces a run of older messages with a single
 * compact summary, preserving the most recent turns verbatim so the immediate
 * thread of conversation is untouched.
 *
 * The summary is produced by the model itself, through the same provider-neutral
 * {@link ModelClient} the loop already uses — so compaction works against any
 * provider and needs no extra dependency.
 *
 * Two invariants make this safe to drop into the loop:
 *
 *  1. **System messages are never summarized.** They carry standing guidance and
 *     are preserved as-is, ahead of the summary.
 *  2. **A `tool_call` is never split from its `tool_result`.** Anthropic (and
 *     others) reject a `tool_use` block with no matching result, or a result with
 *     no call. The keep-boundary is snapped so the kept tail always begins on a
 *     clean turn, never mid-pair.
 */

import { RoleType } from "./types.ts";
import type { ContentPart, Message, ToolCallPart, ToolResultPart } from "./types.ts";
import type { ModelClient, Usage } from "./bridge/types.ts";

/** Tuning for {@link compactConversation}. */
export interface CompactionOptions {
    /**
     * How many of the most recent non-system messages to keep verbatim. The
     * summary covers everything older. Defaults to {@link DEFAULT_KEEP_RECENT}.
     */
    keepRecent?: number;
    /** Upper bound on summary length (model tokens). Defaults to
     *  {@link DEFAULT_SUMMARY_MAX_TOKENS}. */
    summaryMaxTokens?: number;
}

/** Default count of recent messages preserved verbatim. */
export const DEFAULT_KEEP_RECENT = 8;
/** Default token ceiling on the generated summary. */
export const DEFAULT_SUMMARY_MAX_TOKENS = 2_000;

/** Outcome of a compaction attempt. */
export interface CompactionResult {
    /** The rewritten conversation: system messages, then the summary, then the
     *  kept tail. */
    messages: Message[];
    /** Whether anything was actually summarized. False means the conversation
     *  was already short enough and {@link messages} equals the input. */
    compacted: boolean;
    /** How many messages were folded into the summary (0 when not compacted). */
    summarizedCount: number;
    /** Token usage of the summarization call, when one was made. Undefined when
     *  no summary was generated (nothing to compact). The loop folds this into
     *  the run's cumulative usage so the cost of compaction isn't invisible. */
    usage?: Usage;
}

/** Pull the ids a turn's tool_call parts declare. */
function callIds(message: Message): Set<string> {
    const ids = new Set<string>();
    for (const part of message.content) {
        if (part.kind === "tool_call") ids.add((part as ToolCallPart).id);
    }
    return ids;
}

/** Pull the callIds a turn's tool_result parts answer. */
function resultIds(message: Message): Set<string> {
    const ids = new Set<string>();
    for (const part of message.content) {
        if (part.kind === "tool_result") ids.add((part as ToolResultPart).callId);
    }
    return ids;
}

/**
 * Given the index where the kept tail would start, move it *earlier* if that
 * boundary would orphan a tool result from its call.
 *
 * The hazard: the message just before the boundary emitted tool_calls, and the
 * first kept message answers them with tool_results. Summarizing the call but
 * keeping the result leaves a result with no call — a wire error. We walk the
 * boundary back past any leading tool-result turn whose calls live in the turn
 * before it, so the kept tail always starts on a self-contained turn.
 */
function safeBoundary(messages: Message[], start: number): number {
    let boundary = start;
    while (boundary > 0 && boundary < messages.length) {
        const first = messages[boundary];
        const answered = resultIds(first);
        if (answered.size === 0) break; // tail starts clean

        const prev = messages[boundary - 1];
        const calls = callIds(prev);
        // Does the kept tail's leading result answer a call we'd be summarizing?
        const orphaned = [...answered].some((id) => calls.has(id));
        if (!orphaned) break;

        // Pull the call's turn into the kept tail too, then re-check (the call
        // turn might itself answer an even earlier call, though that's rare).
        boundary -= 1;
    }
    return boundary;
}

/** Render one message as a compact transcript line for the summarizer prompt. */
function renderForSummary(message: Message): string {
    const who =
        message.sender.role === RoleType.Agent
            ? "Assistant"
            : message.sender.role === RoleType.System
              ? "System"
              : "User";
    const parts = message.content.map((p) => renderPart(p)).filter(Boolean);
    return `${who}: ${parts.join(" ")}`;
}

function renderPart(part: ContentPart): string {
    switch (part.kind) {
        case "text":
            return part.text;
        case "tool_call":
            return `[called ${part.name}(${safeJson(part.args)})]`;
        case "tool_result":
            return `[tool result${part.isError ? " (error)" : ""}: ${safeJson(part.result)}]`;
    }
}

function safeJson(value: unknown): string {
    if (typeof value === "string") return value;
    try {
        return JSON.stringify(value) ?? String(value);
    } catch {
        return "<unserializable>";
    }
}

/** The instruction we give the model to summarize the older turns. Phrased to
 *  preserve the things a resumed conversation actually needs: decisions, facts
 *  established, open threads — not a play-by-play. */
const SUMMARY_SYSTEM =
    "You are compacting an ongoing conversation to fit a context window. " +
    "Write a concise summary of the transcript below that preserves everything " +
    "needed to continue seamlessly: decisions made, facts established, the " +
    "user's goals and preferences, and any open or unfinished threads. Omit " +
    "pleasantries and verbatim detail. Write in plain prose, third person.";

/**
 * Summarize the older portion of a conversation and return the rewritten
 * message list.
 *
 * Keeps all system messages and the most recent `keepRecent` non-system
 * messages verbatim; everything between is replaced by one model-generated
 * summary, carried as a `user` turn tagged so it's recognizable as injected
 * context rather than something the user said.
 *
 * Returns `compacted: false` (and the input unchanged) when there's nothing
 * worth summarizing — i.e. the non-system history already fits within
 * `keepRecent`. Summarization uses {@link ModelClient.generate}; a failure
 * propagates to the caller, which should decide whether to proceed uncompacted.
 */
export async function compactConversation(
    client: ModelClient,
    messages: Message[],
    options: CompactionOptions = {},
): Promise<CompactionResult> {
    const keepRecent = Math.max(0, Math.floor(options.keepRecent ?? DEFAULT_KEEP_RECENT));
    const summaryMaxTokens = options.summaryMaxTokens ?? DEFAULT_SUMMARY_MAX_TOKENS;

    const system = messages.filter((m) => m.sender.role === RoleType.System);
    const conversation = messages.filter((m) => m.sender.role !== RoleType.System);

    // Nothing to do: the live history already fits within the keep window.
    if (conversation.length <= keepRecent) {
        return { messages, compacted: false, summarizedCount: 0 };
    }

    // Provisional split, then snap the boundary so we never orphan a tool result.
    const provisional = conversation.length - keepRecent;
    const boundary = safeBoundary(conversation, provisional);

    // The snap could pull the whole history into the kept tail (e.g. one giant
    // tool-pair chain). Nothing left to summarize → leave it alone.
    if (boundary <= 0) {
        return { messages, compacted: false, summarizedCount: 0 };
    }

    const toSummarize = conversation.slice(0, boundary);
    const kept = conversation.slice(boundary);

    const transcript = toSummarize.map(renderForSummary).join("\n");
    const result = await client.generate({
        messages: [
            {
                sender: { role: RoleType.System },
                timestamp: Date.now(),
                content: [{ kind: "text", text: SUMMARY_SYSTEM }],
            },
            {
                sender: { role: RoleType.User },
                timestamp: Date.now(),
                content: [{ kind: "text", text: transcript }],
            },
        ],
        maxTokens: summaryMaxTokens,
    });

    const summaryText = result.message.content
        .filter((p): p is Extract<ContentPart, { kind: "text" }> => p.kind === "text")
        .map((p) => p.text)
        .join(" ")
        .trim();

    // A summarizer that returned nothing usable would lose history if we dropped
    // the turns anyway — bail and keep the originals. We still report the usage:
    // the call happened and cost tokens even though it produced no summary.
    if (!summaryText) {
        return { messages, compacted: false, summarizedCount: 0, usage: result.usage };
    }

    const summaryMessage: Message = {
        sender: { role: RoleType.User, name: "summary" },
        timestamp: Date.now(),
        content: [
            {
                kind: "text",
                text: `[Summary of ${toSummarize.length} earlier message(s)]\n${summaryText}`,
            },
        ],
    };

    return {
        messages: [...system, summaryMessage, ...kept],
        compacted: true,
        summarizedCount: toSummarize.length,
        usage: result.usage,
    };
}
