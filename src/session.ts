/**
 * Session — a long-lived, interactive Construct.
 *
 * Where {@link runLoop} drives a single task to completion, a Session is the
 * persistent thing a user *talks to*: it holds the conversation across many
 * user turns, recomputes turn-relevant memory recall and passive context before
 * each turn, streams the model's reply token-by-token, and commits the result
 * back into its own history so the next turn builds on it.
 *
 * It speaks only core types and the bridge interface — the same discipline as
 * the rest of `src/` — so it works against any {@link ModelClient}. Memory is
 * optional: with no store, a Session is just a streaming chat with context.
 */

import { RoleType } from "./types.ts";
import type { Message, ToolDef } from "./types.ts";
import { runLoopStream } from "./bridge/loop.ts";
import type { CompactionConfig, LoopEvent } from "./bridge/loop.ts";
import type { ModelClient, ProviderOptions } from "./bridge/types.ts";
import type { ContextProvider } from "./context.ts";
import { temporalContext } from "./context.ts";
import { MemoryStore } from "./memory.ts";
import { memoryTools, recallContext } from "./memoryTools.ts";
import type { Embedder } from "./embeddings.ts";

/** Configuration for a {@link Session}. */
export interface SessionConfig {
    /** The model client to drive. Required. */
    client: ModelClient;
    /** Base system guidance, present on every turn ahead of recalled memory. */
    system: string;
    /** Tools the model may call, beyond the memory tools a store adds. */
    tools?: ToolDef[];
    /** Memory store. When given, the model gets memory_save/recall/forget tools
     *  and each turn injects turn-relevant recalled memories. */
    store?: MemoryStore;
    /** Embedder for semantic recall. Only meaningful alongside `store`. */
    embedder?: Embedder;
    /** Passive context providers. Defaults to a single temporal provider so the
     *  Construct always knows the current date/time; pass `[]` to disable, or
     *  your own list to replace it. */
    context?: ContextProvider[];
    /** Auto-compaction config, forwarded to the loop. Omit to disable. */
    compaction?: CompactionConfig;
    /** Per-turn tool/turn cap, forwarded to the loop. */
    maxTurns?: number;
    /** Provider-specific knobs (thinking, effort, caching), forwarded as-is. */
    providerOptions?: ProviderOptions;
    /** How many memories turn-relevant recall injects. */
    recallLimit?: number;
}

/** A completed Session turn: what the assistant said, plus run accounting. */
export interface TurnResult {
    /** The assistant's final text for this turn (concatenated text parts). */
    text: string;
    /** Model turns this user-turn took (≥1; more when tools were called). */
    modelTurns: number;
    /** Whether the loop was cut off at maxTurns mid tool-use. */
    stoppedAtMaxTurns: boolean;
    /** Compactions performed during this turn. */
    compactions: number;
    /** Cumulative token usage for this turn's run. */
    usage: { inputTokens: number; outputTokens: number; cacheReadTokens: number };
}

/**
 * A stateful conversation with a model, with memory and streaming.
 *
 * Construct one, then call {@link send} per user message. Each call streams
 * {@link LoopEvent}s (text deltas, tool activity) and, when the async generator
 * returns, yields a {@link TurnResult} as its return value. The conversation
 * grows internally across calls; {@link history} exposes a read-only view.
 */
export class Session {
    private readonly cfg: SessionConfig;
    private readonly tools: ToolDef[];
    private readonly context: ContextProvider[];
    /** The durable conversation — user/assistant/tool turns only. The system
     *  turn is rebuilt per send (recall is turn-relevant), so it is NOT stored
     *  here; it's prepended at send time and never persisted. */
    private conversation: Message[] = [];

    constructor(config: SessionConfig) {
        this.cfg = config;
        // A store contributes its memory tools; the model's own tools come after.
        const memTools = config.store ? memoryTools(config.store, config.embedder) : [];
        this.tools = [...memTools, ...(config.tools ?? [])];
        this.context = config.context ?? [temporalContext()];
    }

    /** A read-only snapshot of the durable conversation (no system turn). */
    history(): readonly Message[] {
        return this.conversation;
    }

    /** Drop all conversation history, starting the Construct fresh. Memory in
     *  the store is untouched — only the in-session transcript is cleared. */
    reset(): void {
        this.conversation = [];
    }

    /**
     * Send a user message and stream the reply.
     *
     * Builds this turn's system prompt (base guidance + memory relevant to the
     * message), appends the user turn to the durable conversation, then streams
     * a {@link runLoopStream} run. The system turn is folded in only for this
     * run — recall is recomputed next turn against the next message — so it is
     * never persisted into {@link history}. On completion the assistant turn(s)
     * and any tool turns are committed back to the conversation.
     *
     * Yields each {@link LoopEvent}; the generator's return value is the
     * {@link TurnResult}.
     */
    async *send(text: string): AsyncGenerator<LoopEvent, TurnResult, void> {
        const userTurn: Message = {
            sender: { role: RoleType.User },
            timestamp: Date.now(),
            content: [{ kind: "text", text }],
        };

        const systemTurn = await this.buildSystem(text);

        // The run sees: system + durable history + this user turn. We pass a
        // copy as the run's starting messages; the loop returns the full
        // post-run conversation, from which we re-extract the durable part.
        const startMessages = [systemTurn, ...this.conversation, userTurn];

        let result: TurnResult | undefined;
        let assistantText = "";

        const stream = runLoopStream(this.cfg.client, {
            messages: startMessages,
            tools: this.tools.length ? this.tools : undefined,
            context: this.context,
            compaction: this.cfg.compaction,
            maxTurns: this.cfg.maxTurns,
            providerOptions: this.cfg.providerOptions,
        });

        for await (const event of stream) {
            if (event.kind === "text") assistantText += event.text;
            if (event.kind === "loop_done") {
                const r = event.result;
                // Commit the durable conversation: everything the run produced
                // except the system turn we prepended (which is rebuilt per turn).
                this.conversation = r.messages.filter((m) => m.sender.role !== RoleType.System);
                result = {
                    text: assistantText.trim(),
                    modelTurns: r.turns,
                    stoppedAtMaxTurns: r.stoppedAtMaxTurns,
                    compactions: r.compactions,
                    usage: {
                        inputTokens: r.usage.inputTokens,
                        outputTokens: r.usage.outputTokens,
                        cacheReadTokens: r.usage.cacheReadTokens,
                    },
                };
                continue;
            }
            yield event;
        }

        // loop_done is always emitted by runLoopStream, so result is set.
        return result!;
    }

    /**
     * Build the system turn for a send: base guidance plus memory relevant to
     * `query`. With no store, it's just the base guidance. Async because
     * semantic recall embeds the query.
     */
    private async buildSystem(query: string): Promise<Message> {
        let text = this.cfg.system;
        if (this.cfg.store) {
            const recalled = await recallContext(this.cfg.store, {
                query,
                embedder: this.cfg.embedder,
                limit: this.cfg.recallLimit,
            });
            if (recalled) text = `${text}\n\n${recalled}`;
        }
        return {
            sender: { role: RoleType.System },
            timestamp: Date.now(),
            content: [{ kind: "text", text }],
        };
    }
}
