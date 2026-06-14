/**
 * Session: a long-lived, interactive Construct.
 *
 * Where {@link runLoop} drives a single task to completion, a Session is the
 * persistent thing a user *talks to*: it holds the conversation across many
 * user turns, recomputes turn-relevant memory recall and passive context before
 * each turn, streams the model's reply token-by-token, and commits the result
 * back into its own history so the next turn builds on it.
 *
 * It speaks only core types and the bridge interface: the same discipline as
 * the rest of `src/`: so it works against any {@link ModelClient}. Memory is
 * optional: with no store, a Session is just a streaming chat with context.
 */

import { RoleType } from "./types.ts";
import type { Message, ToolDef } from "./types.ts";
import { runLoopStream } from "./bridge/loop.ts";
import type { CompactionConfig, LoopEvent } from "./bridge/loop.ts";
import type { ModelClient, ProviderOptions } from "./bridge/types.ts";
import type { ContextProvider } from "./context.ts";
import { temporalContext } from "./context.ts";
import { MemoryStore, MAX_LIMIT } from "./memory.ts";
import { memoryTools, recallContext } from "./memoryTools.ts";
import { eventTools, embedEventIfPossible } from "./eventTools.ts";
import { goalTools, goalContext } from "./goalTools.ts";
import { GoalStore } from "./goals.ts";
import type { Embedder } from "./embeddings.ts";
import { EventStore } from "./events.ts";
import type { Event, EventInput } from "./events.ts";

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
    /**
     * Append-only event log. When given, every turn this Session runs (the user
     * message, each tool call and its result, and the assistant's reply) is
     * appended to the log under {@link sessionId}, so the transcript persists
     * beyond the in-memory {@link history} and outlives the process. The log is
     * the load-bearing substrate {@link MemoryStore} curates over: a Session with
     * one writes its whole life into it for free. Omit to keep a Session purely
     * in-memory.
     */
    events?: EventStore;
    /**
     * The id grouping this Session's events in the {@link events} log. Defaults to
     * a fresh id per Session so two Sessions sharing one log stay separable. Only
     * meaningful alongside `events`; pass a stable id to resume appending to (and
     * recalling) an earlier conversation's transcript.
     */
    sessionId?: string;
    /**
     * Give the model a `transcript_recall` tool over its own {@link events} log,
     * so it can introspect its past (what happened earlier, whether a tool already
     * ran) beyond the in-context history. Scoped to this Session's transcript by
     * default. Defaults to `true` when an `events` log is configured; pass `false`
     * to withhold the tool (the log still records the turn either way). No-op
     * without `events`: there's nothing to recall.
     */
    transcriptRecall?: boolean;
    /**
     * Goal store. When given, the model gets goal_set/update/list tools and each
     * turn injects this session's *active* goals into the system prompt, so the
     * Construct holds its intent across turns. Goals are scoped to this Session's
     * {@link sessionId}. Omit for a goal-less Session. Only the default context
     * provider list picks up the goal injection; passing your own `context`
     * replaces it (add {@link goalContext} yourself if you want both).
     */
    goals?: GoalStore;
    /**
     * Embedder for semantic recall. Meaningful alongside `store` (saved memories
     * are embedded so memory_recall matches by meaning) and alongside `events`
     * (each message turn is embedded as it's logged so `transcript_recall`'s
     * semantic path lights up, not just its lexical FTS one). Best-effort
     * throughout: an embedding outage degrades recall to lexical, never failing a
     * turn. Omit to keep recall purely lexical.
     */
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
    /** The log this Session appends its turns to, or undefined when none was
     *  configured (a purely in-memory Session). */
    private readonly events?: EventStore;
    /** The id under which this Session's events are grouped in {@link events}. */
    private readonly sessionId: string;
    /** Epoch-ms this conversation began, forwarded to context providers so a
     *  temporal provider can report session duration. For a resumed Session
     *  (pinned id with an existing transcript) this is the earliest logged event's
     *  time, so "running for 2 days" reflects the real conversation, not this
     *  process's uptime; otherwise it's construction time. */
    private readonly startedAt: number;
    /** The durable conversation: user/assistant/tool turns only. The system
     *  turn is rebuilt per send (recall is turn-relevant), so it is NOT stored
     *  here; it's prepended at send time and never persisted. */
    private conversation: Message[] = [];
    /** In-flight, fire-and-forget event embeds. An appended message is embedded
     *  off the turn's hot path (embedding is a network call; the turn must not
     *  block on it), so we track the promises here. {@link flushEmbeddings} awaits
     *  them, which the tests use for determinism and a clean shutdown can use to
     *  drain before closing the store. Each promise already swallows its own
     *  failure (see {@link embedMessage}), so this set never holds a rejecting
     *  promise. */
    private readonly pendingEmbeds = new Set<Promise<void>>();

    constructor(config: SessionConfig) {
        this.cfg = config;
        this.events = config.events;
        this.sessionId = config.sessionId ?? freshSessionId();
        // A store contributes its memory tools; an event log can contribute the
        // transcript tool (scoped to this Session); the model's own tools come
        // last. transcript_recall is on by default whenever a log is present.
        const memTools = config.store ? memoryTools(config.store, config.embedder) : [];
        const txTools =
            this.events && config.transcriptRecall !== false
                ? eventTools(this.events, {
                      sessionId: this.sessionId,
                      embedder: config.embedder,
                  })
                : [];
        const goTools = config.goals ? goalTools(config.goals, this.sessionId) : [];
        this.tools = [...memTools, ...txTools, ...goTools, ...(config.tools ?? [])];
        // Default context: the temporal provider, plus a goal provider when a goal
        // store is configured so active goals stand in front of the model each
        // turn. A caller-supplied `context` replaces this whole list.
        this.context = config.context ?? [
            temporalContext(),
            ...(config.goals ? [goalContext(config.goals, this.sessionId)] : []),
        ];
        this.startedAt = this.resolveStart();
    }

    /**
     * Construct a Session resuming a past conversation, with its prior turns
     * already loaded into the model's working context. The one-call form of
     * "construct with the old id, then {@link rehydrate}".
     *
     * Requires `config.events` (there's nothing to resume from without a log) and
     * a `config.sessionId` naming the conversation to pick up. The returned Session
     * appends to and recalls that same transcript, reports its real start time, and
     * starts with {@link history} populated from the log, so the first {@link send}
     * builds on the prior exchange exactly as an uninterrupted turn would.
     *
     * `maxMessages` bounds how much of a long transcript to reload (most recent
     * kept); see {@link rehydrate}.
     */
    static async resume(
        config: SessionConfig & { events: EventStore; sessionId: string },
        maxMessages?: number,
    ): Promise<Session> {
        const session = new Session(config);
        await session.rehydrate(maxMessages);
        return session;
    }

    /** When this conversation began: the earliest event already in the log under
     *  this Session's id (a resume), else now (a fresh Session). Best-effort —
     *  a log read failure falls back to now rather than breaking construction. */
    private resolveStart(): number {
        const now = Date.now();
        if (!this.events) return now;
        try {
            // The oldest event for this session, if any. `recent` is newest-first
            // and capped, so for a long transcript this is the oldest *within the
            // page* — close enough for a coarse "running for N days" phrasing.
            const turns = this.events.recent({ session: this.sessionId });
            const earliest = turns.length ? turns[turns.length - 1]!.ts : now;
            return Math.min(earliest, now);
        } catch {
            return now;
        }
    }

    /** The id this Session's events are grouped under in the log. Read it to
     *  later query {@link EventStore} for this conversation's transcript, or to
     *  resume the Session by passing it back as {@link SessionConfig.sessionId}. */
    get id(): string {
        return this.sessionId;
    }

    /** A read-only snapshot of the durable conversation (no system turn). */
    history(): readonly Message[] {
        return this.conversation;
    }

    /** Drop the in-memory conversation, starting the Construct fresh. Memory in
     *  the store and the appended event log are both untouched: only the live
     *  in-session transcript is cleared. The log keeps the full record, so
     *  {@link transcript} still returns the turns this reset just forgot. */
    reset(): void {
        this.conversation = [];
    }

    /**
     * Read this Session's turns back out of the {@link EventStore} as a scoped
     * view: the events appended under this Session's {@link id}, oldest first
     * (reading order). This is the persistent counterpart to {@link history}:
     * where `history` is the live in-memory conversation (cleared by
     * {@link reset}, lost on process exit), this is the durable record the log
     * kept, queryable across restarts and independent of the current process.
     *
     * Returns an empty array when no log is configured. `limit` bounds the read
     * (default {@link EventStore}'s page size); pass a larger one to page deeper.
     */
    transcript(limit?: number): Event[] {
        if (!this.events) return [];
        // `recent` is newest-first; reverse to natural reading order so the
        // transcript reads top-to-bottom like the conversation it records.
        return this.events.recent({ session: this.sessionId, limit }).reverse();
    }

    /**
     * Rehydrate the in-memory conversation from this Session's logged transcript,
     * so resuming a past conversation starts with that conversation already in the
     * model's working context, not just searchable via `transcript_recall`.
     *
     * Construction alone resumes the *log* (a pinned {@link sessionId} keeps
     * appending to and recalling the same transcript, and {@link startedAt} already
     * reflects the real start). But the constructor is synchronous and leaves
     * {@link history} empty: the model could look its past up, but it didn't *start*
     * with it in front of it. This is the explicit, async step that loads it. The
     * usual resume flow is: construct with the old `sessionId` + the same
     * {@link EventStore}, then `await session.rehydrate()` before the first
     * {@link send}. {@link Session.resume} wraps both.
     *
     * Faithful reconstruction: each logged event maps back to the same message
     * shape {@link send} commits, so a resumed turn sees exactly what an
     * uninterrupted one would —
     *  - a user/agent `message` → a User/Agent text turn,
     *  - a `tool_call` → an Agent turn carrying the {@link ToolCallPart}
     *    (id/name/args recovered from `correlation` + `meta`),
     *  - a `tool_result` → a Tool turn carrying the {@link ToolResultPart}
     *    (the structured payload from `meta.result`, not the FTS text in
     *    `content`).
     * Events the conversation doesn't replay (recall/dream/system signals, or a
     * tool event missing the correlation that threads a call to its result) are
     * skipped rather than guessed at: a malformed transcript degrades to a shorter
     * clean history, never a corrupt one.
     *
     * Replaces the current in-memory conversation (like {@link reset} then load).
     * Returns the number of messages rebuilt. No-op returning 0 without a log.
     *
     * `maxMessages` bounds how much of a very long transcript to pull back into
     * context (oldest dropped first, so the most recent exchange is always kept);
     * omit to load the whole transcript. The log keeps everything regardless, and
     * `transcript_recall` still reaches what wasn't loaded.
     */
    async rehydrate(maxMessages?: number): Promise<number> {
        if (!this.events) return 0;
        const events = this.readFullTranscript();
        const rebuilt = messagesFromEvents(events);
        this.conversation =
            maxMessages !== undefined && rebuilt.length > maxMessages
                ? rebuilt.slice(rebuilt.length - maxMessages)
                : rebuilt;
        return this.conversation.length;
    }

    /** Read this Session's whole transcript oldest-first, paging through the log
     *  so a conversation longer than one page is fully recovered. Best-effort: a
     *  read failure returns what was gathered so far rather than throwing, matching
     *  the log's "observer, never a gate" posture. */
    private readFullTranscript(): Event[] {
        const out: Event[] = [];
        const page = MAX_LIMIT;
        for (let offset = 0; ; offset += page) {
            let batch: Event[];
            try {
                // `recent` is newest-first; gather pages then reverse the whole
                // thing once at the end into reading order.
                batch = this.events!.recent({ session: this.sessionId, limit: page, offset });
            } catch {
                break;
            }
            out.push(...batch);
            if (batch.length < page) break;
        }
        return out.reverse();
    }

    /**
     * Send a user message and stream the reply.
     *
     * Builds this turn's system prompt (base guidance + memory relevant to the
     * message), appends the user turn to the durable conversation, then streams
     * a {@link runLoopStream} run. The system turn is folded in only for this
     * run: recall is recomputed next turn against the next message: so it is
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

        // The user's message is the first thing that happened this turn: log it
        // before the model runs, so even a turn that errors mid-flight leaves its
        // prompt in the transcript. The rest of the turn (tool activity, the
        // reply) is appended as the stream surfaces it below. We keep its id so a
        // memory the model saves this turn can be linked back to the message that
        // prompted it (provenance: curation over the log).
        const userEvent = this.logEvent({ kind: "message", role: "user", content: text });
        // Embed it (off the hot path) so semantic transcript_recall covers this
        // turn, not just lexical FTS. No-op without an embedder or a log.
        this.embedMessage(userEvent);

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
            sessionStart: this.startedAt,
            compaction: this.cfg.compaction,
            maxTurns: this.cfg.maxTurns,
            providerOptions: this.cfg.providerOptions,
        });

        for await (const event of stream) {
            if (event.kind === "text") assistantText += event.text;
            // Mirror the tool lifecycle into the log: a call when it starts, its
            // result when it ends, correlated by the call id so a reader can
            // thread a request to its answer. tool_result content is stringified
            // because the log's `content` is text; the structured payload rides
            // along in `meta`.
            if (event.kind === "tool_start") {
                this.logEvent({
                    kind: "tool_call",
                    role: "agent",
                    content: event.name,
                    correlation: event.id,
                    meta: { name: event.name, args: event.args },
                });
            } else if (event.kind === "tool_end") {
                this.logEvent({
                    kind: "tool_result",
                    role: "tool",
                    content: stringifyResult(event.result),
                    correlation: event.id,
                    meta: { name: event.name, result: event.result, isError: event.isError },
                });
                // A memory the model just saved is curated from *this* turn:
                // point its provenance at the user message that prompted it. This
                // is the overlay turning MemoryStore into curation over the log.
                this.linkSavedMemory(event, userEvent);
            }
            if (event.kind === "loop_done") {
                const r = event.result;
                // The assistant's final text for this turn closes it out in the
                // log (skip an empty reply: a turn that only called tools and
                // stopped has nothing to record as a message).
                const reply = assistantText.trim();
                if (reply) {
                    const replyEvent = this.logEvent({
                        kind: "message",
                        role: "agent",
                        content: reply,
                    });
                    // Embed the agent's reply too: it's the other half of the
                    // conversation a later turn might recall by meaning.
                    this.embedMessage(replyEvent);
                }
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
     * Append one event to the log under this Session's id, if a log is
     * configured. Stamps the {@link sessionId} so every turn is scoped to this
     * conversation. Best-effort by contract: the log is an observer of the
     * conversation, never a gate on it, so a logging failure (a closed store, a
     * bad payload) must not take down the turn the user is mid-way through. We
     * swallow it rather than let it surface as a send error.
     */
    private logEvent(input: Omit<EventInput, "session">): Event | undefined {
        if (!this.events) return undefined;
        try {
            return this.events.append({ ...input, session: this.sessionId });
        } catch {
            // The transcript is a side-record; losing one event must not fail
            // the turn. (A persistent failure shows up as a gap in transcript().)
            return undefined;
        }
    }

    /**
     * Embed one freshly-logged message event so it becomes semantically
     * recallable, and track the in-flight work in {@link pendingEmbeds}.
     *
     * Why only messages, and why off the hot path:
     *  - {@link EventStore.append} never embeds; the log is total but the vector
     *    index is deliberately selective (a linear cosine scan stays cheap only
     *    while most events have no vector). Messages are the semantically
     *    meaningful turns worth recalling by meaning; tool calls/results (often a
     *    whole file or search dump) stay lexical-only by design. This is the call
     *    that lights `transcript_recall`'s semantic path up at all.
     *  - Embedding is a network round-trip. The turn must never block on it, so
     *    we fire it without awaiting and let it settle in the background; the user
     *    sees their reply stream immediately. A turn whose embed hasn't finished
     *    is simply lexical-only until it does, exactly as before.
     *
     * No-op without an embedder, without a log (no event was created), or for a
     * non-message event. The promise swallows its own failure (see
     * {@link embedEventIfPossible}) so {@link pendingEmbeds} never rejects, and
     * removes itself on settle so the set doesn't grow unbounded across a long
     * conversation.
     */
    private embedMessage(event: Event | undefined): void {
        if (!this.events || !this.cfg.embedder || !event) return;
        const store = this.events;
        const task = embedEventIfPossible(store, this.cfg.embedder, event)
            .catch(() => {
                // embedEventIfPossible already swallows EmbeddingError and a
                // closed-store EventError; this guards any other surprise so a
                // background embed can never become an unhandled rejection that
                // crashes the process.
                return false;
            })
            .then(() => {
                this.pendingEmbeds.delete(task);
            });
        this.pendingEmbeds.add(task);
    }

    /**
     * Await every in-flight message embed kicked off by this Session. The embeds
     * run off the turn's hot path (see {@link embedMessage}), so a caller that
     * needs them durable before proceeding — a test asserting semantic recall, or
     * a shutdown draining work before it closes the store — awaits this. No-op
     * when nothing is pending. Never rejects: each tracked promise already
     * swallowed its own failure.
     */
    async flushEmbeddings(): Promise<void> {
        await Promise.all([...this.pendingEmbeds]);
    }

    /**
     * If a tool_end event reports a successful `memory_save`, link the new
     * memory's provenance to `userEvent` (the message that prompted this turn).
     * No-op unless a store, a log, and a captured user event are all present, and
     * the tool actually saved (the result shape `memory_save` returns on success
     * is `{ saved: true, memory: { id } }`). Best-effort like the rest of the
     * logging: a provenance failure must not disturb the turn.
     */
    private linkSavedMemory(event: LoopEvent, userEvent: Event | undefined): void {
        const store = this.cfg.store;
        if (!store || !userEvent) return;
        if (event.kind !== "tool_end" || event.name !== "memory_save" || event.isError) return;
        const result = event.result;
        if (typeof result !== "object" || result === null) return;
        const r = result as { saved?: unknown; memory?: { id?: unknown } };
        if (r.saved !== true || typeof r.memory?.id !== "number") return;
        try {
            store.setProvenance(r.memory.id, userEvent.id);
        } catch {
            // Provenance is an annotation; failing to record it must not fail the
            // turn the user is in the middle of.
        }
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

/** A short, unique id for a Session's events when the caller didn't pin one.
 *  Time-ordered prefix plus a random suffix so two Sessions created in the same
 *  millisecond on one shared log don't collide. Not security-sensitive; it only
 *  needs to separate one conversation's events from another's. */
function freshSessionId(): string {
    const stamp = Date.now().toString(36);
    const rand = Math.random().toString(36).slice(2, 10);
    return `s_${stamp}_${rand}`;
}

/**
 * Rebuild the durable conversation ({@link Message}[], oldest first) from a
 * Session's logged events, the inverse of the append `send` does as it runs. Used
 * by {@link Session.rehydrate} to resume a past conversation into the model's
 * working context.
 *
 * Each event maps back to the message shape it came from; an event the
 * conversation doesn't carry (a `recall`/`dream`/system signal, or a tool event
 * with no `correlation` to thread a call to its result) is skipped, so a partial
 * or hand-edited log degrades to a shorter clean history rather than a corrupt
 * one. The `meta` payloads `send` wrote (`{ name, args }` on a call, `{ result,
 * isError }` on a result) are the authoritative source for the structured parts;
 * the events' text `content` is only the FTS projection and isn't trusted here.
 *
 * Exported so a caller reconstructing a conversation outside a Session (a viewer,
 * a migration) can reuse the exact same mapping.
 */
export function messagesFromEvents(events: readonly Event[]): Message[] {
    const messages: Message[] = [];
    for (const e of events) {
        if (e.kind === "message") {
            // user → User, anything else (agent) → Agent. A message with empty
            // text can't have been a real turn; skip it.
            const text = e.content;
            if (!text) continue;
            const role = e.role === "user" ? RoleType.User : RoleType.Agent;
            messages.push({
                sender: { role },
                timestamp: e.ts,
                content: [{ kind: "text", text }],
            });
        } else if (e.kind === "tool_call") {
            // The call id (correlation) is what threads this to its result; without
            // it the pair can't be reconstructed, so drop it.
            if (!e.correlation) continue;
            const meta = (e.meta ?? {}) as { name?: unknown; args?: unknown };
            const name = typeof meta.name === "string" ? meta.name : e.content;
            messages.push({
                sender: { role: RoleType.Agent },
                timestamp: e.ts,
                content: [{ kind: "tool_call", id: e.correlation, name, args: meta.args }],
            });
        } else if (e.kind === "tool_result") {
            if (!e.correlation) continue;
            const meta = (e.meta ?? {}) as { result?: unknown; isError?: unknown };
            messages.push({
                sender: { role: RoleType.Tool },
                timestamp: e.ts,
                content: [
                    {
                        kind: "tool_result",
                        callId: e.correlation,
                        // The structured payload `send` stashed in meta is the real
                        // result; fall back to the text content only if it's absent.
                        result: "result" in meta ? meta.result : e.content,
                        isError: meta.isError === true ? true : undefined,
                    },
                ],
            });
        }
        // Any other kind (recall, dream, system bookkeeping) isn't part of the
        // replayable conversation; leave it to transcript_recall.
    }
    return messages;
}

/** Render a tool result into the log's text `content` column. A string passes
 *  through verbatim; anything else is JSON-stringified (the structured form is
 *  also kept in the event's `meta`, so this is for lexical/FTS search, not the
 *  authoritative payload). Falls back to String() for a value JSON can't
 *  represent (a circular object, a BigInt), and to a placeholder for the empty
 *  string so the log's non-empty-content rule never rejects a tool's result. */
function stringifyResult(result: unknown): string {
    if (typeof result === "string") return result.length ? result : "(empty)";
    if (result === undefined) return "(no result)";
    try {
        const json = JSON.stringify(result);
        return json && json.length ? json : String(result);
    } catch {
        return String(result);
    }
}
