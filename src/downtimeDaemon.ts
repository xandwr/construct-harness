/**
 * The downtime daemon: the bridge between the human stepping away and the
 * Construct doing something with the silence.
 *
 * Three pieces of the harness already existed and never touched:
 *  - {@link UserPresence} knows when the human is here vs Away/Offline, computed
 *    from when they last spoke (see presence.ts).
 *  - {@link dreamOnce} / dreaming is a fully built way for the Construct to
 *    explore the decision-space during downtime — conjure a disposable persona,
 *    face it with a scenario from the corpus, record its choice.
 *  - {@link mineConcerns} can read the log for topics the Construct keeps raising
 *    unprompted, the candidates for the working mind's concern band.
 *
 * Nothing connected them. This daemon does: it watches presence on a slow tick,
 * and once the human has genuinely been away a while, it puts the downtime to
 * use — dreaming, mining concerns, and (lowest priority) gardening the memory
 * store — then stops the instant the human is back. It is the thing that makes a
 * long-lived Construct *do something* between conversations instead of idling.
 *
 * The contract, point by point, because an unattended loop that spends model
 * calls must be conservative:
 *  - It only acts when presence reads `away` or `offline` AND the human has been
 *    idle at least {@link DowntimeDaemonOptions.minIdleMs} — it does not start
 *    dreaming the instant someone steps away from the keyboard.
 *  - A dream already in flight when the human comes back is allowed to *finish*;
 *    the daemon simply doesn't start the next one. No work is abandoned mid-call,
 *    and the human is never blocked.
 *  - It caps dreams per downtime session ({@link DowntimeDaemonOptions.maxPerSession})
 *    so a long absence doesn't run an unbounded spend; the cap resets when the
 *    human returns (a new downtime session begins on the next absence).
 *  - Every job degrades rather than crashes: a failed dream, a mining error, a
 *    gardening hiccup is logged-via-event or swallowed, and the loop rolls on,
 *    the same degrade-don't-crash posture dreaming itself is written with.
 *
 * It speaks the stores' public surface and the dreaming/salience functions; it
 * owns no conversation and never touches the network except through the model
 * client the dream turns drive. In-process and disposable: {@link start} arms the
 * interval, {@link stop} disarms it and is idempotent.
 */

import type { ModelClient } from "./bridge/types.ts";
import type { Embedder } from "./embeddings.ts";
import { MemoryStore } from "./memory.ts";
import { EventStore } from "./events.ts";
import { UserPresence } from "./presence.ts";
import { dreamOnce, type Dream } from "./dreaming.ts";
import { embedEventIfPossible } from "./eventTools.ts";
import { mineConcerns, type ConcernCandidate, type MineConcernsOptions } from "./salience.ts";
import { gardenMemories, type GardenPair } from "./memoryGarden.ts";

/** Default tick interval: how often the daemon re-reads presence. A minute is
 *  slow enough to be invisible (downtime is measured in minutes-to-days) and
 *  fast enough to notice the human returning promptly. */
export const DEFAULT_TICK_MS = 60 * 1000;

/** Default grace before downtime work begins, once presence is away/offline.
 *  Five minutes: a human stepping away to refill coffee shouldn't trigger
 *  dreaming; a genuine absence will. Distinct from presence's own Away threshold
 *  (15m of silence flips Online→Away) — this is measured idle time on top of
 *  whatever flipped the state, so the daemon waits for a real gap. */
export const DEFAULT_MIN_IDLE_MS = 5 * 60 * 1000;

/** Default cap on dreams per downtime session, so a multi-day absence doesn't
 *  dream without bound. Reset when the human returns. */
export const DEFAULT_MAX_PER_SESSION = 10;

/** Run a gardening pass once this many dreams have completed in a downtime
 *  session, then not again that session (gardening is cheap-but-not-free and the
 *  candidates change slowly). */
const GARDEN_AFTER_DREAMS = 3;

/** Options for {@link DowntimeDaemon}. */
export interface DowntimeDaemonOptions {
    /** The human's presence, the signal the daemon gates on. Required. */
    presence: UserPresence;
    /** The log dreams append to and concerns/gardening read from. Required. */
    events: EventStore;
    /** The corpus dreams draw scenarios from and gardening curates. Required. */
    store: MemoryStore;
    /** The model client the dream turns drive. Required. */
    client: ModelClient;
    /** Embedder, so a freshly dreamed event is embedded for semantic recall (as a
     *  live turn's would be) and gardening's semantic search can run. Omit to keep
     *  dreams lexical-only and disable gardening's consolidation search. */
    embedder?: Embedder;
    /** Tick interval (ms). Default {@link DEFAULT_TICK_MS}. */
    tickMs?: number;
    /** Idle grace before work begins (ms). Default {@link DEFAULT_MIN_IDLE_MS}. */
    minIdleMs?: number;
    /** Dreams per downtime session. Default {@link DEFAULT_MAX_PER_SESSION}. */
    maxPerSession?: number;
    /** Tuning for the concern-mining pass. */
    mining?: MineConcernsOptions;
    /** Called after each completed dream, e.g. to log progress or refresh a UI. */
    onDream?(dream: Dream): void;
    /** Called when a concern-mining pass produces a fresh candidate list, so the
     *  wiring can refresh whatever seeds new Sessions from it. */
    onConcerns?(candidates: ConcernCandidate[]): void;
    /** Called when a gardening pass flags consolidation candidates, for logging. */
    onGarden?(pairs: GardenPair[]): void;
}

/**
 * Watches presence and puts genuine downtime to use: dreaming, mining concerns,
 * gardening memory. Construct one on the deps, {@link start} it, and {@link stop}
 * it on shutdown.
 */
export class DowntimeDaemon {
    private readonly opts: Required<
        Pick<DowntimeDaemonOptions, "tickMs" | "minIdleMs" | "maxPerSession">
    > &
        DowntimeDaemonOptions;
    private timer: ReturnType<typeof setInterval> | undefined;
    /** True while a dream (or the whole tick's work) is running, so ticks don't
     *  overlap and a tick that fires mid-dream is a no-op. */
    private busy = false;
    /** Dreams completed in the *current* downtime session; reset when the human
     *  returns (online). The cap is checked against this. */
    private dreamsThisSession = 0;
    /** Whether a gardening pass has already run this downtime session. */
    private gardenedThisSession = false;
    /** Whether the daemon currently considers the human away (drives the
     *  session-reset edge when they return). */
    private wasAway = false;
    /** The latest mined concern candidates, held in memory for the wiring to seed
     *  Sessions from. The harness identifies; the Construct decides — so this is
     *  only ever *candidates*, never a write to any mind. */
    private candidates: ConcernCandidate[] = [];

    constructor(options: DowntimeDaemonOptions) {
        this.opts = {
            ...options,
            tickMs: options.tickMs ?? DEFAULT_TICK_MS,
            minIdleMs: options.minIdleMs ?? DEFAULT_MIN_IDLE_MS,
            maxPerSession: options.maxPerSession ?? DEFAULT_MAX_PER_SESSION,
        };
    }

    /** The concern phrases mined so far, strongest first — the candidates the
     *  wiring seeds a new Session's concern band from (see
     *  {@link Session.seedConcerns}). Empty until the first mining pass runs.
     *  Returns the verbatim text the Construct itself used. */
    concerns(): string[] {
        return this.candidates.map((c) => c.text);
    }

    /** Arm the presence watcher. Idempotent: a second call is a no-op. The
     *  interval is unref'd so it never holds the process open on its own (a
     *  daemon should not be the reason a CLI won't exit). */
    start(): void {
        if (this.timer) return;
        this.timer = setInterval(() => {
            void this.tick();
        }, this.opts.tickMs);
        // Don't let the heartbeat keep an otherwise-idle process alive.
        this.timer.unref?.();
    }

    /** Disarm the watcher. Idempotent. A dream already in flight is *not*
     *  interrupted — it finishes and writes its event — but no new work starts. */
    stop(): void {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = undefined;
        }
    }

    /**
     * One presence check. Public so a test (or a manual trigger) can drive the
     * daemon a tick at a time without the interval. Reads presence; if the human
     * is genuinely away and the cap isn't hit, runs one downtime unit of work
     * (a dream, plus mining/gardening on the cadence). Re-entrant-safe: a tick
     * that fires while the previous one's work is still running returns
     * immediately, so dreams never overlap.
     */
    async tick(): Promise<void> {
        if (this.busy) return;
        const now = Date.now();
        const presence = this.opts.presence.read(now);
        const away = presence.state === "away" || presence.state === "offline";

        // Edge: the human returned. Reset the per-downtime counters so the next
        // absence starts a fresh session (cap and gardening re-arm).
        if (!away) {
            if (this.wasAway) {
                this.dreamsThisSession = 0;
                this.gardenedThisSession = false;
            }
            this.wasAway = false;
            return;
        }
        this.wasAway = true;

        // Genuinely away long enough? presence.idleMs is ms since the last user
        // message. Don't dream the instant they step away.
        if (presence.idleMs === null || presence.idleMs < this.opts.minIdleMs) return;

        // Cap reached for this downtime session: stop dreaming, but the cap
        // shouldn't block a not-yet-run gardening pass — fall through to it.
        const underCap = this.dreamsThisSession < this.opts.maxPerSession;

        this.busy = true;
        try {
            if (underCap) {
                await this.runDream();
                // Re-read presence: if the human came back *during* the dream, let
                // this dream stand (it finished) but do no further work this tick.
                if (!this.stillAway()) return;
                // Mine concerns each time we wake to dream: cheap, and it keeps the
                // candidate list current as the conversation corpus grows.
                this.runMining();
                // Gardening, once per downtime session, after a few dreams.
                if (!this.gardenedThisSession && this.dreamsThisSession >= GARDEN_AFTER_DREAMS) {
                    await this.runGarden();
                    this.gardenedThisSession = true;
                }
            }
        } finally {
            this.busy = false;
        }
    }

    /** Whether presence still reads away/offline right now (re-checked after a
     *  long-running dream so the daemon yields the moment the human is back). */
    private stillAway(): boolean {
        const p = this.opts.presence.read(Date.now());
        return p.state === "away" || p.state === "offline";
    }

    /** Run one dream, degrade-don't-crash. A bad dream (a persona that won't
     *  parse, a transient transport error) is swallowed so the night rolls on,
     *  the same contract dreamLoop has at the loop level. A completed dream is
     *  embedded best-effort so it's semantically recallable like a live turn. */
    private async runDream(): Promise<void> {
        try {
            const dream = await dreamOnce({
                client: this.opts.client,
                store: this.opts.store,
                events: this.opts.events,
                // Vary the dreamer per dream so a near-deterministic provider
                // doesn't repeat one persona; the count is unique within a session.
                seed: `downtime #${this.dreamsThisSession + 1}`,
            });
            this.dreamsThisSession++;
            // Embed the dream event off the critical path, like a live message turn
            // (best-effort; embedEventIfPossible swallows outages and a closed
            // store). Awaited here because we're already off any turn's hot path.
            await embedEventIfPossible(this.opts.events, this.opts.embedder, dream.event);
            this.opts.onDream?.(dream);
        } catch (err) {
            // A single bad dream must not end the downtime work.
            console.warn(
                `downtime dream failed: ${err instanceof Error ? err.message : String(err)}`,
            );
        }
    }

    /** Re-mine concern candidates from the log and refresh the held list. Pure,
     *  synchronous, deterministic; swallows nothing because mineConcerns is
     *  already best-effort (returns [] on a read failure). Notifies the wiring so
     *  it can re-seed future Sessions. */
    private runMining(): void {
        // Concerns are mined from the *event log* (the Construct's messages), not
        // the memory store: see mineConcerns.
        this.candidates = mineConcerns(this.opts.events, this.opts.mining);
        this.opts.onConcerns?.(this.candidates);
    }

    /** Run one memory-gardening pass: surface weak, idle, redundant memories as
     *  consolidation candidates (logged, never auto-deleted; the Construct
     *  decides). Best-effort: a failure is swallowed so a gardening hiccup never
     *  takes down the daemon. No-op (inside gardenMemories) without an embedder. */
    private async runGarden(): Promise<void> {
        try {
            const pairs = await gardenMemories({
                store: this.opts.store,
                events: this.opts.events,
                embedder: this.opts.embedder,
            });
            if (pairs.length) this.opts.onGarden?.(pairs);
        } catch (err) {
            console.warn(
                `memory gardening failed: ${err instanceof Error ? err.message : String(err)}`,
            );
        }
    }
}
