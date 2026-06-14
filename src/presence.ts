/**
 * UserPresence: the harness's read on whether the human is here.
 *
 * A long-lived Construct outlives any one conversation, and the daemon can keep
 * running while the person who talks to it walks away. This holds the answer to
 * "is the human around right now?" the way Discord shows a presence dot: Online
 * when they're actively talking, Away after a stretch of silence, Do Not Disturb
 * when they've said don't bother me, Offline when they're gone.
 *
 * The signal is *messages, not machine activity*. The harness never watches the
 * keyboard or the window; it only knows the human is here because they sent
 * something. So presence is computed from two inputs:
 *
 *  - **last activity** — the timestamp of the most recent user message. Touched
 *    on every turn the human sends (see the chat route). Recent ⇒ Online; after
 *    {@link AWAY_AFTER_MS} of silence ⇒ Away. This is the automatic axis.
 *  - **a manual override** — a state the human pinned by hand: `dnd` to say
 *    "I'm here but don't interrupt", or `offline` to deliberately appear gone.
 *    An override wins over the automatic axis and stays until cleared, matching
 *    how Discord's manual statuses are sticky while Away is derived.
 *
 * The one subtlety is that sending a message means "I'm here". So a fresh user
 * turn clears an `offline` override (you can't be talking and offline), but
 * preserves `dnd` (you can be here, talking, and still not want to be pinged) —
 * see {@link touch}. Online and Away are never stored as overrides; they're what
 * the automatic axis computes, so picking "Online" in the UI just clears any
 * override back to automatic.
 *
 * Deliberately in-process and dependency-light: no database, no node imports. The
 * state is a timestamp and an optional string, held on one object the server
 * keeps for its lifetime. Durable presence (surviving a restart) and Offline
 * driven by the client actually disconnecting are later goals; today the client
 * and daemon launch together, so a booted process means a present human, and the
 * default automatic status is Online.
 */

/** The presence states, in the order a UI would list them. `online` and `away`
 *  are computed from activity; `dnd` and `offline` are also valid manual
 *  overrides a human can pin (see {@link Override}). */
export const PRESENCE_STATES = ["online", "away", "dnd", "offline"] as const;
export type PresenceState = (typeof PRESENCE_STATES)[number];

/** The states a human may pin by hand. `online` clears back to the automatic
 *  axis (Online/Away by activity) rather than freezing Online forever, so it is
 *  spelled `auto` here — the absence of an override. `away` is intentionally not
 *  pinnable: it's a derived state, the thing the automatic axis says after
 *  silence, not a status you announce. */
export type Override = "dnd" | "offline";

/** How long after the last user message the automatic axis flips Online → Away.
 *  Fifteen minutes, matching the goal: "no message for 15 minutes or more →
 *  Away". A read past this with no override and no newer activity reports Away. */
export const AWAY_AFTER_MS = 15 * 60 * 1000;

/** The computed presence at an instant: the state to show plus the inputs behind
 *  it, so a caller (the status route, a UI) can render the dot and explain it
 *  ("away — last message 22m ago") without recomputing. */
export interface Presence {
    /** The state to display: the override if one is pinned, else the automatic
     *  Online/Away by activity. */
    state: PresenceState;
    /** Whether {@link state} comes from a pinned override (true) or the automatic
     *  activity axis (false). Lets a UI mark a manual status distinctly. */
    manual: boolean;
    /** The override currently pinned, or null when on the automatic axis. */
    override: Override | null;
    /** Epoch-ms of the most recent user message, or null if the human hasn't
     *  spoken this process yet. */
    lastActiveTs: number | null;
    /** Milliseconds since {@link lastActiveTs} at the moment this was read, or
     *  null when there's been no activity yet. */
    idleMs: number | null;
}

/**
 * Holds the human's presence for one server process and computes it on read.
 *
 * Construct one per server (held on the deps). Call {@link touch} whenever a user
 * message arrives; call {@link setOverride} from the presence write route; call
 * {@link read} from the presence read route. All three take an explicit `now`
 * (epoch-ms) so the logic is pure and a test can drive the clock — the routes
 * pass `Date.now()`.
 */
export class UserPresence {
    /** Epoch-ms of the most recent user message, or null until the first turn. */
    private lastActiveTs: number | null;
    /** The pinned manual override, or null when on the automatic axis. */
    private override: Override | null = null;

    /**
     * @param bootTs Treat the process boot as the human's first activity, so a
     *   freshly launched daemon reads Online rather than "never seen" (the client
     *   and daemon launch together today, so a boot means a present human). Pass
     *   null to start with no activity, e.g. for a process that should appear
     *   Offline until the first real message. The routes pass `Date.now()`.
     */
    constructor(bootTs: number | null = null) {
        this.lastActiveTs = bootTs;
    }

    /**
     * Record that the human just sent a message: they're here now. Advances the
     * automatic axis (this `now` becomes the new last-activity, so the read is
     * Online again and the Away countdown restarts) and clears an `offline`
     * override — you can't be talking and offline. A `dnd` override is preserved:
     * being present and not-to-be-disturbed are compatible, so DND survives until
     * the human lifts it themselves.
     */
    touch(now: number): void {
        this.lastActiveTs = now;
        if (this.override === "offline") this.override = null;
    }

    /**
     * Pin a manual override, or clear back to the automatic axis. `dnd` and
     * `offline` pin that state until changed; `online` (or null) clears any
     * override so presence follows activity again — picking "Online" means "stop
     * overriding", not "freeze Online", since the automatic axis already reports
     * Online right after a message. `away` is rejected: it's a derived state, not
     * one you announce. Returns the freshly computed {@link Presence}.
     *
     * @throws {RangeError} if `want` isn't one of online/dnd/offline.
     */
    setOverride(want: "online" | Override, now: number): Presence {
        if (want === "online") {
            this.override = null;
        } else if (want === "dnd" || want === "offline") {
            this.override = want;
        } else {
            throw new RangeError(
                `cannot set presence to "${want}": only online (clears the override), dnd, or offline`,
            );
        }
        return this.read(now);
    }

    /**
     * Compute the presence to show at `now`. A pinned override wins and is
     * reported as manual. Otherwise the automatic axis: Online while the last
     * message is within {@link AWAY_AFTER_MS}, Away once it isn't (and Away too
     * when there's been no activity at all — a process that never heard from the
     * human isn't "online").
     */
    read(now: number): Presence {
        const idleMs = this.lastActiveTs === null ? null : Math.max(0, now - this.lastActiveTs);
        if (this.override !== null) {
            return {
                state: this.override,
                manual: true,
                override: this.override,
                lastActiveTs: this.lastActiveTs,
                idleMs,
            };
        }
        const automatic: PresenceState =
            idleMs !== null && idleMs < AWAY_AFTER_MS ? "online" : "away";
        return {
            state: automatic,
            manual: false,
            override: null,
            lastActiveTs: this.lastActiveTs,
            idleMs,
        };
    }
}
