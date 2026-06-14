/**
 * Passive context: content the harness folds into each model turn without the
 * model (or the caller) asking for it.
 *
 * Two things motivate this module:
 *
 *  1. Some context is *temporal*: it must be recomputed every turn because its
 *     value changes as the conversation runs (the current date and time is the
 *     canonical example: by the time a tool loop returns, minutes have passed).
 *     A static system `Message` built once, before {@link runLoop}, would freeze
 *     that value at boot.
 *
 *  2. Some context is *fixed but pervasive*: a standing instruction or fact we
 *     want present on every request regardless of conversation state.
 *
 * Both are "passive context providers": pure functions of a {@link ContextScope}
 * that return a {@link ContextContribution}, or `undefined` to contribute
 * nothing this turn. The loop evaluates them just before each `generate`, so the
 * conversation history the caller holds stays clean: the injected content lives
 * only on the wire, recomputed per turn.
 *
 * This module speaks only core types and has no provider or I/O dependency, in
 * keeping with the rest of `src/` (the bridge is where SDKs live, not here).
 */

import { RoleType } from "./types.ts";
import type { Message } from "./types.ts";

/**
 * What a provider sees when it's asked to contribute. Kept lean: the
 * conversation so far (read-only), the turn index, and an optional session-start
 * timestamp. Providers that need the wall clock read it themselves: passing a
 * frozen "now" in here would defeat the whole point for temporal providers,
 * which must observe time advancing. `sessionStart` is the exception — it's a
 * *fixed* reference point (when the conversation began), so it's safe to pass in,
 * and a provider can't recover it from `messages` alone once early turns are
 * compacted away.
 */
export interface ContextScope {
    /** The conversation as it stands this turn, oldest first. Read-only. */
    readonly messages: readonly Message[];
    /** 0-based index of the model turn about to run (0 = first call). */
    readonly turn: number;
    /**
     * Epoch-ms the conversation began, when the caller knows it (a Session passes
     * its own start). Lets a provider report how long the session has run without
     * relying on the first message's timestamp, which compaction may have dropped.
     * Undefined when the caller doesn't track it.
     */
    readonly sessionStart?: number;
}

/**
 * What a provider hands back. Both fields are optional so a provider can target
 * either fold point (or, rarely, both):
 *
 *  - `system` text is appended to the system prompt for this turn. This is the
 *    right channel for ambient guidance and facts (temporal awareness included):
 *    it rides the cached prefix shape providers already use and doesn't clutter
 *    the turn array.
 *  - `messages` are injected into the outgoing turn array for this turn: use
 *    this only when the content must read as a conversational turn rather than
 *    system guidance (rare; most passive context belongs in `system`).
 */
export interface ContextContribution {
    system?: string;
    messages?: Message[];
}

/**
 * A passive context provider: pure, named, evaluated before every turn.
 * Returning `undefined` means "nothing to add this turn" and is distinct from
 * returning empty text: it lets a provider stay silent when it has nothing
 * relevant (e.g. a provider gated on conversation state).
 *
 * `contribute` may be synchronous (return the contribution directly) or async
 * (return a promise the loop awaits). Synchronous is the common, cheapest case —
 * the temporal provider just reads the clock — and stays the default. Async
 * exists for the provider that must read a store to know what to inject (an
 * agent's open goals, say); keep that read fast and indexed, because it sits on
 * the hot path ahead of *every* model turn. Anything heavier than a single keyed
 * lookup belongs in a tool or the one-time system build, not here.
 */
export interface ContextProvider {
    /** Stable identifier, for logging and so the same provider is recognizable
     *  across turns. */
    readonly name: string;
    contribute(
        scope: ContextScope,
    ): ContextContribution | undefined | Promise<ContextContribution | undefined>;
}

/**
 * Fold a turn's passive context onto the outgoing message list.
 *
 * Returns a *new* array: the caller's conversation is never mutated. System
 * contributions are appended to the conversation's existing system guidance as
 * additional `role: "system"` turns (the Anthropic mapper concatenates all
 * system text, so order within the system channel is append-order). Message
 * contributions are appended after the conversation so they sit closest to the
 * model's next turn, where standing reminders are most effective.
 *
 * Async because a provider may read a store to decide what to inject. Providers
 * run concurrently (they're independent), but their contributions are folded in
 * *provider order*, not completion order, so the system-text join is stable turn
 * to turn regardless of which lookup finished first. A provider that throws is
 * dropped for this turn rather than failing the turn: passive context is an
 * enhancement, never a gate on the conversation (mirrors the Session's
 * best-effort logging). Empty / whitespace-only system text is dropped so a
 * provider returning `""` doesn't inject a blank system turn.
 *
 * `sessionStart`, when given, is threaded into the {@link ContextScope} so a
 * temporal provider can report session duration.
 */
export async function applyContext(
    messages: readonly Message[],
    providers: readonly ContextProvider[],
    turn: number,
    sessionStart?: number,
): Promise<Message[]> {
    if (providers.length === 0) return [...messages];

    const scope: ContextScope = { messages, turn, sessionStart };

    // Run every provider, swallowing a single provider's failure to a `null`
    // contribution so one bad provider can't take down the turn. Order is
    // preserved by `Promise.all` resolving the array positionally.
    const contributions = await Promise.all(
        providers.map(async (provider) => {
            try {
                return (await provider.contribute(scope)) ?? null;
            } catch {
                return null;
            }
        }),
    );

    const systemTexts: string[] = [];
    const injected: Message[] = [];
    for (const contribution of contributions) {
        if (!contribution) continue;
        if (contribution.system && contribution.system.trim()) {
            systemTexts.push(contribution.system);
        }
        if (contribution.messages?.length) {
            injected.push(...contribution.messages);
        }
    }

    const out = [...messages];

    if (systemTexts.length) {
        out.push({
            sender: { role: RoleType.System, name: "context" },
            timestamp: Date.now(),
            content: [{ kind: "text", text: systemTexts.join("\n\n") }],
        });
    }
    out.push(...injected);

    return out;
}

// ── Temporal provider ─────────────────────────────────────────────────────────

/** Config for {@link temporalContext}. */
export interface TemporalOptions {
    /**
     * IANA timezone (e.g. `"Europe/Dublin"`) the time is rendered in. Defaults
     * to the host's resolved local timezone, so the harness reflects wherever it
     * runs. An invalid zone falls back to the host default rather than throwing
     *: passive context must never break a turn.
     */
    timeZone?: string;
    /** Locale for formatting. Defaults to `"en-US"` for stable, readable output
     *  independent of the host's locale. */
    locale?: string;
    /**
     * Also state how long since the previous turn and how long the session has
     * run, when those are derivable (a prior message timestamp; a
     * {@link ContextScope.sessionStart}). On by default — it's the cheap part of
     * temporal awareness, and the thing the bare wall-clock can't give the model.
     * Set false for just the absolute date/time.
     */
    elapsed?: boolean;
}

/** Render a millisecond span as a short, human relative phrase ("3 days", "an
 *  hour", "just now"). Coarse on purpose: the model reasons about scale, not
 *  precision, and a stable phrasing keeps the cached prefix from churning every
 *  second. Always non-negative; a clock skew that yields a negative span reads as
 *  "just now". */
export function humanizeDuration(ms: number): string {
    const s = Math.max(0, Math.round(ms / 1000));
    if (s < 45) return "just now";
    const mins = Math.round(s / 60);
    if (mins < 60) return mins <= 1 ? "a minute" : `${mins} minutes`;
    const hours = Math.round(mins / 60);
    if (hours < 24) return hours <= 1 ? "an hour" : `${hours} hours`;
    const days = Math.round(hours / 24);
    if (days < 7) return days <= 1 ? "a day" : `${days} days`;
    const weeks = Math.round(days / 7);
    if (days < 30) return weeks <= 1 ? "a week" : `${weeks} weeks`;
    const months = Math.round(days / 30);
    if (days < 365) return months <= 1 ? "a month" : `${months} months`;
    const years = Math.round(days / 365);
    return years <= 1 ? "a year" : `${years} years`;
}

/** The newest message timestamp in a turn's scope, or undefined when there are
 *  no messages yet (turn 0 of a fresh conversation). Reads the last entry: the
 *  conversation is oldest-first, so the tail is the most recent thing said. */
function lastMessageTime(messages: readonly Message[]): number | undefined {
    const last = messages[messages.length - 1];
    return last?.timestamp;
}

/** Resolve the host's local IANA timezone, with a hard fallback to UTC if the
 *  runtime can't report one. */
function hostTimeZone(): string {
    try {
        return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    } catch {
        return "UTC";
    }
}

/**
 * A passive provider that states the current date and time in the user's
 * timezone, recomputed every turn.
 *
 * The text is phrased as ambient fact ("The current date and time is …") rather
 * than instruction, since that's what it is: orientation the model can lean on
 * for relative-time reasoning ("tomorrow", "in an hour") without us having to
 * thread a clock through every prompt.
 *
 * The timezone and locale are validated once, at construction, by doing a
 * formatting probe; an invalid timezone degrades to the host default so a typo
 * in config never crashes a turn.
 */
export function temporalContext(options: TemporalOptions = {}): ContextProvider {
    const locale = options.locale ?? "en-US";
    const requested = options.timeZone ?? hostTimeZone();
    const elapsed = options.elapsed ?? true;

    // Probe the requested zone once; fall back to the host default if it's not a
    // zone the runtime recognizes. Doing this here (not per turn) keeps the hot
    // path allocation-light and surfaces a bad config at setup time.
    const timeZone = validTimeZone(requested) ? requested : hostTimeZone();

    const formatter = new Intl.DateTimeFormat(locale, {
        timeZone,
        dateStyle: "full",
        timeStyle: "long",
    });

    return {
        name: "temporal",
        contribute(scope) {
            const nowMs = Date.now();
            const lines = [
                `The current date and time in the user's timezone (${timeZone}) is ${formatter.format(new Date(nowMs))}.`,
            ];
            if (elapsed) {
                // Time since the previous turn: lets the model tell a follow-up a
                // few seconds later from one resumed after a long gap. Skipped on
                // the opening turn (nothing came before it).
                const last = lastMessageTime(scope.messages);
                if (last !== undefined && nowMs - last >= 0) {
                    const ago = humanizeDuration(nowMs - last);
                    if (ago !== "just now") {
                        lines.push(`The previous message was ${ago} ago.`);
                    }
                }
                // How long this conversation has been going, when the caller
                // tracks its start. Orientation for "earlier today" vs a session
                // spanning days.
                if (scope.sessionStart !== undefined && nowMs - scope.sessionStart >= 0) {
                    const dur = humanizeDuration(nowMs - scope.sessionStart);
                    if (dur !== "just now") {
                        lines.push(`This conversation has been running for ${dur}.`);
                    }
                }
            }
            return { system: lines.join("\n") };
        },
    };
}

/** True if the runtime accepts `tz` as an IANA timezone. `Intl` throws a
 *  `RangeError` for unknown zones, which we treat as "invalid". */
function validTimeZone(tz: string): boolean {
    try {
        new Intl.DateTimeFormat("en-US", { timeZone: tz });
        return true;
    } catch {
        return false;
    }
}
