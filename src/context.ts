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
 * What a provider sees when it's asked to contribute. Deliberately minimal: the
 * conversation so far (read-only) and the turn index. Providers that need the
 * wall clock read it themselves: passing a frozen "now" in here would defeat
 * the whole point for temporal providers, which must observe time advancing.
 */
export interface ContextScope {
    /** The conversation as it stands this turn, oldest first. Read-only. */
    readonly messages: readonly Message[];
    /** 0-based index of the model turn about to run (0 = first call). */
    readonly turn: number;
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
 * A passive context provider: pure, synchronous, named. Returning `undefined`
 * means "nothing to add this turn" and is distinct from returning empty text:
 * it lets a provider stay silent when it has nothing relevant (e.g. a provider
 * gated on conversation state).
 *
 * Kept synchronous on purpose: passive context is evaluated on the hot path
 * before *every* turn, so it must not block on I/O. Anything that needs a
 * network or disk read belongs in a tool or in the one-time system build, not
 * here.
 */
export interface ContextProvider {
    /** Stable identifier, for logging and so the same provider is recognizable
     *  across turns. */
    readonly name: string;
    contribute(scope: ContextScope): ContextContribution | undefined;
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
 * Empty / whitespace-only system text is dropped so a provider returning `""`
 * doesn't inject a blank system turn.
 */
export function applyContext(
    messages: readonly Message[],
    providers: readonly ContextProvider[],
    turn: number,
): Message[] {
    if (providers.length === 0) return [...messages];

    const scope: ContextScope = { messages, turn };
    const systemTexts: string[] = [];
    const injected: Message[] = [];

    for (const provider of providers) {
        const contribution = provider.contribute(scope);
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
        contribute() {
            const now = formatter.format(new Date());
            return {
                system: `The current date and time in the user's timezone (${timeZone}) is ${now}.`,
            };
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
