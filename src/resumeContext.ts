/**
 * The resume catch-up: a one-shot context contribution that tells the Construct
 * what happened *while it was away*, on the first turn of a conversation resumed
 * after a real gap.
 *
 * The harness already tells the Construct *how long* the gap was — the temporal
 * provider reports "the previous message was 3 days ago". What it never told it
 * was *what happened in those three days*. But things did: the downtime daemon
 * dreamed (a disposable persona facing a scenario from the corpus; see
 * {@link ./dreaming.ts}), and goals may have changed status. Those are real
 * events on the log, in the gap window, that the Construct slept through.
 *
 * So this provider fires exactly once — turn 0 of a session whose last activity
 * is older than {@link RESUME_THRESHOLD_MS} — and assembles a short "While you
 * were away" block from the log: the dreams it had and the goals that moved,
 * since it last spoke. It is the symmetric partner to {@link ./dreamContext}:
 * that pushes the single freshest dream every turn; this catches the Construct up
 * on the *batch* of dreams (and goal changes) that accumulated during the
 * absence, the first time it wakes back up.
 *
 * Crucially this is NOT a model call. It is pure, structured text assembly from
 * log rows — query the dream/goal events in the gap window, format their already-
 * recorded fields (the persona's name, the scenario theme, a goal's new status).
 * Like the working mind, it never authors content the Construct didn't produce:
 * a dreamed persona's name is the name the dream loop stored, a goal's status is
 * the status the store recorded. The harness only *arranges* what's already on
 * the log, framed as orientation the Construct can lean on, not an instruction.
 *
 * Speaks only core types, the {@link EventStore}'s read surface, and the two
 * event-kind constants (dream, goal): no provider, no model, no I/O beyond the
 * bounded, indexed log reads. Best-effort throughout: a store read failure
 * contributes nothing rather than gating the turn (the provider mechanism also
 * drops a throwing provider), because passive context must never break a resume.
 */

import type { ContextProvider, ContextScope } from "./context.ts";
import { humanizeDuration } from "./context.ts";
import { EventStore } from "./events.ts";
import type { Event } from "./events.ts";
import { DREAM_EVENT_KIND } from "./dreaming.ts";
import { GOAL_EVENT_KIND } from "./goals.ts";
import type { GoalEventMeta, GoalStatus } from "./goals.ts";
import { dreamEventToView } from "./dreamTools.ts";
import type { Personality } from "./critics.ts";

/**
 * How long a conversation must have been idle before the catch-up fires. Thirty
 * minutes: shorter than that and the Construct didn't really "sleep" (the daemon
 * needs a stretch of away-ness before it dreams at all, see the downtime daemon's
 * `minIdleMs`), and a few-minute pause is already covered by the temporal
 * provider's "previous message N ago". This gate is about a genuine absence, the
 * kind that accumulates dreams worth catching up on.
 */
export const RESUME_THRESHOLD_MS = 30 * 60 * 1000;

/** The largest number of dreams the catch-up enumerates by name before summing
 *  the rest as "and N more". A long absence can pile up many dreams; listing all
 *  of them would bury the turn, so we name a handful and count the tail. */
const MAX_DREAMS_LISTED = 5;

/** Per-field cap so one verbose dream theme can't blow out the block. The view's
 *  scenario is already capped by dreamEventToView; this is a tighter trim for the
 *  single-line "theme" we render here. */
const THEME_CAP = 120;

/** Options for {@link resumeContext}. */
export interface ResumeContextOptions {
    /**
     * Idle threshold (ms) above which the catch-up fires. Defaults to
     * {@link RESUME_THRESHOLD_MS}. Lower it to catch up after shorter gaps (a
     * test, an aggressive daemon); raise it to only ever catch up after long
     * absences.
     */
    thresholdMs?: number;
    /** Max dreams listed by name before the rest are summed. Defaults to
     *  {@link MAX_DREAMS_LISTED}. */
    maxDreamsListed?: number;
}

/**
 * Render a dreamed persona's handle for the catch-up line: their name, and their
 * role when they have one ("Vera Ostrakh, a cartographer"). Mirrors the dream
 * injection's terse handle; kept local so this module doesn't reach into
 * dreamTools' internals.
 */
function personaHandle(p: Personality): string {
    const name = typeof p.name === "string" && p.name.trim() ? p.name.trim() : "someone";
    const role = typeof p.role === "string" && p.role.trim() ? p.role.trim() : "";
    return role ? `${name}, ${role}` : name;
}

/** Collapse a scenario into a one-clause theme: its first sentence (or first
 *  clause), trimmed to {@link THEME_CAP}. The catch-up wants the *gist* of what
 *  the dream was about ("an opacity dilemma"), not the whole dilemma prose, which
 *  the Construct can recall in full via dream_recall if it wants it. */
function dreamTheme(scenario: string): string {
    const trimmed = scenario.trim();
    if (!trimmed) return "";
    // First sentence end, else first newline, else the whole thing.
    const sentence = trimmed.search(/[.!?](\s|$)/);
    const firstLine = trimmed.indexOf("\n");
    let cut = trimmed.length;
    if (sentence !== -1) cut = Math.min(cut, sentence + 1);
    if (firstLine !== -1) cut = Math.min(cut, firstLine);
    let theme = trimmed.slice(0, cut).trim();
    if (theme.length > THEME_CAP) theme = theme.slice(0, THEME_CAP).trim() + "…";
    return theme;
}

/** Render one dream event as a "dreamed as X (theme)" line, or null when it has
 *  no usable persona/theme (a malformed dream row degrades to skipped, not a
 *  crash). */
function dreamLine(e: Event): string | null {
    const view = dreamEventToView(e);
    const handle = personaHandle(view.persona);
    if (handle === "someone" && !view.scenario.trim()) return null;
    const theme = dreamTheme(view.scenario);
    return theme ? `${handle} (${theme})` : handle;
}

/** Read a goal-change event's structured meta defensively. The EventStore
 *  degrades a corrupt meta to undefined; an unexpected shape yields undefined
 *  fields rather than throwing. */
function goalMeta(e: Event): Partial<GoalEventMeta> {
    const m = (e.meta ?? {}) as Record<string, unknown>;
    const change = typeof m.change === "string" ? (m.change as GoalEventMeta["change"]) : undefined;
    const status = typeof m.status === "string" ? (m.status as GoalStatus) : undefined;
    return { change, status };
}

/**
 * Summarize the goal changes in the window into one human line, or null when
 * nothing goal-related happened. Counts the lifecycle moves that matter to "what
 * changed while I was away": goals completed, abandoned, and newly set. An
 * in-place edit isn't surfaced (the text changing isn't a status the Construct
 * needs to re-orient around).
 */
function goalSummary(events: Event[]): string | null {
    let completed = 0;
    let abandoned = 0;
    let added = 0;
    for (const e of events) {
        const { change, status } = goalMeta(e);
        if (change === "created") added++;
        else if (change === "status") {
            if (status === "done") completed++;
            else if (status === "abandoned") abandoned++;
        }
    }
    const parts: string[] = [];
    if (completed) parts.push(`${completed} goal${completed === 1 ? "" : "s"} completed`);
    if (abandoned) parts.push(`${abandoned} abandoned`);
    if (added) parts.push(`${added} newly set`);
    if (parts.length === 0) return null;
    return parts.join(", ") + ".";
}

/**
 * The last activity timestamp this turn resumes *after*: the timestamp of the
 * message before the one being sent now.
 *
 * The subtlety: by the time a provider runs, the turn's *current* user message is
 * already the tail of `scope.messages` (the Session appends it before the model
 * call), stamped ~now. So the tail is "now", not the gap we want — the gap is
 * between the *previous* conversation's last message and now. That previous
 * message is the second-to-last entry. Undefined when there's no prior message
 * (a fresh conversation, or one whose only message is the turn being sent), in
 * which case there is nothing to catch up after.
 */
function lastActivityTs(scope: ContextScope): number | undefined {
    const prior = scope.messages[scope.messages.length - 2];
    return prior?.timestamp;
}

/**
 * Build the catch-up provider: a one-shot "While you were away" contribution for
 * the first turn of a conversation resumed after a gap longer than the threshold.
 *
 * Gated three ways, all of which must hold (matching the design): it is the
 * opening turn (`scope.turn === 0`), the caller tracks a session start
 * (`scope.sessionStart !== undefined`, the mark of a real Session, not a
 * throwaway), and the last activity is older than the threshold. When any fails
 * it contributes nothing — so a fresh conversation, a continuing turn, or a
 * quick follow-up never sees the block.
 *
 * When it does fire, it queries the dream and goal-change events in the window
 * `[lastActivity, now]` and formats them. If neither produced anything (the
 * Construct was away but didn't dream and no goal moved), it stays silent rather
 * than inject an empty "while you were away: nothing" — there's nothing to catch
 * up on. Not session-scoped on the dream read (dreams belong to the Construct as
 * a whole); the goal read spans the window too, picking up both global and this
 * session's goal changes.
 */
export function resumeContext(
    store: EventStore,
    options: ResumeContextOptions = {},
): ContextProvider {
    const thresholdMs = options.thresholdMs ?? RESUME_THRESHOLD_MS;
    const maxDreamsListed = options.maxDreamsListed ?? MAX_DREAMS_LISTED;

    return {
        name: "resume",
        contribute(scope) {
            // Gate 1: opening turn only. The catch-up is for waking up, not mid-
            // conversation.
            if (scope.turn !== 0) return undefined;
            // Gate 2: a real Session tracks its start; a throwaway (the dream
            // inner Sessions, persona generation) passes no sessionStart and should
            // never get a catch-up block.
            if (scope.sessionStart === undefined) return undefined;
            // Gate 3: a genuine gap. No prior activity ⇒ nothing to resume after.
            const last = lastActivityTs(scope);
            if (last === undefined) return undefined;
            const now = Date.now();
            const gap = now - last;
            if (gap <= thresholdMs) return undefined;

            // Pure log assembly from the gap window. Best-effort: a store read
            // failure contributes nothing rather than breaking the resume.
            let dreams: Event[];
            let goalEvents: Event[];
            try {
                // Dreams aren't session-scoped (they belong to the Construct as a
                // whole), so read every dream since the gap began, newest first.
                dreams = store.recent({ kind: DREAM_EVENT_KIND, since: last + 1 });
                goalEvents = store.recent({ kind: GOAL_EVENT_KIND, since: last + 1 });
            } catch {
                return undefined;
            }

            const lines: string[] = [];

            if (dreams.length) {
                const rendered = dreams.map(dreamLine).filter((l): l is string => l !== null);
                if (rendered.length) {
                    const shown = rendered.slice(0, maxDreamsListed);
                    const extra = rendered.length - shown.length;
                    const tail = extra > 0 ? `, and ${extra} more` : "";
                    lines.push(`- Dreamed as ${shown.join("; ")}${tail}.`);
                }
            }

            const goals = goalSummary(goalEvents);
            if (goals) lines.push(`- Goals: ${goals}`);

            // Nothing actually happened in the gap worth surfacing: stay silent.
            if (lines.length === 0) return undefined;

            const away = humanizeDuration(gap);
            const header = `While you were away (${away}), here's what happened — orientation as you pick this conversation back up, not something to act on:`;
            return { system: `${header}\n${lines.join("\n")}` };
        },
    };
}
