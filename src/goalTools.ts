/**
 * Bridges the {@link GoalStore} into the agentic loop, the way memoryTools and
 * noteTools bridge their stores.
 *
 * Two channels, mirroring memory:
 *  - {@link goalTools} builds the `ToolDef`s the model calls to set a goal, mark
 *    one done or abandoned, edit one, or list them during a run.
 *  - {@link goalContext} is the passive provider that injects the *active* goals
 *    into the system prompt every turn, so the agent pursues them without having
 *    to ask what they were. This is the load-bearing half: a goal the model has
 *    to recall on its own is a goal it will drift from; a goal standing in front
 *    of it every turn is one it holds.
 *
 * Goals are scoped to the Session that set them (its id), so one conversation's
 * intent doesn't bleed into another's. The tools speak plain JSON in and out and
 * translate {@link GoalError} into a clean message the model can read.
 */

import type { ToolDef } from "./types.ts";
import { Goal, GoalError, GoalStore, isGoalStatus } from "./goals.ts";
import type { GoalStatus } from "./goals.ts";
import type { ContextProvider } from "./context.ts";

/** How many active goals goalContext injects, and the tools list by default. A
 *  Construct juggling more than a handful of live goals has a focus problem the
 *  harness shouldn't paper over; the cap keeps the per-turn injection bounded. */
export const DEFAULT_GOAL_LIMIT = 12;

/** The serializable view of a goal handed back to the model. */
export interface GoalView {
    id: number;
    content: string;
    status: GoalStatus;
    created: number;
    updated: number;
}

function toView(g: Goal): GoalView {
    return {
        id: g.id,
        content: g.content,
        status: g.status,
        created: g.created,
        updated: g.updated,
    };
}

/** Narrow an unknown args bag to a record without trusting its fields yet. */
function asRecord(args: unknown): Record<string, unknown> {
    return typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
}

/**
 * Build the goal tool set over a store, scoped to one session.
 *
 * `sessionId` scopes every goal this set creates and lists, so the agent's goals
 * belong to *this* conversation. Pass undefined for a store-global set (e.g. a
 * single-session REPL). The loop's own arg validation enforces `required`; these
 * handlers defend the rest and translate {@link GoalError} into a readable result.
 */
export function goalTools(store: GoalStore, sessionId?: string): ToolDef[] {
    const session = sessionId;

    const set: ToolDef = {
        name: "goal_set",
        description:
            "Record a goal you're pursuing across turns: an objective to keep in " +
            "view until it's done. Active goals are shown back to you every turn, so " +
            "set one when the human gives you a task worth holding, and don't set " +
            "goals for trivial one-step asks.",
        parameters: {
            type: "object",
            properties: {
                content: { type: "string", description: "The goal, as a clear line of intent." },
            },
            required: ["content"],
        },
        async run(args) {
            const a = asRecord(args);
            try {
                const goal = store.create({ content: a.content as string, session });
                return { set: true, goal: toView(goal) };
            } catch (err) {
                if (err instanceof GoalError) return { set: false, error: err.message };
                throw err;
            }
        },
    };

    const update: ToolDef = {
        name: "goal_update",
        description:
            "Update a goal by id: mark it 'done' when achieved or 'abandoned' when " +
            "dropped (both stop it being shown each turn), set it back to 'active', " +
            "and/or revise its text. Provide `status`, `content`, or both.",
        parameters: {
            type: "object",
            properties: {
                id: { type: "number", description: "The id of the goal to update." },
                status: {
                    type: "string",
                    enum: ["active", "done", "abandoned"],
                    description: "New lifecycle state.",
                },
                content: { type: "string", description: "Revised goal text." },
            },
            required: ["id"],
        },
        async run(args) {
            const a = asRecord(args);
            if (typeof a.id !== "number" || !Number.isFinite(a.id)) {
                return { updated: false, error: "id must be a finite number" };
            }
            // Nothing to change is a no-op the model should hear about, not a
            // silent success.
            if (a.status === undefined && a.content === undefined) {
                return { updated: false, error: "provide a status and/or content to update" };
            }
            try {
                let goal: Goal | undefined;
                let found = false;
                if (a.content !== undefined) {
                    goal = store.edit(a.id, a.content as string);
                    found = found || goal !== undefined;
                }
                if (a.status !== undefined) {
                    if (!isGoalStatus(a.status)) {
                        return {
                            updated: false,
                            error: "status must be one of active, done, abandoned",
                        };
                    }
                    goal = store.setStatus(a.id, a.status);
                    found = found || goal !== undefined;
                }
                if (!goal) {
                    return {
                        updated: false,
                        error: found ? "update failed" : `no goal with id ${a.id}`,
                    };
                }
                return { updated: true, goal: toView(goal) };
            } catch (err) {
                if (err instanceof GoalError) return { updated: false, error: err.message };
                throw err;
            }
        },
    };

    const list: ToolDef = {
        name: "goal_list",
        description:
            "List your goals. Active goals are already shown to you each turn; use " +
            "this to review completed or abandoned ones, or to re-check the list. " +
            "Filter by `status` ('active' | 'done' | 'abandoned'); omit for all.",
        parameters: {
            type: "object",
            properties: {
                status: {
                    type: "string",
                    enum: ["active", "done", "abandoned"],
                    description: "Only goals in this state. Omit for every state.",
                },
                limit: { type: "number", description: "Max results (default 12)." },
            },
        },
        async run(args) {
            const a = asRecord(args);
            const status = isGoalStatus(a.status) ? a.status : undefined;
            const limit = typeof a.limit === "number" ? a.limit : DEFAULT_GOAL_LIMIT;
            const goals = store.list({ status, session, limit });
            return { count: goals.length, goals: goals.map(toView) };
        },
    };

    return [set, update, list];
}

/**
 * A passive provider that injects active goals into the system prompt every turn,
 * so the Construct keeps them in view. It folds in two distinct sets, kept under
 * separate headings so the model can tell shared standing intent from the work of
 * the conversation it's in:
 *
 *  - **Shared goals** — store-global goals (`session IS NULL`), visible to *every*
 *    conversation. These are the human-editable, cross-session intent the Goals
 *    page maintains: standing objectives every Construct turn should serve.
 *  - **This conversation's active goals** — goals scoped to this {@link sessionId},
 *    the ones the agent set (or a human set against this session) for the work at
 *    hand. Omitted entirely for a session-less store (a single-session REPL needs
 *    no such distinction; its goals are all "this conversation's").
 *
 * Async (the now-supported provider shape): it reads the store each turn. Each
 * read is a single indexed lookup (idx_goals_session_status), cheap enough for
 * the hot path; anything heavier wouldn't belong here. Returns `undefined` (adds
 * nothing) when there are no active goals in either set, so a goal-less
 * conversation pays no tokens for an empty list.
 */
export function goalContext(store: GoalStore, sessionId?: string): ContextProvider {
    return {
        name: "goals",
        async contribute() {
            // Shared goals belong to no session; read them with the global scope so
            // a bare session filter doesn't sweep in every session's goals.
            const shared = store.list({
                status: "active",
                scope: "global",
                limit: DEFAULT_GOAL_LIMIT,
            });
            // This conversation's own goals, only when the store is session-scoped.
            const session = sessionId
                ? store.list({
                      status: "active",
                      scope: "session",
                      session: sessionId,
                      limit: DEFAULT_GOAL_LIMIT,
                  })
                : [];
            if (shared.length === 0 && session.length === 0) return undefined;

            const sections: string[] = [];
            const render = (gs: typeof shared) =>
                gs.map((g) => `- (#${g.id}) ${g.content}`).join("\n");
            if (shared.length > 0) {
                sections.push(
                    `Shared goals (standing intent for every conversation; pursue these):\n${render(shared)}`,
                );
            }
            if (session.length > 0) {
                sections.push(
                    `This conversation's active goals (pursue these; mark done with ` +
                        `goal_update when achieved):\n${render(session)}`,
                );
            }
            return { system: sections.join("\n\n") };
        },
    };
}
