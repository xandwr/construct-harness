/**
 * GoalStore: a Construct's working sense of purpose.
 *
 * Where {@link MemoryStore} holds durable facts the agent chose to keep and the
 * {@link EventStore} holds everything that happened, this holds what the agent is
 * *trying to do*: a small set of goals, each a line of intent with a lifecycle
 * (active → done or abandoned). It is the thing a long-lived agent otherwise
 * lacks — between turns it has memory of the past but no held intent for the
 * present, so every turn starts goal-blind. A handful of active goals, injected
 * each turn (see goalContext in goalTools.ts), gives it that thread.
 *
 * Deliberately the simplest store in the harness: goals are a current working
 * set read in full each turn, not a corpus searched by relevance, so there is no
 * FTS or vector index, no embeddings, no ranking — just create, list by status,
 * update status, and edit text. It shares one database file (and one schema
 * `user_version`) with {@link MemoryStore}: both call the same {@link migrate},
 * so opening either brings the whole schema current.
 *
 * Provider-neutral and synchronous: it imports only `node:sqlite` and the shared
 * SQLite helpers, owns no model API, and never reaches the network.
 */

import { DatabaseSync } from "node:sqlite";
import { migrate, DEFAULT_BUSY_TIMEOUT, type StoreOptions } from "./memory.ts";
import { clampLimit, MAX_CONTENT_LENGTH } from "./sqlite.ts";

/** {@link GoalStore} options: the shared store options plus a goal-only event
 *  sink. Kept separate from {@link StoreOptions} (which MemoryStore and
 *  EventStore also use) so only the goal store carries the `onChange` channel. */
export interface GoalStoreOptions extends StoreOptions {
    /** Invoked after each successful write so the change can be logged (added,
     *  deleted, status, edited). Best-effort: the store catches a throwing sink
     *  so a logging failure never undoes the goal write. */
    onChange?: GoalEventSink;
}

/**
 * Thrown when a goal fails validation before it reaches the database. Callers can
 * `instanceof`-check this to tell "you gave me bad data" from a real storage
 * failure. Mirrors {@link MemoryError} / {@link EventError}.
 */
export class GoalError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "GoalError";
    }
}

/** A goal's lifecycle state. `active` is being pursued; `done` was achieved;
 *  `abandoned` was dropped without achieving it (kept, not deleted, so the record
 *  of intent survives). Matches the CHECK constraint in the schema. */
export type GoalStatus = "active" | "done" | "abandoned";

/** What happened to a goal, for the {@link GoalEventSink}. `created` and `deleted`
 *  are the lifecycle endpoints a human or the agent drives directly; `status` and
 *  `edited` are in-place changes. The sink decides which it cares to record. */
export type GoalChange = "created" | "deleted" | "status" | "edited";

/**
 * The event-log `kind` a goal change is recorded under, the goal counterpart to
 * `DREAM_EVENT_KIND`. One kind covers the whole lifecycle (created, deleted,
 * status, edited); the change is discriminated by `meta.change`, so a reader can
 * filter goal events with `recent({ kind: GOAL_EVENT_KIND })` and tell what
 * happened from the payload. Defined here (not at the wiring site) so any reader
 * of the log — the server's {@link GoalEventSink}, the resume catch-up provider —
 * agrees on the constant by construction rather than a scattered string literal.
 */
export const GOAL_EVENT_KIND = "goal";

/**
 * The shape a goal change writes into a {@link GOAL_EVENT_KIND} event's `meta`.
 * Read defensively (the EventStore degrades a corrupt meta to `undefined`), but
 * named here so both the writer (the sink) and readers (the resume provider) work
 * against one declared structure rather than re-guessing the fields.
 */
export interface GoalEventMeta {
    change: GoalChange;
    goalId: number;
    status: GoalStatus;
}

/**
 * A callback the store invokes after a successful write, so a change to the
 * agent's intent can be recorded in the event log the way a message or a tool
 * call is — "goals also emit events when added/deleted." It is the store's only
 * outward channel: `goals.ts` stays provider-neutral (it imports no EventStore),
 * and the wiring side decides what a goal event looks like (see the server's
 * {@link GoalStore} construction). The store calls it best-effort — a sink that
 * throws must not fail the goal write that already committed — so the callback,
 * or the store's call of it, swallows errors.
 *
 * `goal` is the post-write row for a create/status/edit, or the row as it last
 * stood for a delete (so the sink can name what was removed). `now` is the same
 * timestamp stamped on the write, so the event and the goal agree on when.
 */
export type GoalEventSink = (change: GoalChange, goal: Goal, now: number) => void;

const STATUSES: readonly GoalStatus[] = ["active", "done", "abandoned"];

/** True for a value the store will accept as a {@link GoalStatus}. */
export function isGoalStatus(value: unknown): value is GoalStatus {
    return typeof value === "string" && (STATUSES as readonly string[]).includes(value);
}

/** A persisted goal. Construct via {@link GoalStore.create}; the store assigns
 *  the real `id` and timestamps. */
export class Goal {
    id: number;
    content: string;
    status: GoalStatus;
    /** Optional id scoping the goal to one conversation (a Session's id). A goal
     *  with no session is global to the store. */
    session?: string;
    created: number;
    /** Last time the row was written (created, status change, or edit). */
    updated: number;

    constructor(fields: {
        id?: number;
        content: string;
        status?: GoalStatus;
        session?: string;
        created: number;
        updated?: number;
    }) {
        this.id = fields.id ?? 0;
        this.content = fields.content;
        this.status = fields.status ?? "active";
        this.session = fields.session;
        this.created = fields.created;
        this.updated = fields.updated ?? fields.created;
    }
}

/** Filters for a goal read. */
export interface GoalQuery {
    /** Only goals in this state. Omit for every state. */
    status?: GoalStatus;
    /** Only goals scoped to this session. Omit to read across all sessions.
     *  Ignored when {@link scope} is `"global"` (a global read has no session). */
    session?: string;
    /**
     * Restrict by ownership rather than by a specific session id:
     *  - `"global"` — only store-global goals (`session IS NULL`), visible to
     *    every conversation. This is the distinction a bare
     *    `list({ session: undefined })` *cannot* draw: undefined means "no session
     *    filter" (every goal), whereas `scope: "global"` means "the goals belonging
     *    to no session". Set it to read the shared goals on their own.
     *  - `"session"` — only goals scoped to {@link session} (requires `session`).
     *    Equivalent to passing `session` with no scope; named for symmetry.
     *  - omitted — the legacy behavior: filter by `session` when given, else every
     *    goal across all sessions.
     */
    scope?: "global" | "session";
    /** Max rows. Defaults to the shared {@link DEFAULT_LIMIT}; capped at MAX_LIMIT. */
    limit?: number;
}

interface GoalRow {
    id: number;
    content: string;
    status: GoalStatus;
    session: string | null;
    created: number;
    updated: number;
}

function rowToGoal(row: GoalRow): Goal {
    return new Goal({
        id: row.id,
        content: row.content,
        status: row.status,
        session: row.session ?? undefined,
        created: row.created,
        updated: row.updated,
    });
}

/** Validate and trim goal text, throwing {@link GoalError} on anything we refuse
 *  to store. Same defensive posture as memory's normalizeInput. */
function normalizeContent(content: unknown): string {
    if (typeof content !== "string") {
        throw new GoalError("content must be a string");
    }
    const trimmed = content.trim();
    if (trimmed.length === 0) {
        throw new GoalError("content must not be empty");
    }
    if (trimmed.length > MAX_CONTENT_LENGTH) {
        throw new GoalError(`content exceeds ${MAX_CONTENT_LENGTH} characters`);
    }
    return trimmed;
}

/**
 * SQLite-backed store of the agent's goals.
 *
 * Shares one database file (and one schema `user_version`) with
 * {@link MemoryStore}: both call the same {@link migrate}. Construct one, create
 * goals on it, list and update them, and {@link close} it. File-backed stores run
 * in WAL mode by default so many readers and one writer work concurrently.
 *
 * Injectable `now` per write keeps timestamps deterministic in tests; it defaults
 * to {@link Date.now}.
 */
export class GoalStore {
    private readonly db: DatabaseSync;
    private readonly walEnabled: boolean;
    private readonly schemaVersion: number;
    private readonly insertStmt;
    private readonly getStmt;
    private readonly updateStatusStmt;
    private readonly updateContentStmt;
    private readonly deleteStmt;
    /** Optional channel for recording writes as events; see {@link GoalEventSink}. */
    private readonly onChange?: GoalEventSink;
    private closed = false;

    constructor(options: string | GoalStoreOptions = "db.sqlite") {
        const opts: GoalStoreOptions =
            typeof options === "string" ? { location: options } : options;
        const location = opts.location ?? "db.sqlite";
        this.onChange = opts.onChange;
        const busyTimeout = opts.busyTimeout ?? DEFAULT_BUSY_TIMEOUT;
        const wantWal = (opts.wal ?? true) && location !== ":memory:";

        this.db = new DatabaseSync(location);
        this.db.exec(`PRAGMA busy_timeout = ${Math.max(0, Math.floor(busyTimeout))}`);

        if (wantWal) {
            const row = this.db.prepare(`PRAGMA journal_mode = WAL`).get() as {
                journal_mode?: string;
            };
            this.walEnabled = row?.journal_mode?.toLowerCase() === "wal";
            if (this.walEnabled) this.db.exec(`PRAGMA synchronous = NORMAL`);
        } else {
            this.walEnabled = false;
        }

        // The same one migration runner MemoryStore uses: one user_version per
        // file, shared by every store in it.
        this.schemaVersion = migrate(this.db);

        this.insertStmt = this.db.prepare(
            `INSERT INTO goals (content, status, session, created, updated)
             VALUES (?, ?, ?, ?, ?)`,
        );
        this.getStmt = this.db.prepare(`SELECT * FROM goals WHERE id = ?`);
        this.updateStatusStmt = this.db.prepare(
            `UPDATE goals SET status = ?, updated = ? WHERE id = ?`,
        );
        this.updateContentStmt = this.db.prepare(
            `UPDATE goals SET content = ?, updated = ? WHERE id = ?`,
        );
        this.deleteStmt = this.db.prepare(`DELETE FROM goals WHERE id = ?`);
    }

    private assertOpen() {
        if (this.closed) throw new GoalError("store is closed");
    }

    /** Notify the {@link GoalEventSink} of a committed write, best-effort. A sink
     *  that throws is swallowed: the goal write already happened, and a failure to
     *  *log* it must not turn into a failure to *do* it. */
    private emit(change: GoalChange, goal: Goal, now: number) {
        if (!this.onChange) return;
        try {
            this.onChange(change, goal, now);
        } catch {
            // Best-effort logging: never let a sink failure escape a goal write.
        }
    }

    /** Create a goal (status defaults to 'active'), returning it with its id. */
    create(input: { content: string; session?: string; now?: number }): Goal {
        this.assertOpen();
        const content = normalizeContent(input.content);
        const now = input.now ?? Date.now();
        const session = input.session ?? null;
        const result = this.insertStmt.run(content, "active", session, now, now);
        const goal = new Goal({
            id: Number(result.lastInsertRowid),
            content,
            status: "active",
            session: session ?? undefined,
            created: now,
            updated: now,
        });
        this.emit("created", goal, now);
        return goal;
    }

    /** Fetch one goal by id, or undefined if none exists. */
    get(id: number): Goal | undefined {
        this.assertOpen();
        const row = this.getStmt.get(id) as GoalRow | undefined;
        return row ? rowToGoal(row) : undefined;
    }

    /**
     * List goals, newest intent last so the model reads them in the order they
     * were set (oldest first, the natural order of a to-do list). Filters by
     * status, session, and/or {@link GoalQuery.scope}. The hot call is
     * `list({ status: 'active', session })` (goalContext, every turn), which the
     * idx_goals_session_status index serves without a scan; `scope: 'global'`
     * reads the shared goals (`session IS NULL`) the same index also covers.
     */
    list(opts: GoalQuery = {}): Goal[] {
        this.assertOpen();
        const { where, params } = this.buildFilter(opts);
        const sql = `SELECT * FROM goals ${where} ORDER BY created ASC, id ASC LIMIT ?`;
        params.push(clampLimit(opts.limit));
        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as GoalRow[];
        return rows.map(rowToGoal);
    }

    /** Build the shared WHERE clause for {@link list} and {@link count}: status,
     *  and a session predicate that honors {@link GoalQuery.scope} —
     *  `session IS NULL` for global, `session = ?` for a specific session. */
    private buildFilter(opts: Omit<GoalQuery, "limit">): { where: string; params: unknown[] } {
        const conditions: string[] = [];
        const params: unknown[] = [];
        if (opts.status !== undefined) {
            conditions.push("status = ?");
            params.push(opts.status);
        }
        if (opts.scope === "global") {
            // The distinction a bare session filter can't draw: goals owned by no
            // session, shared across every conversation.
            conditions.push("session IS NULL");
        } else if (opts.session !== undefined) {
            conditions.push("session = ?");
            params.push(opts.session);
        }
        const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
        return { where, params };
    }

    /** Move a goal to a new status (e.g. mark it done). Returns the updated goal,
     *  or undefined if no goal has that id. */
    setStatus(id: number, status: GoalStatus, now = Date.now()): Goal | undefined {
        this.assertOpen();
        if (!isGoalStatus(status)) {
            throw new GoalError(`status must be one of ${STATUSES.join(", ")}`);
        }
        const changed = this.updateStatusStmt.run(status, now, id).changes > 0;
        if (!changed) return undefined;
        const goal = this.get(id);
        if (goal) this.emit("status", goal, now);
        return goal;
    }

    /** Edit a goal's text (e.g. to sharpen it). Returns the updated goal, or
     *  undefined if no goal has that id. */
    edit(id: number, content: string, now = Date.now()): Goal | undefined {
        this.assertOpen();
        const text = normalizeContent(content);
        const changed = this.updateContentStmt.run(text, now, id).changes > 0;
        if (!changed) return undefined;
        const goal = this.get(id);
        if (goal) this.emit("edited", goal, now);
        return goal;
    }

    /** Permanently remove a goal. Prefer {@link setStatus} to 'abandoned' when the
     *  record of intent is worth keeping; this is for genuine mistakes. Returns
     *  whether a row was removed. The row is read before the delete so the
     *  {@link GoalEventSink} can name what was removed (the row is gone after). */
    delete(id: number, now = Date.now()): boolean {
        this.assertOpen();
        // Read first so a sink that records the deletion has the goal it removed;
        // skip the read entirely when nothing is listening.
        const goal = this.onChange ? this.get(id) : undefined;
        const removed = this.deleteStmt.run(id).changes > 0;
        if (removed && goal) this.emit("deleted", goal, now);
        return removed;
    }

    /** Count goals matching the filters (no limit applied). Honors
     *  {@link GoalQuery.scope} the same way {@link list} does. */
    count(opts: Omit<GoalQuery, "limit"> = {}): number {
        this.assertOpen();
        const { where, params } = this.buildFilter(opts);
        const row = this.db
            .prepare(`SELECT COUNT(*) AS n FROM goals ${where}`)
            .get(...(params as never[])) as { n: number };
        return row.n;
    }

    /** Whether this store is actually running in WAL mode. */
    get wal(): boolean {
        return this.walEnabled;
    }

    /** The schema version the underlying database was migrated to on open. */
    get version(): number {
        return this.schemaVersion;
    }

    /** Release the underlying database handle. Idempotent. Checkpoints first so a
     *  file-backed store doesn't leave a populated `-wal` sidecar behind. */
    close() {
        if (this.closed) return;
        if (this.walEnabled) {
            try {
                this.db.exec(`PRAGMA wal_checkpoint(TRUNCATE)`);
            } catch {
                // Best-effort: a checkpoint failure must not prevent close.
            }
        }
        this.closed = true;
        this.db.close();
    }
}
