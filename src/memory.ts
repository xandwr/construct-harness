import { DatabaseSync } from "node:sqlite";

/**
 * Thrown when a memory fails validation before it ever reaches the database.
 * Callers can `instanceof`-check this to distinguish "you gave me bad data"
 * from a genuine sqlite/storage failure.
 */
export class MemoryError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "MemoryError";
    }
}

/** Importance is a normalized [0, 1] score; out-of-range values are rejected. */
export const MIN_IMPORTANCE = 0;
export const MAX_IMPORTANCE = 1;

/** Hard ceiling on stored content so a runaway write can't bloat the db. */
export const MAX_CONTENT_LENGTH = 100_000;
/** Hard ceiling on a single tag, and on the number of tags per memory. */
export const MAX_TAG_LENGTH = 256;
export const MAX_TAGS = 64;

/** Fields a caller may supply when creating a memory. */
export interface MemoryInput {
    content: string;
    tags?: string[];
    importance?: number;
    /** Defaults to "now"; injectable so tests are deterministic. */
    created?: number;
}

/** Options for {@link MemoryStore.all} / {@link MemoryStore.search}. */
export interface QueryOptions {
    /** Max rows to return. Defaults to {@link DEFAULT_LIMIT}; capped at {@link MAX_LIMIT}. */
    limit?: number;
    /** Number of rows to skip (for pagination). */
    offset?: number;
    /** Only return memories carrying ALL of these tags. */
    tags?: string[];
}

export const DEFAULT_LIMIT = 100;
export const MAX_LIMIT = 1000;

/** How long (ms) a writer waits on a locked db before giving up. */
export const DEFAULT_BUSY_TIMEOUT = 5_000;

/** Tuning knobs for {@link MemoryStore}'s underlying database. */
export interface StoreOptions {
    /** Path to the sqlite file, or `:memory:` for an ephemeral store. */
    location?: string;
    /** Writer lock wait in ms before SQLITE_BUSY. Defaults to {@link DEFAULT_BUSY_TIMEOUT}. */
    busyTimeout?: number;
    /**
     * Enable Write-Ahead Logging for concurrent readers + one writer.
     * Defaults to true for file-backed stores; always off (and meaningless)
     * for `:memory:`, which keeps a private per-connection database.
     */
    wal?: boolean;
}

/** Raised when the on-disk schema can't be reconciled with this code. */
export class MigrationError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "MigrationError";
    }
}

/**
 * Ordered schema migrations. Each entry's `up` brings the database from version
 * `i` to version `i + 1` (1-indexed: MIGRATIONS[0] produces schema version 1).
 *
 * Rules that keep this honest:
 *  - APPEND ONLY. Never edit or reorder a published migration — add a new one.
 *    Changing history would make already-migrated databases silently wrong.
 *  - Each `up` runs inside a transaction managed by {@link migrate}; it should
 *    not BEGIN/COMMIT itself.
 *  - Migration 1 is written defensively (IF NOT EXISTS) so it adopts pre-
 *    versioning databases — those created before user_version was tracked
 *    already have the table and simply advance to version 1 as a no-op.
 */
const MIGRATIONS: ReadonlyArray<{ name: string; up: (db: DatabaseSync) => void }> = [
    {
        name: "create memory table and indices",
        up(db) {
            db.exec(`
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created INTEGER NOT NULL,
                    updated INTEGER NOT NULL,
                    tags TEXT,
                    importance REAL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory (importance DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_created ON memory (created DESC);
            `);
        },
    },
    {
        // Full-text index over content, so recall can rank by lexical relevance
        // (bm25) instead of only importance/recency. The FTS table is an
        // external-content index ("content='memory'"): it stores no copy of the
        // text itself, just the inverted index, and reads the real text back
        // from `memory` via rowid. Triggers keep it in lockstep with writes.
        name: "add fts5 full-text index over content",
        up(db) {
            db.exec(`
                -- The porter stemmer folds morphological variants together
                -- ("allergies" matches "allergic", "deploys" matches "deploy"),
                -- which is what makes turn-relevant recall robust to the exact
                -- wording a user happens to use.
                CREATE VIRTUAL TABLE memory_fts USING fts5(
                    content,
                    content='memory',
                    content_rowid='id',
                    tokenize='porter'
                );

                -- Keep the index in sync. For UPDATE/DELETE we first push a
                -- 'delete' row (the special INSERT below) to retract the old
                -- terms, then index the new content.
                CREATE TRIGGER memory_ai AFTER INSERT ON memory BEGIN
                    INSERT INTO memory_fts (rowid, content) VALUES (new.id, new.content);
                END;
                CREATE TRIGGER memory_ad AFTER DELETE ON memory BEGIN
                    INSERT INTO memory_fts (memory_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                END;
                CREATE TRIGGER memory_au AFTER UPDATE ON memory BEGIN
                    INSERT INTO memory_fts (memory_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                    INSERT INTO memory_fts (rowid, content) VALUES (new.id, new.content);
                END;

                -- Backfill any rows that predate the index.
                INSERT INTO memory_fts (rowid, content)
                    SELECT id, content FROM memory;
            `);
        },
    },
];

/** The schema version this build of the code expects. */
export const SCHEMA_VERSION = MIGRATIONS.length;

/**
 * Bring `db` up to {@link SCHEMA_VERSION}, running each pending migration in its
 * own transaction and bumping `PRAGMA user_version` as it goes. Returns the
 * resulting version.
 *
 * Refuses to touch a database whose version is *newer* than this code knows
 * about — that means an older binary opened a future schema, and proceeding
 * could corrupt it. Fail loudly instead.
 */
function migrate(db: DatabaseSync): number {
    const row = db.prepare(`PRAGMA user_version`).get() as { user_version: number };
    let version = row.user_version;

    if (version > MIGRATIONS.length) {
        throw new MigrationError(
            `database schema version ${version} is newer than this code supports ` +
                `(max ${MIGRATIONS.length}); upgrade the application`,
        );
    }

    for (let target = version + 1; target <= MIGRATIONS.length; target++) {
        const migration = MIGRATIONS[target - 1];
        db.exec("BEGIN");
        try {
            migration.up(db);
            // user_version doesn't accept bound params; target is a trusted int.
            db.exec(`PRAGMA user_version = ${target}`);
            db.exec("COMMIT");
        } catch (err) {
            db.exec("ROLLBACK");
            throw new MigrationError(
                `migration ${target} (${migration.name}) failed: ${(err as Error).message}`,
            );
        }
        version = target;
    }

    return version;
}

export class Memory {
    id: number;
    content: string;
    created: number;
    /** Last time the row was written (created or updated). */
    updated: number;
    tags: string[];
    importance?: number;

    constructor(input: MemoryInput) {
        const { content, tags, importance, created } = normalizeInput(input);
        this.id = 0;
        this.content = content;
        this.created = created;
        this.updated = created;
        this.tags = tags;
        this.importance = importance;
    }
}

/**
 * Validate and normalize raw input into the canonical shape we persist.
 * Throws {@link MemoryError} on anything we refuse to store.
 */
function normalizeInput(input: MemoryInput): {
    content: string;
    tags: string[];
    importance?: number;
    created: number;
} {
    if (input === null || typeof input !== "object") {
        throw new MemoryError("memory input must be an object");
    }

    const content = input.content;
    if (typeof content !== "string") {
        throw new MemoryError("content must be a string");
    }
    const trimmed = content.trim();
    if (trimmed.length === 0) {
        throw new MemoryError("content must not be empty");
    }
    if (content.length > MAX_CONTENT_LENGTH) {
        throw new MemoryError(`content exceeds ${MAX_CONTENT_LENGTH} characters`);
    }

    const tags = normalizeTags(input.tags);

    let importance: number | undefined;
    if (input.importance !== undefined && input.importance !== null) {
        const n = input.importance;
        if (typeof n !== "number" || !Number.isFinite(n)) {
            throw new MemoryError("importance must be a finite number");
        }
        if (n < MIN_IMPORTANCE || n > MAX_IMPORTANCE) {
            throw new MemoryError(
                `importance must be within [${MIN_IMPORTANCE}, ${MAX_IMPORTANCE}]`,
            );
        }
        importance = n;
    }

    let created = input.created;
    if (created === undefined) {
        created = Date.now();
    } else if (typeof created !== "number" || !Number.isFinite(created)) {
        throw new MemoryError("created must be a finite number");
    }

    return { content, tags, importance, created };
}

/** Dedupe, trim, drop empties, and bound the tag list. */
function normalizeTags(raw: string[] | undefined): string[] {
    if (raw === undefined || raw === null) return [];
    if (!Array.isArray(raw)) {
        throw new MemoryError("tags must be an array of strings");
    }
    const seen = new Set<string>();
    for (const t of raw) {
        if (typeof t !== "string") {
            throw new MemoryError("each tag must be a string");
        }
        const tag = t.trim();
        if (tag.length === 0) continue;
        if (tag.length > MAX_TAG_LENGTH) {
            throw new MemoryError(`tag exceeds ${MAX_TAG_LENGTH} characters`);
        }
        seen.add(tag);
    }
    if (seen.size > MAX_TAGS) {
        throw new MemoryError(`too many tags (max ${MAX_TAGS})`);
    }
    return [...seen];
}

interface MemoryRow {
    id: number;
    content: string;
    created: number;
    updated: number;
    tags: string | null;
    importance: number | null;
}

/**
 * Reconstruct a {@link Memory} from a db row. Tolerant of legacy/corrupt
 * `tags` payloads: a row whose tags don't parse as a string array degrades to
 * an empty tag list rather than throwing and taking down a whole query.
 */
function rowToMemory(row: MemoryRow): Memory {
    const m = new Memory({
        content: row.content,
        tags: parseTags(row.tags),
        importance: row.importance ?? undefined,
        created: row.created,
    });
    m.id = row.id;
    m.updated = row.updated;
    return m;
}

function parseTags(raw: string | null): string[] {
    if (!raw) return [];
    try {
        const parsed: unknown = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed.filter((t): t is string => typeof t === "string");
    } catch {
        return [];
    }
}

/**
 * SQLite-backed store for {@link Memory} objects.
 *
 * The database is injectable: pass a path (default `db.sqlite`) or `:memory:`
 * for an isolated, ephemeral store — which is what the tests use so they never
 * touch disk or share state. Construct one, use it, and {@link close} it.
 *
 * File-backed stores run in WAL mode by default, so many readers and a single
 * writer can work concurrently without blocking each other, and a configurable
 * busy-timeout keeps a momentarily-locked writer from failing outright.
 */
export class MemoryStore {
    private readonly db: DatabaseSync;
    private readonly walEnabled: boolean;
    private readonly schemaVersion: number;
    private readonly insertStmt;
    private readonly getStmt;
    private readonly updateStmt;
    private readonly deleteStmt;
    private readonly countStmt;
    private readonly clearStmt;
    private closed = false;

    constructor(options: string | StoreOptions = "db.sqlite") {
        const opts: StoreOptions = typeof options === "string" ? { location: options } : options;
        const location = opts.location ?? "db.sqlite";
        const busyTimeout = opts.busyTimeout ?? DEFAULT_BUSY_TIMEOUT;
        // WAL is pointless for an in-memory db (private per connection), so it
        // only ever applies to file-backed stores.
        const wantWal = (opts.wal ?? true) && location !== ":memory:";

        this.db = new DatabaseSync(location);

        // Bound how long a writer waits on a lock before SQLITE_BUSY.
        this.db.exec(`PRAGMA busy_timeout = ${Math.max(0, Math.floor(busyTimeout))}`);

        if (wantWal) {
            // journal_mode returns the resulting mode; confirm WAL actually took
            // (it can silently fall back, e.g. on some network filesystems).
            const row = this.db.prepare(`PRAGMA journal_mode = WAL`).get() as {
                journal_mode?: string;
            };
            this.walEnabled = row?.journal_mode?.toLowerCase() === "wal";
            if (this.walEnabled) {
                // NORMAL is the standard, durable-enough pairing with WAL.
                this.db.exec(`PRAGMA synchronous = NORMAL`);
            }
        } else {
            this.walEnabled = false;
        }

        this.schemaVersion = migrate(this.db);

        this.insertStmt = this.db.prepare(
            `INSERT INTO memory (content, created, updated, tags, importance)
             VALUES (?, ?, ?, ?, ?)`,
        );
        this.getStmt = this.db.prepare(`SELECT * FROM memory WHERE id = ?`);
        this.updateStmt = this.db.prepare(
            `UPDATE memory SET content = ?, updated = ?, tags = ?, importance = ? WHERE id = ?`,
        );
        this.deleteStmt = this.db.prepare(`DELETE FROM memory WHERE id = ?`);
        this.countStmt = this.db.prepare(`SELECT COUNT(*) AS n FROM memory`);
        this.clearStmt = this.db.prepare(`DELETE FROM memory`);
    }

    private assertOpen() {
        if (this.closed) throw new MemoryError("store is closed");
    }

    /** Persist a memory, assigning its real id from the database. */
    save(memory: Memory): Memory {
        this.assertOpen();
        // Re-validate: the Memory may have been mutated after construction.
        const norm = normalizeInput(memory);
        const result = this.insertStmt.run(
            norm.content,
            norm.created,
            memory.updated || norm.created,
            serializeTags(norm.tags),
            norm.importance ?? null,
        );
        memory.id = Number(result.lastInsertRowid);
        memory.content = norm.content;
        memory.tags = norm.tags;
        memory.importance = norm.importance;
        memory.created = norm.created;
        return memory;
    }

    /**
     * Apply in-place edits to an existing memory and persist them.
     * `created` is immutable; `updated` is stamped to `now`. Returns the
     * refreshed memory, or undefined if no row with that id exists.
     */
    update(
        id: number,
        patch: Partial<Pick<MemoryInput, "content" | "tags" | "importance">>,
        now = Date.now(),
    ): Memory | undefined {
        this.assertOpen();
        const existing = this.get(id);
        if (!existing) return undefined;

        const merged = normalizeInput({
            content: patch.content ?? existing.content,
            tags: patch.tags ?? existing.tags,
            importance: "importance" in patch ? patch.importance : existing.importance,
            created: existing.created,
        });

        this.updateStmt.run(
            merged.content,
            now,
            serializeTags(merged.tags),
            merged.importance ?? null,
            id,
        );

        existing.content = merged.content;
        existing.tags = merged.tags;
        existing.importance = merged.importance;
        existing.updated = now;
        return existing;
    }

    get(id: number): Memory | undefined {
        this.assertOpen();
        const row = this.getStmt.get(id) as MemoryRow | undefined;
        return row ? rowToMemory(row) : undefined;
    }

    /**
     * Return memories ordered by importance then recency. Supports limit,
     * offset, and AND-tag filtering. Defaults to {@link DEFAULT_LIMIT} rows so a
     * growing table never returns an unbounded result set.
     */
    all(opts: QueryOptions = {}): Memory[] {
        return this.query(null, opts);
    }

    /**
     * Case-insensitive substring search over content, otherwise identical to
     * {@link all}. An empty/whitespace query behaves like {@link all}.
     */
    search(text: string, opts: QueryOptions = {}): Memory[] {
        const needle = typeof text === "string" ? text.trim() : "";
        return this.query(needle.length ? needle : null, opts);
    }

    /**
     * Rank memories by lexical relevance to `text` using the FTS5 index (bm25),
     * with importance as a gentle tiebreak. This is what auto-recall wants: rows
     * most relevant to *this turn*, not the globally most-important rows.
     *
     * Unlike {@link search} (substring `LIKE`), matching here is token-based —
     * the query is reduced to its word tokens and OR-matched, so a whole
     * sentence still finds memories that share any meaningful term. A query with
     * no usable tokens (or no FTS hits) yields an empty array; callers that want
     * a fallback should handle that themselves.
     */
    searchRelevant(text: string, opts: QueryOptions = {}): Memory[] {
        this.assertOpen();
        const match = toFtsQuery(text);
        if (match === null) return [];

        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));
        const filterTags = normalizeTags(opts.tags);

        const where: string[] = ["memory_fts MATCH ?"];
        const params: unknown[] = [match];
        for (const tag of filterTags) {
            where.push(`m.tags LIKE ? ESCAPE '\\'`);
            params.push(`%"${escapeLike(tag)}"%`);
        }

        // bm25() is ascending (more-negative = better), so order by it directly,
        // then let importance break ties among comparably-relevant rows.
        const sql =
            `SELECT m.* FROM memory_fts ` +
            `JOIN memory m ON m.id = memory_fts.rowid ` +
            `WHERE ${where.join(" AND ")} ` +
            `ORDER BY bm25(memory_fts), m.importance IS NULL, m.importance DESC ` +
            `LIMIT ? OFFSET ?`;
        params.push(limit, offset);

        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as MemoryRow[];
        return rows.map(rowToMemory);
    }

    private query(needle: string | null, opts: QueryOptions): Memory[] {
        this.assertOpen();
        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));
        const filterTags = normalizeTags(opts.tags);

        const where: string[] = [];
        const params: unknown[] = [];

        if (needle !== null) {
            where.push(`content LIKE ? ESCAPE '\\'`);
            params.push(`%${escapeLike(needle)}%`);
        }
        // AND-match each tag against the JSON-array text. We match the quoted,
        // exact tag token (`"tag"`) to avoid substring collisions between tags.
        for (const tag of filterTags) {
            where.push(`tags LIKE ? ESCAPE '\\'`);
            params.push(`%"${escapeLike(tag)}"%`);
        }

        const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";
        const sql =
            `SELECT * FROM memory ${whereSql} ` +
            `ORDER BY importance IS NULL, importance DESC, created DESC ` +
            `LIMIT ? OFFSET ?`;
        params.push(limit, offset);

        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as MemoryRow[];
        return rows.map(rowToMemory);
    }

    /** Total number of stored memories. */
    count(): number {
        this.assertOpen();
        const row = this.countStmt.get() as { n: number };
        return row.n;
    }

    delete(id: number): boolean {
        this.assertOpen();
        return this.deleteStmt.run(id).changes > 0;
    }

    /** Remove every memory; returns how many were deleted. */
    clear(): number {
        this.assertOpen();
        return Number(this.clearStmt.run().changes);
    }

    /** Whether this store is actually running in WAL mode. */
    get wal(): boolean {
        return this.walEnabled;
    }

    /** The schema version the underlying database was migrated to on open. */
    get version(): number {
        return this.schemaVersion;
    }

    /**
     * Fold the WAL back into the main database file and truncate it. No-op when
     * WAL isn't active. Useful before backing up the file or to cap WAL growth
     * under a long-running writer.
     */
    checkpoint(): void {
        this.assertOpen();
        if (!this.walEnabled) return;
        this.db.exec(`PRAGMA wal_checkpoint(TRUNCATE)`);
    }

    /**
     * Release the underlying database handle. Idempotent. Checkpoints first so a
     * file-backed store doesn't leave a populated `-wal` sidecar behind.
     */
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

function serializeTags(tags: string[]): string | null {
    return tags.length ? JSON.stringify(tags) : null;
}

function clampLimit(limit: number | undefined): number {
    if (limit === undefined) return DEFAULT_LIMIT;
    if (!Number.isFinite(limit) || limit <= 0) return DEFAULT_LIMIT;
    return Math.min(Math.floor(limit), MAX_LIMIT);
}

/** Escape LIKE wildcards so user text is matched literally (with ESCAPE '\'). */
function escapeLike(s: string): string {
    return s.replace(/[\\%_]/g, (c) => `\\${c}`);
}

/**
 * Turn free text into a safe FTS5 MATCH query.
 *
 * Raw user text can't go straight into MATCH: characters like `"`, `*`, `:`,
 * `(`, `-`, and the bareword `AND`/`OR`/`NOT` are FTS operators and would
 * either throw a syntax error or change the query's meaning. So we extract
 * alphanumeric word tokens, wrap each in double quotes (which makes it a
 * literal string token, neutralizing operators), and OR them together — any
 * shared term makes a memory a candidate, and bm25 sorts by how well it matches.
 *
 * Returns null when the text has no usable tokens, so the caller can treat that
 * as "no query" rather than issuing an empty MATCH (which is itself an error).
 */
function toFtsQuery(text: string): string | null {
    if (typeof text !== "string") return null;
    const tokens = text.toLowerCase().match(/[\p{L}\p{N}]+/gu);
    if (!tokens || tokens.length === 0) return null;
    // Dedupe to keep the query compact; cap to bound pathological inputs.
    const unique = [...new Set(tokens)].slice(0, MAX_FTS_TOKENS);
    return unique.map((t) => `"${t}"`).join(" OR ");
}

/** Cap on tokens fed into a single FTS MATCH, to bound a giant prompt. */
const MAX_FTS_TOKENS = 32;
