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
 */
export class MemoryStore {
    private readonly db: DatabaseSync;
    private readonly insertStmt;
    private readonly getStmt;
    private readonly updateStmt;
    private readonly deleteStmt;
    private readonly countStmt;
    private readonly clearStmt;
    private closed = false;

    constructor(location = "db.sqlite") {
        this.db = new DatabaseSync(location);
        this.db.exec(`
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

    /** Release the underlying database handle. Idempotent. */
    close() {
        if (this.closed) return;
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
