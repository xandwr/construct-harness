import { DatabaseSync } from "node:sqlite";
import { blobToVector, cosineSimilarity, vectorToBlob } from "./embeddings.ts";
import {
    clampLimit,
    escapeLike,
    toFtsQuery,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_CONTENT_LENGTH,
} from "./sqlite.ts";

// Re-export the shared SQLite helpers/constants that callers (and the existing
// tests) import from this module, so the extraction into ./sqlite.ts stays
// source-compatible. These resolve to the same bindings ./sqlite.ts exports, so
// star-exporting both modules from mod.ts is unambiguous.
export { clampLimit, escapeLike, toFtsQuery, DEFAULT_LIMIT, MAX_LIMIT, MAX_CONTENT_LENGTH };

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

/** A memory paired with its cosine similarity to a query vector, from
 *  {@link MemoryStore.semanticSearch}. Score is in roughly [-1, 1], higher =
 *  more similar. */
export interface SemanticHit {
    memory: Memory;
    score: number;
}

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
 *  - APPEND ONLY. Never edit or reorder a published migration: add a new one.
 *    Changing history would make already-migrated databases silently wrong.
 *  - Each `up` runs inside a transaction managed by {@link migrate}; it should
 *    not BEGIN/COMMIT itself.
 *  - Migration 1 is written defensively (IF NOT EXISTS) so it adopts pre-
 *    versioning databases: those created before user_version was tracked
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
    {
        // Vector index for semantic (meaning-based) recall. Each row holds one
        // memory's embedding as a little-endian float32 BLOB plus its dimension,
        // keyed by the memory's id. Unlike FTS, embeddings can't be computed in
        // SQL: they require a network call to an embedding model: so rows are
        // written explicitly by the application (see setEmbedding / backfill),
        // NOT by an INSERT/UPDATE trigger.
        //
        // We still trigger on DELETE and on content UPDATE: a deleted memory's
        // vector must go with it, and an edited memory's vector is now stale, so
        // we drop it and let the application re-embed. This keeps the table from
        // ever serving a vector that doesn't match the current content.
        name: "add vector index for semantic search",
        up(db) {
            db.exec(`
                CREATE TABLE IF NOT EXISTS memory_vec (
                    rowid INTEGER PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE,
                    dim   INTEGER NOT NULL,
                    vec   BLOB NOT NULL
                );

                CREATE TRIGGER memory_vec_ad AFTER DELETE ON memory BEGIN
                    DELETE FROM memory_vec WHERE rowid = old.id;
                END;

                -- Only invalidate the vector when the *content* actually changed;
                -- a metadata-only edit (tags/importance) leaves the embedding
                -- valid, so we avoid a needless re-embed.
                CREATE TRIGGER memory_vec_au AFTER UPDATE OF content ON memory
                    WHEN new.content <> old.content
                BEGIN
                    DELETE FROM memory_vec WHERE rowid = old.id;
                END;
            `);
        },
    },
    {
        // The append-only event log: the raw substrate every runtime signal
        // (message, tool_call, tool_result, recall, dream, ...) is written to.
        // It lives in the SAME database file and under the SAME user_version as
        // `memory`, so it ships as a migration in this one authoritative array
        // rather than a second migration runner (two runners on one user_version
        // would corrupt the version accounting). EventStore calls this same
        // migrate(); MemoryStore opening an events-migrated file is a no-op.
        //
        // It deliberately mirrors `memory`'s FTS5 + vector indexing exactly: an
        // external-content porter-stemmed FTS index over `content`, kept in sync
        // by an insert/delete/update trigger trio, and a selective vector table
        // keyed by event id with cascade-delete and content-update invalidation.
        // Events are append-only at the API layer, so the UPDATE triggers should
        // never fire in practice; we keep them anyway (belt-and-braces, and they
        // cost nothing if no UPDATE ever runs) so the indexes can never serve a
        // row that doesn't match the current content.
        name: "create event log table with fts and vector indices",
        up(db) {
            db.exec(`
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          INTEGER NOT NULL,
                    kind        TEXT NOT NULL,
                    role        TEXT,
                    content     TEXT NOT NULL,
                    meta        TEXT,
                    session     TEXT,
                    correlation TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts DESC);
                CREATE INDEX IF NOT EXISTS idx_events_kind ON events (kind);
                CREATE INDEX IF NOT EXISTS idx_events_session ON events (session);

                -- External-content FTS5 over content (same pattern as memory_fts):
                -- the porter stemmer folds morphological variants together so a
                -- lexical replay query is robust to exact wording.
                CREATE VIRTUAL TABLE event_fts USING fts5(
                    content,
                    content='events',
                    content_rowid='id',
                    tokenize='porter'
                );

                CREATE TRIGGER event_ai AFTER INSERT ON events BEGIN
                    INSERT INTO event_fts (rowid, content) VALUES (new.id, new.content);
                END;
                CREATE TRIGGER event_ad AFTER DELETE ON events BEGIN
                    INSERT INTO event_fts (event_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                END;
                CREATE TRIGGER event_au AFTER UPDATE ON events BEGIN
                    INSERT INTO event_fts (event_fts, rowid, content)
                        VALUES ('delete', old.id, old.content);
                    INSERT INTO event_fts (rowid, content) VALUES (new.id, new.content);
                END;

                -- Selective vector index (same pattern as memory_vec): the log is
                -- total, the vector index is opt-in. Most events (tool_result,
                -- system turns) never get a vector, which keeps the linear cosine
                -- scan from becoming the bottleneck. Rows are written explicitly
                -- by the application, never by an insert trigger.
                CREATE TABLE IF NOT EXISTS event_vec (
                    rowid INTEGER PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
                    dim   INTEGER NOT NULL,
                    vec   BLOB NOT NULL
                );

                CREATE TRIGGER event_vec_ad AFTER DELETE ON events BEGIN
                    DELETE FROM event_vec WHERE rowid = old.id;
                END;

                CREATE TRIGGER event_vec_au AFTER UPDATE OF content ON events
                    WHEN new.content <> old.content
                BEGIN
                    DELETE FROM event_vec WHERE rowid = old.id;
                END;
            `);
        },
    },
    {
        // The annotation overlay that makes a memory *curation over the log*
        // rather than a second, parallel store. Each row links one memory to the
        // event it was saved from: the turn in the transcript a fact was distilled
        // out of. This is the table `events.ts` reserves the log's shape for: the
        // log stays strictly content-bearing, and provenance lives here, on the
        // memory side, where it belongs (a memory's provenance is a property of
        // the memory, not the immutable event).
        //
        // The two foreign keys lean opposite directions on purpose:
        //  - memory_id is the PRIMARY KEY (one provenance row per memory) and
        //    CASCADEs: forget the memory and its provenance goes with it.
        //  - event_id is ON DELETE SET NULL. The log is append-only and never
        //    deletes in practice, so this is belt-and-braces; but if an event
        //    ever did vanish, the memory should survive with its provenance
        //    nulled, not be dragged down with it. A memory outranks the pointer.
        // Both cascades need `PRAGMA foreign_keys = ON`, which both stores set.
        name: "add memory_meta provenance overlay linking a memory to its event",
        up(db) {
            db.exec(`
                CREATE TABLE IF NOT EXISTS memory_meta (
                    memory_id INTEGER PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE,
                    event_id  INTEGER REFERENCES events(id) ON DELETE SET NULL,
                    created   INTEGER NOT NULL
                );
                -- Reverse lookup (which memories were curated from this event?)
                -- without scanning the whole overlay.
                CREATE INDEX IF NOT EXISTS idx_memory_meta_event ON memory_meta (event_id);
            `);
        },
    },
    {
        // The knowledge-base `notes` table: human-and-agent-editable markdown,
        // a separate corpus from `memory` (see NotesStore in notes.ts). It lives
        // in the SAME database file and under the SAME user_version as `memory`
        // and `events`, so it ships as a migration in this one authoritative
        // array rather than a second runner.
        //
        // The two columns that make two-way file sync tractable, and that the
        // memory machinery does NOT have, are:
        //  - `uuid`: a stable join key written into each file's frontmatter, so a
        //    note's identity survives a rename/move in either direction (the path
        //    can change; the uuid never does).
        //  - `path`: the note's relative location inside the KB folder; the
        //    "folder structure" is just this column, not a separate index.
        //  - `content_hash`: a hash of the synced content, the basis of conflict
        //    detection (cheaper and more reliable than timestamp comparison).
        // `frontmatter` keeps any human-added keys we don't model as columns, so
        // a person can add arbitrary metadata without a schema change.
        name: "create notes table with fts and vector indices",
        up(db) {
            db.exec(`
                CREATE TABLE IF NOT EXISTS notes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid         TEXT NOT NULL UNIQUE,
                    path         TEXT NOT NULL UNIQUE,
                    title        TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    frontmatter  TEXT,
                    content_hash TEXT NOT NULL,
                    created      INTEGER NOT NULL,
                    updated      INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_notes_updated ON notes (updated DESC);
                CREATE INDEX IF NOT EXISTS idx_notes_path ON notes (path);

                -- External-content FTS5 over content (same pattern as memory_fts):
                -- the porter stemmer folds morphological variants so recall is
                -- robust to exact wording. Title is indexed alongside content so a
                -- query matching only the title still surfaces the note.
                CREATE VIRTUAL TABLE notes_fts USING fts5(
                    title,
                    content,
                    content='notes',
                    content_rowid='id',
                    tokenize='porter'
                );

                CREATE TRIGGER notes_ai AFTER INSERT ON notes BEGIN
                    INSERT INTO notes_fts (rowid, title, content)
                        VALUES (new.id, new.title, new.content);
                END;
                CREATE TRIGGER notes_ad AFTER DELETE ON notes BEGIN
                    INSERT INTO notes_fts (notes_fts, rowid, title, content)
                        VALUES ('delete', old.id, old.title, old.content);
                END;
                CREATE TRIGGER notes_au AFTER UPDATE ON notes BEGIN
                    INSERT INTO notes_fts (notes_fts, rowid, title, content)
                        VALUES ('delete', old.id, old.title, old.content);
                    INSERT INTO notes_fts (rowid, title, content)
                        VALUES (new.id, new.title, new.content);
                END;

                -- Selective vector index (same pattern as memory_vec): vectors are
                -- written by the application (an embed is a network call), never by
                -- an insert trigger. We trigger on DELETE and on content UPDATE so a
                -- deleted note's vector goes with it and an edited note's stale
                -- vector is dropped for re-embed.
                CREATE TABLE IF NOT EXISTS notes_vec (
                    rowid INTEGER PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
                    dim   INTEGER NOT NULL,
                    vec   BLOB NOT NULL
                );

                CREATE TRIGGER notes_vec_ad AFTER DELETE ON notes BEGIN
                    DELETE FROM notes_vec WHERE rowid = old.id;
                END;

                CREATE TRIGGER notes_vec_au AFTER UPDATE OF content ON notes
                    WHEN new.content <> old.content
                BEGIN
                    DELETE FROM notes_vec WHERE rowid = old.id;
                END;
            `);
        },
    },
    {
        // The "linked" half of "separate store, linked": explicit relations from
        // a note to the memories and other notes it references. This is
        // deliberately NOT an Obsidian-style backlink graph: it stores the edges
        // a caller asserts, and the reverse lookup is a single indexed query, not
        // a transitive graph computation.
        //
        // A link points at a memory XOR a note when created (the store enforces
        // exactly-one at insert time). The cascades match each side's lifetime:
        // deleting the from-note drops its outgoing links; deleting a linked note
        // drops links pointing at it; a linked memory going away nulls the pointer
        // (a note's link should not vanish silently just because the memory it
        // referenced was forgotten, the way memory_meta nulls a deleted event).
        //
        // The CHECK is therefore "AT MOST one target", not "exactly one": the
        // exactly-one rule is an insert-time invariant the application owns, but a
        // to_memory SET NULL legitimately leaves a memory-link with both columns
        // null (the link survives, its target forgotten). A stricter "exactly one"
        // CHECK would make that SET NULL violate the constraint and block the
        // memory's deletion outright, which is the wrong failure.
        name: "add note_links relation table",
        up(db) {
            db.exec(`
                CREATE TABLE IF NOT EXISTS note_links (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_note INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                    to_memory INTEGER REFERENCES memory(id) ON DELETE SET NULL,
                    to_note   INTEGER REFERENCES notes(id) ON DELETE CASCADE,
                    kind      TEXT,
                    created   INTEGER NOT NULL,
                    -- At most one target set: rules out a link pointing at both a
                    -- memory and a note. (Exactly-one is enforced by the store at
                    -- insert; SET NULL may later leave both null, which is allowed.)
                    CHECK (NOT (to_memory IS NOT NULL AND to_note IS NOT NULL))
                );
                -- Forward lookup (a note's outgoing links) and reverse lookups (who
                -- links to this memory / this note), each a single indexed scan.
                CREATE INDEX IF NOT EXISTS idx_note_links_from ON note_links (from_note);
                CREATE INDEX IF NOT EXISTS idx_note_links_to_memory ON note_links (to_memory);
                CREATE INDEX IF NOT EXISTS idx_note_links_to_note ON note_links (to_note);
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
 * about: that means an older binary opened a future schema, and proceeding
 * could corrupt it. Fail loudly instead.
 *
 * Exported so sibling stores in the same database file (see {@link EventStore})
 * share this one migration runner rather than forking a second: there is a
 * single `user_version` per file, and two runners racing to advance it would
 * corrupt the version accounting.
 */
export function migrate(db: DatabaseSync): number {
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
 * for an isolated, ephemeral store: which is what the tests use so they never
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
    private readonly upsertVecStmt;
    private readonly deleteVecStmt;
    private readonly getVecStmt;
    private readonly missingVecStmt;
    private readonly upsertMetaStmt;
    private readonly getMetaStmt;
    private readonly deleteMetaStmt;
    private readonly memoriesFromEventStmt;
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

        // Honor the memory_vec → memory ON DELETE CASCADE (SQLite leaves foreign
        // keys off per connection by default). The explicit delete trigger is a
        // belt-and-braces backup, but the cascade is the contract.
        this.db.exec(`PRAGMA foreign_keys = ON`);

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

        this.upsertVecStmt = this.db.prepare(
            `INSERT INTO memory_vec (rowid, dim, vec) VALUES (?, ?, ?)
             ON CONFLICT(rowid) DO UPDATE SET dim = excluded.dim, vec = excluded.vec`,
        );
        this.deleteVecStmt = this.db.prepare(`DELETE FROM memory_vec WHERE rowid = ?`);
        this.getVecStmt = this.db.prepare(`SELECT vec FROM memory_vec WHERE rowid = ?`);
        // Memories with no (or a stale, hence absent) embedding yet: the
        // backfill work-list. Newest first so recent memories get vectors soonest.
        this.missingVecStmt = this.db.prepare(
            `SELECT m.id FROM memory m
             LEFT JOIN memory_vec v ON v.rowid = m.id
             WHERE v.rowid IS NULL
             ORDER BY m.created DESC
             LIMIT ?`,
        );

        // Provenance overlay: link a memory to the event it was curated from.
        // Upsert so re-pointing a memory's provenance replaces the old link
        // rather than erroring on the PRIMARY KEY.
        this.upsertMetaStmt = this.db.prepare(
            `INSERT INTO memory_meta (memory_id, event_id, created) VALUES (?, ?, ?)
             ON CONFLICT(memory_id) DO UPDATE SET event_id = excluded.event_id`,
        );
        this.getMetaStmt = this.db.prepare(`SELECT event_id FROM memory_meta WHERE memory_id = ?`);
        this.deleteMetaStmt = this.db.prepare(`DELETE FROM memory_meta WHERE memory_id = ?`);
        // Reverse lookup: which memories were curated from a given event, newest
        // memory first. Joins memory in so callers get full rows in one query.
        this.memoriesFromEventStmt = this.db.prepare(
            `SELECT m.* FROM memory_meta mm
             JOIN memory m ON m.id = mm.memory_id
             WHERE mm.event_id = ?
             ORDER BY m.created DESC, m.id DESC`,
        );
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
     * Unlike {@link search} (substring `LIKE`), matching here is token-based:
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

    // ── Vector / semantic search ──────────────────────────────────────────────
    //
    // The store owns *storage and comparison* of embeddings, never their
    // production: vectors are computed by an Embedder (a network call) and passed
    // in, keeping this class synchronous and I/O-free of any model API.

    /**
     * Store (or replace) the embedding for a memory. Returns false if no memory
     * with that id exists: we won't keep an orphan vector. The vector is
     * expected to be L2-normalized (as {@link Embedder} guarantees) so that
     * semantic ranking can use a plain dot product.
     */
    setEmbedding(id: number, vector: Float32Array): boolean {
        this.assertOpen();
        if (!this.get(id)) return false;
        this.upsertVecStmt.run(id, vector.length, vectorToBlob(vector));
        return true;
    }

    /** Drop a memory's embedding, e.g. before re-embedding. Returns whether a
     *  row was removed. (DELETE/content-UPDATE on the memory does this too, via
     *  trigger.) */
    deleteEmbedding(id: number): boolean {
        this.assertOpen();
        return this.deleteVecStmt.run(id).changes > 0;
    }

    /** Whether a memory currently has a stored embedding. */
    hasEmbedding(id: number): boolean {
        this.assertOpen();
        return this.getVecStmt.get(id) !== undefined;
    }

    /**
     * Ids of memories with no current embedding: the backfill work-list. A
     * memory loses its embedding when its content is edited (trigger), so this
     * also surfaces rows whose vectors went stale and need recomputing. Newest
     * first; bounded by `limit` (default {@link DEFAULT_LIMIT}).
     */
    idsMissingEmbedding(limit = DEFAULT_LIMIT): number[] {
        this.assertOpen();
        const rows = this.missingVecStmt.all(clampLimit(limit)) as Array<{ id: number }>;
        return rows.map((r) => r.id);
    }

    /**
     * Rank memories by semantic similarity to a query vector (cosine), highest
     * first. This is the meaning-based counterpart to {@link searchRelevant}'s
     * lexical match: it finds memories that mean the same thing even when they
     * share no words with the query.
     *
     * The query vector must come from the same embedding model the stored
     * vectors did: mismatched dimensions simply score 0 and drop out. Memories
     * without an embedding are invisible here (embed them first via
     * {@link setEmbedding} / a backfill).
     *
     * Implementation note: this scans every stored vector and ranks in JS. For a
     * personal memory store (thousands of rows) that's microseconds; a larger
     * corpus would want an ANN index, which is a future migration, not a rewrite
     * of this signature.
     */
    semanticSearch(query: Float32Array, opts: QueryOptions = {}): SemanticHit[] {
        this.assertOpen();
        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));
        const filterTags = normalizeTags(opts.tags);

        // Join the memory row in so we can apply the same tag filter as the other
        // queries and reconstruct full Memory objects without a second lookup.
        const where: string[] = [];
        const params: unknown[] = [];
        for (const tag of filterTags) {
            where.push(`m.tags LIKE ? ESCAPE '\\'`);
            params.push(`%"${escapeLike(tag)}"%`);
        }
        const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";
        const sql =
            `SELECT m.*, v.vec AS vec FROM memory_vec v ` +
            `JOIN memory m ON m.id = v.rowid ${whereSql}`;

        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as Array<
            MemoryRow & { vec: Uint8Array }
        >;

        const scored: SemanticHit[] = [];
        for (const row of rows) {
            const score = cosineSimilarity(query, blobToVector(row.vec));
            scored.push({ memory: rowToMemory(row), score });
        }
        // Highest similarity first; importance breaks ties among equally-similar
        // memories (mirrors the lexical path's secondary sort).
        scored.sort(
            (a, b) => b.score - a.score || (b.memory.importance ?? 0) - (a.memory.importance ?? 0),
        );
        return scored.slice(offset, offset + limit);
    }

    // ── Provenance overlay (curation over the log) ────────────────────────────
    //
    // A memory points at the event it was saved from: the turn in the transcript
    // a fact was distilled out of. This is what makes the store *curation over
    // the log* rather than a second, parallel store. The link lives in
    // `memory_meta`; these methods are its whole surface. The event log itself
    // stays annotation-free (see {@link EventStore}): provenance is a property of
    // the memory, recorded on the memory side.

    /**
     * Record that a memory was curated from a given event. Returns false if no
     * memory with that id exists (we won't keep an orphan provenance row).
     *
     * The `eventId` must reference a real row in the `events` table of the *same*
     * database file: the foreign key enforces it, so pointing at an event that
     * isn't there (e.g. when memory and the log live in separate files) throws a
     * FOREIGN KEY constraint error rather than recording a dangling link. The
     * intended layout is one shared file (see {@link EventStore}); a caller that
     * splits them gets no provenance, which is the right failure for a pointer
     * that can't be honored.
     *
     * Idempotent and re-pointable: calling again replaces the link rather than
     * erroring, so a re-curated memory tracks the latest event it came from.
     */
    setProvenance(memoryId: number, eventId: number, now = Date.now()): boolean {
        this.assertOpen();
        if (!this.get(memoryId)) return false;
        this.upsertMetaStmt.run(memoryId, eventId, now);
        return true;
    }

    /**
     * The event a memory was curated from, or undefined if it has no recorded
     * provenance (or its event was since deleted, nulling the link). Returns the
     * event *id*, not the event itself: the memory store doesn't hold the log, so
     * the caller reads the row from their {@link EventStore} over the same file.
     */
    provenanceOf(memoryId: number): number | undefined {
        this.assertOpen();
        const row = this.getMetaStmt.get(memoryId) as { event_id: number | null } | undefined;
        if (!row || row.event_id === null) return undefined;
        return row.event_id;
    }

    /** Drop a memory's provenance link without deleting the memory. Returns
     *  whether a row was removed. (Deleting the memory does this too, via the
     *  memory_id CASCADE.) */
    clearProvenance(memoryId: number): boolean {
        this.assertOpen();
        return this.deleteMetaStmt.run(memoryId).changes > 0;
    }

    /**
     * The reverse lookup: every memory curated from a given event, newest first.
     * Lets a caller go from a point in the transcript to the durable facts it
     * produced: the curation that event earned. Empty when nothing was saved
     * from it.
     */
    memoriesFromEvent(eventId: number): Memory[] {
        this.assertOpen();
        const rows = this.memoriesFromEventStmt.all(eventId) as unknown as MemoryRow[];
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
