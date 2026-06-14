/**
 * NotesStore: the knowledge-base substrate, a separate corpus from {@link MemoryStore}.
 *
 * Where `memory` holds short, agent-curated facts auto-injected into every turn,
 * `notes` holds longer human-and-agent documentation the agent opts into reading.
 * The two share the same machinery (one database file, one migration runner, the
 * same FTS5 + selective-vector pattern, the same shared SQLite helpers) but stay
 * distinct stores so human docs and agent memory chatter never bleed into one
 * another's recall.
 *
 * What makes a note different from a memory, and why this is its own class:
 *  - A note has a stable `uuid` (written into its file's frontmatter) and a
 *    `path` (its location in the KB folder). These are the identity and location
 *    that let the same row survive a rename or move on disk, in either direction.
 *    The uuid is the join key two-way file sync hangs on; getting it onto every
 *    row from day one is why retrofitting it later would be painful.
 *  - A note carries a `content_hash` of its synced state, the basis of conflict
 *    detection (cheaper and more reliable than comparing timestamps alone).
 *  - A note keeps a `frontmatter` JSON blob of any human-added keys we don't
 *    model as columns, so a person can add arbitrary metadata in their editor
 *    without a schema change.
 *  - Notes relate to memories and to each other through {@link note_links}:
 *    explicit edges, deliberately not an Obsidian-style transitive backlink graph.
 *
 * Like its siblings this store owns *storage and comparison* of embeddings, never
 * their production (that is an {@link Embedder}'s job), so it stays synchronous
 * and free of any model API. A corrupt `frontmatter` payload degrades to `{}` on
 * read rather than throwing and taking down a whole query.
 */

import { DatabaseSync } from "node:sqlite";
import { createHash } from "node:crypto";
import { blobToVector, cosineSimilarity, vectorToBlob } from "./embeddings.ts";
import { migrate, DEFAULT_BUSY_TIMEOUT, type StoreOptions } from "./memory.ts";
import { clampLimit, escapeLike, toFtsQuery, DEFAULT_LIMIT, MAX_CONTENT_LENGTH } from "./sqlite.ts";
import type { FrontmatterValue } from "./frontmatter.ts";

/** Thrown when a note fails validation before it ever reaches the database.
 *  Mirrors {@link MemoryError}: distinguishes "bad data" from a storage fault. */
export class NoteError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "NoteError";
    }
}

/** Hard ceiling on a note's title, and on its relative path length. */
export const MAX_TITLE_LENGTH = 512;
export const MAX_PATH_LENGTH = 1024;

/** The frontmatter map we persist: arbitrary human keys plus the ones we model.
 *  Values follow the frontmatter subset (scalars and string arrays). */
export type NoteFrontmatter = Record<string, FrontmatterValue>;

/** Fields a caller may supply when creating a note. The store assigns `id`, and
 *  derives `content_hash` and `updated`; `uuid` is minted when omitted. */
export interface NoteInput {
    /** Stable identity, written into the file's frontmatter. Minted when omitted
     *  (a brand-new note); supplied when adopting an existing file's uuid. */
    uuid?: string;
    /** Relative path within the KB folder, e.g. `subdir/title.md`. Required. */
    path: string;
    /** Display title. Required, non-empty. */
    title: string;
    /** Markdown body (frontmatter stripped). FTS- and vector-indexed. */
    content: string;
    /** Arbitrary frontmatter keys not modeled as columns. */
    frontmatter?: NoteFrontmatter;
    /** Defaults to "now"; injectable so tests are deterministic. */
    created?: number;
}

/** Options for the note read surface. */
export interface NoteQueryOptions {
    /** Max rows. Defaults to {@link DEFAULT_LIMIT}; capped at MAX_LIMIT. */
    limit?: number;
    /** Rows to skip (pagination). */
    offset?: number;
    /** Only notes whose path begins with this prefix (folder filter), e.g.
     *  `projects/` returns everything under that subtree. */
    pathPrefix?: string;
}

/** A note paired with its cosine similarity to a query vector. */
export interface NoteSemanticHit {
    note: Note;
    score: number;
}

/** The kind of thing a {@link NoteLink} points at, and to which id. */
export interface NoteLink {
    id: number;
    fromNote: number;
    /** Exactly one of these is set; the other is null. */
    toMemory: number | null;
    toNote: number | null;
    kind: string | null;
    created: number;
}

/**
 * A persisted note. Construct via {@link NotesStore.save}; the store assigns the
 * real `id`, derives `contentHash`, and mints a `uuid` when none was supplied.
 */
export class Note {
    id: number;
    uuid: string;
    path: string;
    title: string;
    content: string;
    frontmatter: NoteFrontmatter;
    contentHash: string;
    created: number;
    /** Last time the row was written (created or updated). */
    updated: number;

    constructor(input: NoteInput) {
        const norm = normalizeInput(input);
        this.id = 0;
        this.uuid = norm.uuid;
        this.path = norm.path;
        this.title = norm.title;
        this.content = norm.content;
        this.frontmatter = norm.frontmatter;
        this.contentHash = hashContent(norm.title, norm.content, norm.frontmatter);
        this.created = norm.created;
        this.updated = norm.created;
    }
}

/** The canonical, validated shape we persist. */
interface NormalizedNote {
    uuid: string;
    path: string;
    title: string;
    content: string;
    frontmatter: NoteFrontmatter;
    created: number;
}

/**
 * Hash the parts of a note that define its synced state: title, body, and the
 * modeled frontmatter. This is the basis of conflict detection: if the hash a
 * writer computed differs from the row's stored hash, the row changed underneath
 * it. We hash a canonical JSON of {title, content, frontmatter} (keys sorted) so
 * the hash is stable regardless of frontmatter key order.
 */
export function hashContent(title: string, content: string, frontmatter: NoteFrontmatter): string {
    const canonical = JSON.stringify({
        title,
        content,
        frontmatter: sortKeys(frontmatter),
    });
    return createHash("sha256").update(canonical, "utf8").digest("hex");
}

/** Return a shallow copy of an object with keys in sorted order, so a canonical
 *  JSON of it is stable across insertion order. */
function sortKeys(obj: NoteFrontmatter): NoteFrontmatter {
    const out: NoteFrontmatter = {};
    for (const key of Object.keys(obj).sort()) out[key] = obj[key];
    return out;
}

/** Validate and normalize raw note input. Throws {@link NoteError} on anything
 *  we refuse to store. */
function normalizeInput(input: NoteInput): NormalizedNote {
    if (input === null || typeof input !== "object") {
        throw new NoteError("note input must be an object");
    }

    const path = normalizePath(input.path);

    if (typeof input.title !== "string") {
        throw new NoteError("title must be a string");
    }
    const title = input.title.trim();
    if (title.length === 0) {
        throw new NoteError("title must not be empty");
    }
    if (title.length > MAX_TITLE_LENGTH) {
        throw new NoteError(`title exceeds ${MAX_TITLE_LENGTH} characters`);
    }

    if (typeof input.content !== "string") {
        throw new NoteError("content must be a string");
    }
    if (input.content.length > MAX_CONTENT_LENGTH) {
        throw new NoteError(`content exceeds ${MAX_CONTENT_LENGTH} characters`);
    }
    // A note's body may be empty (a title-only stub is valid); memory's
    // non-empty-content rule does not apply here.
    const content = input.content;

    const frontmatter = normalizeFrontmatter(input.frontmatter);

    const uuid = input.uuid === undefined ? mintUuid() : validateUuid(input.uuid);

    let created = input.created;
    if (created === undefined) {
        created = Date.now();
    } else if (typeof created !== "number" || !Number.isFinite(created)) {
        throw new NoteError("created must be a finite number");
    }

    return { uuid, path, title, content, frontmatter, created };
}

/**
 * Validate and normalize a note's relative path. We forbid absolute paths,
 * parent-directory escapes (`..`), and backslashes, so a path can never point
 * outside the KB folder when the sync engine joins it onto the KB root: a
 * path-traversal guard, not just tidiness. The path must end in `.md`.
 */
export function normalizePath(raw: unknown): string {
    if (typeof raw !== "string") {
        throw new NoteError("path must be a string");
    }
    // Normalize separators and collapse redundant slashes.
    let path = raw
        .trim()
        .replace(/\\/g, "/")
        .replace(/\/{2,}/g, "/");
    // Strip a leading slash: paths are always relative to the KB root.
    path = path.replace(/^\/+/, "");
    if (path.length === 0) {
        throw new NoteError("path must not be empty");
    }
    if (path.length > MAX_PATH_LENGTH) {
        throw new NoteError(`path exceeds ${MAX_PATH_LENGTH} characters`);
    }
    const segments = path.split("/");
    for (const seg of segments) {
        if (seg === "..") {
            throw new NoteError("path must not contain '..' segments");
        }
        if (seg === "") {
            // A trailing slash or empty interior segment is malformed.
            throw new NoteError("path must not contain empty segments");
        }
        // Reserve the null byte and control chars; an editor never produces them
        // and they would corrupt a filename.
        if (/[\x00-\x1f\x7f]/.test(seg)) {
            throw new NoteError("path must not contain control characters");
        }
    }
    if (!path.toLowerCase().endsWith(".md")) {
        throw new NoteError("path must end in .md");
    }
    return path;
}

/** Validate and normalize the frontmatter map: reject reserved column-backed keys
 *  (uuid/title/path live in real columns, not the blob) and non-subset values. */
function normalizeFrontmatter(raw: NoteFrontmatter | undefined): NoteFrontmatter {
    if (raw === undefined || raw === null) return {};
    if (typeof raw !== "object" || Array.isArray(raw)) {
        throw new NoteError("frontmatter must be an object");
    }
    const reserved = new Set(["uuid", "title", "path"]);
    const out: NoteFrontmatter = {};
    for (const [key, value] of Object.entries(raw)) {
        if (reserved.has(key)) continue; // modeled as a column; don't double-store
        if (!isFrontmatterValue(value)) {
            throw new NoteError(
                `frontmatter value for "${key}" is not a supported scalar or string array`,
            );
        }
        out[key] = value;
    }
    return out;
}

/** Whether a value fits the frontmatter subset (scalar or string array). */
function isFrontmatterValue(v: unknown): v is FrontmatterValue {
    if (v === null) return true;
    const t = typeof v;
    if (t === "string" || t === "boolean") return true;
    if (t === "number") return Number.isFinite(v as number);
    if (Array.isArray(v)) return v.every((x) => typeof x === "string");
    return false;
}

/** Mint a fresh v4-ish uuid using the platform's crypto. */
export function mintUuid(): string {
    // randomUUID is available on globalThis.crypto in Node 19+.
    return globalThis.crypto.randomUUID();
}

/** Validate a supplied uuid: a non-empty, reasonably-shaped string. We don't
 *  force a strict v4 regex (a human could paste any stable id), but we forbid
 *  whitespace and bound the length so it stays a usable join key. */
function validateUuid(raw: unknown): string {
    if (typeof raw !== "string") {
        throw new NoteError("uuid must be a string");
    }
    const uuid = raw.trim();
    if (uuid.length === 0) {
        throw new NoteError("uuid must not be empty");
    }
    if (uuid.length > 128) {
        throw new NoteError("uuid is too long");
    }
    if (/\s/.test(uuid)) {
        throw new NoteError("uuid must not contain whitespace");
    }
    return uuid;
}

interface NoteRow {
    id: number;
    uuid: string;
    path: string;
    title: string;
    content: string;
    frontmatter: string | null;
    content_hash: string;
    created: number;
    updated: number;
}

/** Reconstruct a {@link Note} from a db row. Tolerant of a corrupt
 *  `frontmatter` payload: a row whose blob doesn't parse degrades to `{}`. */
function rowToNote(row: NoteRow): Note {
    const note = Object.create(Note.prototype) as Note;
    note.id = row.id;
    note.uuid = row.uuid;
    note.path = row.path;
    note.title = row.title;
    note.content = row.content;
    note.frontmatter = parseFrontmatterBlob(row.frontmatter);
    note.contentHash = row.content_hash;
    note.created = row.created;
    note.updated = row.updated;
    return note;
}

function parseFrontmatterBlob(raw: string | null): NoteFrontmatter {
    if (!raw) return {};
    try {
        const parsed: unknown = JSON.parse(raw);
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return {};
        const out: NoteFrontmatter = {};
        for (const [k, v] of Object.entries(parsed)) {
            if (isFrontmatterValue(v)) out[k] = v;
        }
        return out;
    } catch {
        return {};
    }
}

function serializeFrontmatter(fm: NoteFrontmatter): string | null {
    return Object.keys(fm).length ? JSON.stringify(fm) : null;
}

/** Fields a caller may patch on an existing note. `uuid` is immutable. */
export type NotePatch = Partial<Pick<NoteInput, "path" | "title" | "content" | "frontmatter">>;

/**
 * SQLite-backed store for {@link Note} objects, the KB's curated corpus.
 *
 * Shares one database file and one schema `user_version` with {@link MemoryStore}
 * and {@link EventStore}: all three call the same {@link migrate}. File-backed
 * stores run in WAL mode by default. Construct one, use it, {@link close} it.
 */
export class NotesStore {
    private readonly db: DatabaseSync;
    private readonly walEnabled: boolean;
    private readonly schemaVersion: number;
    private readonly insertStmt;
    private readonly getStmt;
    private readonly getByUuidStmt;
    private readonly getByPathStmt;
    private readonly updateStmt;
    private readonly deleteStmt;
    private readonly countStmt;
    private readonly upsertVecStmt;
    private readonly deleteVecStmt;
    private readonly getVecStmt;
    private readonly missingVecStmt;
    private readonly insertLinkStmt;
    private readonly linksFromStmt;
    private readonly linksToMemoryStmt;
    private readonly linksToNoteStmt;
    private readonly deleteLinkStmt;
    private closed = false;

    constructor(options: string | StoreOptions = "db.sqlite") {
        const opts: StoreOptions = typeof options === "string" ? { location: options } : options;
        const location = opts.location ?? "db.sqlite";
        const busyTimeout = opts.busyTimeout ?? DEFAULT_BUSY_TIMEOUT;
        const wantWal = (opts.wal ?? true) && location !== ":memory:";

        this.db = new DatabaseSync(location);
        this.db.exec(`PRAGMA busy_timeout = ${Math.max(0, Math.floor(busyTimeout))}`);
        // Honor the notes_vec / note_links ON DELETE cascades.
        this.db.exec(`PRAGMA foreign_keys = ON`);

        if (wantWal) {
            const row = this.db.prepare(`PRAGMA journal_mode = WAL`).get() as {
                journal_mode?: string;
            };
            this.walEnabled = row?.journal_mode?.toLowerCase() === "wal";
            if (this.walEnabled) this.db.exec(`PRAGMA synchronous = NORMAL`);
        } else {
            this.walEnabled = false;
        }

        this.schemaVersion = migrate(this.db);

        this.insertStmt = this.db.prepare(
            `INSERT INTO notes (uuid, path, title, content, frontmatter, content_hash, created, updated)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        );
        this.getStmt = this.db.prepare(`SELECT * FROM notes WHERE id = ?`);
        this.getByUuidStmt = this.db.prepare(`SELECT * FROM notes WHERE uuid = ?`);
        this.getByPathStmt = this.db.prepare(`SELECT * FROM notes WHERE path = ?`);
        this.updateStmt = this.db.prepare(
            `UPDATE notes SET path = ?, title = ?, content = ?, frontmatter = ?,
                              content_hash = ?, updated = ?
             WHERE id = ?`,
        );
        this.deleteStmt = this.db.prepare(`DELETE FROM notes WHERE id = ?`);
        this.countStmt = this.db.prepare(`SELECT COUNT(*) AS n FROM notes`);

        this.upsertVecStmt = this.db.prepare(
            `INSERT INTO notes_vec (rowid, dim, vec) VALUES (?, ?, ?)
             ON CONFLICT(rowid) DO UPDATE SET dim = excluded.dim, vec = excluded.vec`,
        );
        this.deleteVecStmt = this.db.prepare(`DELETE FROM notes_vec WHERE rowid = ?`);
        this.getVecStmt = this.db.prepare(`SELECT vec FROM notes_vec WHERE rowid = ?`);
        this.missingVecStmt = this.db.prepare(
            `SELECT n.id FROM notes n
             LEFT JOIN notes_vec v ON v.rowid = n.id
             WHERE v.rowid IS NULL
             ORDER BY n.updated DESC, n.id DESC
             LIMIT ?`,
        );

        this.insertLinkStmt = this.db.prepare(
            `INSERT INTO note_links (from_note, to_memory, to_note, kind, created)
             VALUES (?, ?, ?, ?, ?)`,
        );
        this.linksFromStmt = this.db.prepare(
            `SELECT * FROM note_links WHERE from_note = ? ORDER BY id`,
        );
        this.linksToMemoryStmt = this.db.prepare(
            `SELECT * FROM note_links WHERE to_memory = ? ORDER BY id`,
        );
        this.linksToNoteStmt = this.db.prepare(
            `SELECT * FROM note_links WHERE to_note = ? ORDER BY id`,
        );
        this.deleteLinkStmt = this.db.prepare(`DELETE FROM note_links WHERE id = ?`);
    }

    private assertOpen() {
        if (this.closed) throw new NoteError("store is closed");
    }

    /**
     * Persist a note, assigning its real id. The `uuid` and `path` are UNIQUE; a
     * collision (a second note minted onto an existing path, or a duplicate uuid)
     * surfaces as a {@link NoteError} rather than an opaque sqlite constraint
     * error, so a caller can react (e.g. the sync engine treating a path clash as
     * a different note).
     */
    save(note: Note): Note {
        this.assertOpen();
        // Re-validate: the Note may have been mutated after construction.
        const norm = normalizeInput(note);
        const hash = hashContent(norm.title, norm.content, norm.frontmatter);
        try {
            const result = this.insertStmt.run(
                norm.uuid,
                norm.path,
                norm.title,
                norm.content,
                serializeFrontmatter(norm.frontmatter),
                hash,
                norm.created,
                note.updated || norm.created,
            );
            note.id = Number(result.lastInsertRowid);
        } catch (err) {
            throw asNoteError(err);
        }
        note.uuid = norm.uuid;
        note.path = norm.path;
        note.title = norm.title;
        note.content = norm.content;
        note.frontmatter = norm.frontmatter;
        note.contentHash = hash;
        note.created = norm.created;
        return note;
    }

    /**
     * Apply edits to an existing note and persist them. `uuid` and `created` are
     * immutable; `updated` is stamped to `now`, and `content_hash` is recomputed.
     * Returns the refreshed note, or undefined if no row with that id exists.
     */
    update(id: number, patch: NotePatch, now = Date.now()): Note | undefined {
        this.assertOpen();
        const existing = this.get(id);
        if (!existing) return undefined;

        // Merge then re-validate the whole shape, preserving the immutable uuid.
        const merged = normalizeInput({
            uuid: existing.uuid,
            path: patch.path ?? existing.path,
            title: patch.title ?? existing.title,
            content: patch.content ?? existing.content,
            frontmatter: "frontmatter" in patch ? patch.frontmatter : existing.frontmatter,
            created: existing.created,
        });
        const hash = hashContent(merged.title, merged.content, merged.frontmatter);

        try {
            this.updateStmt.run(
                merged.path,
                merged.title,
                merged.content,
                serializeFrontmatter(merged.frontmatter),
                hash,
                now,
                id,
            );
        } catch (err) {
            throw asNoteError(err);
        }

        existing.path = merged.path;
        existing.title = merged.title;
        existing.content = merged.content;
        existing.frontmatter = merged.frontmatter;
        existing.contentHash = hash;
        existing.updated = now;
        return existing;
    }

    get(id: number): Note | undefined {
        this.assertOpen();
        const row = this.getStmt.get(id) as NoteRow | undefined;
        return row ? rowToNote(row) : undefined;
    }

    /** Fetch by the stable join key (the identity that survives renames). */
    getByUuid(uuid: string): Note | undefined {
        this.assertOpen();
        const row = this.getByUuidStmt.get(uuid) as NoteRow | undefined;
        return row ? rowToNote(row) : undefined;
    }

    /** Fetch by relative path (the file's location in the KB folder). */
    getByPath(path: string): Note | undefined {
        this.assertOpen();
        let key: string;
        try {
            key = normalizePath(path);
        } catch {
            // An invalid path can never match a stored (normalized) path.
            return undefined;
        }
        const row = this.getByPathStmt.get(key) as NoteRow | undefined;
        return row ? rowToNote(row) : undefined;
    }

    /** Notes ordered by recency (updated desc), with optional path-prefix folder
     *  filter, limit, and offset. */
    all(opts: NoteQueryOptions = {}): Note[] {
        return this.query(null, opts);
    }

    /** Case-insensitive substring search over title+content, otherwise like
     *  {@link all}. An empty query behaves like {@link all}. */
    search(text: string, opts: NoteQueryOptions = {}): Note[] {
        const needle = typeof text === "string" ? text.trim() : "";
        return this.query(needle.length ? needle : null, opts);
    }

    /**
     * Rank notes by lexical relevance to `text` (FTS5/bm25 over title+content),
     * the lexical counterpart to {@link semanticSearch}. Token-based matching
     * (see {@link toFtsQuery}); a token-less query or no hits yields [].
     */
    searchRelevant(text: string, opts: NoteQueryOptions = {}): Note[] {
        this.assertOpen();
        const match = toFtsQuery(text);
        if (match === null) return [];

        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));

        const where: string[] = ["notes_fts MATCH ?"];
        const params: unknown[] = [match];
        appendPathPrefix(where, params, opts.pathPrefix, "n");

        const sql =
            `SELECT n.* FROM notes_fts ` +
            `JOIN notes n ON n.id = notes_fts.rowid ` +
            `WHERE ${where.join(" AND ")} ` +
            `ORDER BY bm25(notes_fts), n.updated DESC, n.id DESC ` +
            `LIMIT ? OFFSET ?`;
        params.push(limit, offset);

        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as NoteRow[];
        return rows.map(rowToNote);
    }

    /**
     * Rank notes by semantic similarity to a query vector (cosine), highest
     * first, the meaning-based counterpart to {@link searchRelevant}. Notes
     * without an embedding are invisible here. A dimension mismatch scores 0.
     */
    semanticSearch(query: Float32Array, opts: NoteQueryOptions = {}): NoteSemanticHit[] {
        this.assertOpen();
        const where: string[] = [];
        const params: unknown[] = [];
        appendPathPrefix(where, params, opts.pathPrefix, "n");
        const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";

        const sql =
            `SELECT n.*, v.vec AS vec FROM notes_vec v ` +
            `JOIN notes n ON n.id = v.rowid ${whereSql}`;

        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as Array<
            NoteRow & { vec: Uint8Array }
        >;

        const scored: NoteSemanticHit[] = [];
        for (const row of rows) {
            const score = cosineSimilarity(query, blobToVector(row.vec));
            scored.push({ note: rowToNote(row), score });
        }
        scored.sort(
            (a, b) => b.score - a.score || b.note.updated - a.note.updated || b.note.id - a.note.id,
        );

        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));
        return scored.slice(offset, offset + limit);
    }

    // ── Embeddings (selective) ────────────────────────────────────────────────

    /** Store (or replace) a note's embedding. Returns false if no note with that
     *  id exists (no orphan vectors). Expects an L2-normalized vector. */
    setEmbedding(id: number, vector: Float32Array): boolean {
        this.assertOpen();
        if (!this.getStmt.get(id)) return false;
        this.upsertVecStmt.run(id, vector.length, vectorToBlob(vector));
        return true;
    }

    /** Drop a note's embedding (e.g. before re-embedding). Returns whether a row
     *  was removed. (Deleting the note, or editing its content, does this too.) */
    deleteEmbedding(id: number): boolean {
        this.assertOpen();
        return this.deleteVecStmt.run(id).changes > 0;
    }

    /** Whether a note currently has a stored embedding. */
    hasEmbedding(id: number): boolean {
        this.assertOpen();
        return this.getVecStmt.get(id) !== undefined;
    }

    /** Ids of notes with no current embedding: the backfill work-list, newest
     *  first, bounded by `limit`. */
    idsMissingEmbedding(limit = DEFAULT_LIMIT): number[] {
        this.assertOpen();
        const rows = this.missingVecStmt.all(clampLimit(limit)) as Array<{ id: number }>;
        return rows.map((r) => r.id);
    }

    // ── Links (explicit relations, not a backlink graph) ──────────────────────

    /**
     * Link a note to a memory or to another note. Exactly one of `toMemory` /
     * `toNote` must be set (the CHECK enforces it). Returns the created link.
     * A reference to a non-existent target surfaces as a {@link NoteError}
     * (foreign key), so a dangling link is never recorded.
     */
    link(
        fromNote: number,
        target: { toMemory: number } | { toNote: number },
        kind?: string,
        now = Date.now(),
    ): NoteLink {
        this.assertOpen();
        const toMemory = "toMemory" in target ? target.toMemory : null;
        const toNote = "toNote" in target ? target.toNote : null;
        if ((toMemory === null) === (toNote === null)) {
            throw new NoteError("a link must target exactly one of a memory or a note");
        }
        if (toNote !== null && toNote === fromNote) {
            throw new NoteError("a note cannot link to itself");
        }
        let result;
        try {
            result = this.insertLinkStmt.run(fromNote, toMemory, toNote, kind ?? null, now);
        } catch (err) {
            throw asNoteError(err);
        }
        return {
            id: Number(result.lastInsertRowid),
            fromNote,
            toMemory,
            toNote,
            kind: kind ?? null,
            created: now,
        };
    }

    /** A note's outgoing links (what it references). */
    linksFrom(noteId: number): NoteLink[] {
        this.assertOpen();
        return (this.linksFromStmt.all(noteId) as unknown as LinkRow[]).map(rowToLink);
    }

    /** Reverse lookup: links pointing at a given memory. */
    linksToMemory(memoryId: number): NoteLink[] {
        this.assertOpen();
        return (this.linksToMemoryStmt.all(memoryId) as unknown as LinkRow[]).map(rowToLink);
    }

    /** Reverse lookup: links pointing at a given note. */
    linksToNote(noteId: number): NoteLink[] {
        this.assertOpen();
        return (this.linksToNoteStmt.all(noteId) as unknown as LinkRow[]).map(rowToLink);
    }

    /** Remove a single link by its id. Returns whether a row was removed. */
    unlink(linkId: number): boolean {
        this.assertOpen();
        return this.deleteLinkStmt.run(linkId).changes > 0;
    }

    // ── Counting / lifecycle ──────────────────────────────────────────────────

    count(): number {
        this.assertOpen();
        const row = this.countStmt.get() as { n: number };
        return row.n;
    }

    delete(id: number): boolean {
        this.assertOpen();
        return this.deleteStmt.run(id).changes > 0;
    }

    private query(needle: string | null, opts: NoteQueryOptions): Note[] {
        this.assertOpen();
        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));

        const where: string[] = [];
        const params: unknown[] = [];
        if (needle !== null) {
            // Substring over title OR content, both escaped.
            where.push(`(title LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\')`);
            const like = `%${escapeLike(needle)}%`;
            params.push(like, like);
        }
        appendPathPrefix(where, params, opts.pathPrefix, null);

        const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";
        const sql =
            `SELECT * FROM notes ${whereSql} ` +
            `ORDER BY updated DESC, id DESC ` +
            `LIMIT ? OFFSET ?`;
        params.push(limit, offset);

        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as NoteRow[];
        return rows.map(rowToNote);
    }

    /** Whether this store is actually running in WAL mode. */
    get wal(): boolean {
        return this.walEnabled;
    }

    /** The schema version the underlying database was migrated to on open. */
    get version(): number {
        return this.schemaVersion;
    }

    /** Fold the WAL back into the main db file and truncate it. No-op without WAL. */
    checkpoint(): void {
        this.assertOpen();
        if (!this.walEnabled) return;
        this.db.exec(`PRAGMA wal_checkpoint(TRUNCATE)`);
    }

    /** Release the database handle. Idempotent. Checkpoints first so no populated
     *  `-wal` sidecar is left behind. */
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

interface LinkRow {
    id: number;
    from_note: number;
    to_memory: number | null;
    to_note: number | null;
    kind: string | null;
    created: number;
}

function rowToLink(row: LinkRow): NoteLink {
    return {
        id: row.id,
        fromNote: row.from_note,
        toMemory: row.to_memory,
        toNote: row.to_note,
        kind: row.kind,
        created: row.created,
    };
}

/** Add a `path LIKE prefix%` folder filter when a (non-empty) prefix is given. */
function appendPathPrefix(
    where: string[],
    params: unknown[],
    prefix: string | undefined,
    alias: string | null,
): void {
    if (typeof prefix !== "string" || prefix.trim() === "") return;
    const col = alias ? `${alias}.path` : "path";
    where.push(`${col} LIKE ? ESCAPE '\\'`);
    params.push(`${escapeLike(prefix)}%`);
}

/** Translate a raw sqlite error into a {@link NoteError} with a legible message,
 *  recognizing the UNIQUE (uuid/path) and FOREIGN KEY constraint cases the store
 *  can hit, so callers get "duplicate path" rather than a driver string. */
function asNoteError(err: unknown): NoteError {
    const message = err instanceof Error ? err.message : String(err);
    if (/UNIQUE constraint failed: notes\.path/.test(message)) {
        return new NoteError("a note already exists at that path");
    }
    if (/UNIQUE constraint failed: notes\.uuid/.test(message)) {
        return new NoteError("a note with that uuid already exists");
    }
    if (/FOREIGN KEY/.test(message)) {
        return new NoteError("link target does not exist");
    }
    if (/CHECK constraint failed/.test(message)) {
        return new NoteError("a link must target exactly one of a memory or a note");
    }
    // Re-wrap anything else so callers always get a NoteError from the store.
    return new NoteError(message);
}
