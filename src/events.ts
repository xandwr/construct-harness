/**
 * EventStore: the append-only event log, the raw substrate beneath memory.
 *
 * Where {@link MemoryStore} holds a curated set of facts the agent chose to keep,
 * the event log holds *everything that happened*: every message, tool call, tool
 * result, recall, and dream, in the order it occurred. It is the source of truth
 * the higher-level views are meant to be scoped queries over. Memory is an
 * annotation overlay on top of this: the `memory_meta(memory_id, event_id, ...)`
 * table (see {@link MemoryStore.setProvenance}) links a curated fact back to the
 * event it was distilled from, which is why this store keeps `events` strictly
 * content-bearing and annotation-free, and never exposes a content UPDATE or
 * DELETE: immutability is the substrate's whole value. The only mutable thing
 * EventStore owns is an event's embedding (set/delete, for re-embed).
 *
 * Two indexing properties shape the design:
 *  - The log is total; the vector index is selective. {@link EventStore.append}
 *    never embeds. Embedding is a separate, explicit, opt-in call (see
 *    {@link EventStore.setEmbedding} and the {@link EventStore.idsMissingEmbedding}
 *    backfill). Most events never get a vector, which keeps the linear cosine
 *    scan in {@link EventStore.semanticSearch} from becoming the bottleneck.
 *  - The FTS index is total: every appended event is lexically searchable for
 *    free (it costs only an inverted-index entry, computed in SQL by a trigger).
 *
 * Degradation story: a corrupt `meta` payload on read degrades to `undefined`
 * rather than throwing and taking a whole query down (mirrors memory's tolerant
 * tag parse). A mismatched query-vector dimension simply scores 0 and drops out.
 *
 * This store is provider-neutral: it imports only `node:sqlite`, the embedding
 * (de)serializers, and the shared SQLite helpers. It stores and compares
 * vectors but never produces them (that is an {@link Embedder}'s job), so it
 * stays synchronous and free of any model API.
 */

import { DatabaseSync } from "node:sqlite";
import { blobToVector, cosineSimilarity, vectorToBlob } from "./embeddings.ts";
import { migrate, DEFAULT_BUSY_TIMEOUT, type StoreOptions } from "./memory.ts";
import { clampLimit, toFtsQuery, DEFAULT_LIMIT, MAX_CONTENT_LENGTH } from "./sqlite.ts";

/**
 * Thrown when an event fails validation before it ever reaches the database.
 * Callers can `instanceof`-check this to distinguish "you gave me bad data"
 * from a genuine sqlite/storage failure. Mirrors {@link MemoryError}.
 */
export class EventError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "EventError";
    }
}

/** Fields a caller may supply when appending an event. */
export interface EventInput {
    /** What kind of signal this is: 'message' | 'tool_call' | 'tool_result' |
     *  'recall' | 'dream' | ... Required, non-empty. Free-form by design so new
     *  signal kinds need no schema change. */
    kind: string;
    /** The human/textual payload, FTS-indexed. Required, non-empty, bounded by
     *  {@link MAX_CONTENT_LENGTH}. */
    content: string;
    /** Who/what produced it: 'user' | 'agent' | 'system' | 'tool' | undefined. */
    role?: string;
    /** Optional structured payload (tool args, ids, ...). JSON-serialized into
     *  the `meta` column; rejected at validation time if it can't stringify. */
    meta?: unknown;
    /** Optional id grouping the events of one conversation. */
    session?: string;
    /** Optional id threading a tool_call to its tool_result. */
    correlation?: string;
    /** Event time in ms. Defaults to {@link Date.now}; injectable for tests. */
    ts?: number;
}

/**
 * A persisted event. Construct via {@link EventStore.append}; the store assigns
 * the real `id`. `ts` is the event time in ms. `meta` is the parsed structured
 * payload (or `undefined` if absent or corrupt on read).
 */
export class Event {
    id: number;
    ts: number;
    kind: string;
    role?: string;
    content: string;
    meta?: unknown;
    session?: string;
    correlation?: string;

    constructor(fields: {
        id?: number;
        ts: number;
        kind: string;
        role?: string;
        content: string;
        meta?: unknown;
        session?: string;
        correlation?: string;
    }) {
        this.id = fields.id ?? 0;
        this.ts = fields.ts;
        this.kind = fields.kind;
        this.role = fields.role;
        this.content = fields.content;
        this.meta = fields.meta;
        this.session = fields.session;
        this.correlation = fields.correlation;
    }
}

/** Filters and pagination for the event read surface. */
export interface EventQuery {
    /** Max rows to return. Defaults to {@link DEFAULT_LIMIT}; capped at MAX_LIMIT. */
    limit?: number;
    /** Number of rows to skip (for pagination). */
    offset?: number;
    /** Only return events of this kind. */
    kind?: string;
    /** Only return events in this session. */
    session?: string;
    /** Only return events with `ts >= since` (inclusive). */
    since?: number;
    /** Only return events with `ts <= until` (inclusive). */
    until?: number;
}

/** An event paired with its cosine similarity to a query vector, from
 *  {@link EventStore.semanticSearch}. Higher score = more similar. */
export interface EventSemanticHit {
    event: Event;
    score: number;
}

/** Allowed image MIME types for an attachment, kept in lockstep with the core
 *  {@link ImagePart} union and what the Anthropic bridge will accept. */
export type AttachmentMediaType = "image/jpeg" | "image/png";

/** What a caller supplies to attach one image to an event. `data` is the raw
 *  bytes (the store keeps them as a BLOB, not base64). */
export interface AttachmentInput {
    mediaType: AttachmentMediaType;
    data: Uint8Array;
    filename?: string;
}

/** An attachment's metadata, without its bytes: enough to list and label an
 *  event's images (and form the URL to fetch each) without loading megabytes.
 *  See {@link EventStore.attachmentsFor}. */
export interface AttachmentMeta {
    id: number;
    eventId: number;
    mediaType: string;
    filename?: string;
}

/** An attachment with its bytes, as {@link EventStore.getAttachment} returns. */
export interface Attachment extends AttachmentMeta {
    data: Uint8Array;
}

/**
 * The canonical, validated shape we persist. `ts` is resolved (never undefined),
 * `meta` is the JSON text (or null), the rest are trimmed-or-null.
 */
interface NormalizedEvent {
    kind: string;
    content: string;
    role: string | null;
    metaText: string | null;
    session: string | null;
    correlation: string | null;
    ts: number;
}

/**
 * Validate and normalize raw input into the canonical shape we persist. Throws
 * {@link EventError} on anything we refuse to store. Mirrors memory's
 * `normalizeInput`: same defensive posture, same "bad data is the caller's
 * fault, surface it loudly" contract.
 */
function normalizeEventInput(input: EventInput): NormalizedEvent {
    if (input === null || typeof input !== "object") {
        throw new EventError("event input must be an object");
    }

    if (typeof input.kind !== "string") {
        throw new EventError("kind must be a string");
    }
    const kind = input.kind.trim();
    if (kind.length === 0) {
        throw new EventError("kind must not be empty");
    }

    if (typeof input.content !== "string") {
        throw new EventError("content must be a string");
    }
    if (input.content.trim().length === 0) {
        throw new EventError("content must not be empty");
    }
    if (input.content.length > MAX_CONTENT_LENGTH) {
        throw new EventError(`content exceeds ${MAX_CONTENT_LENGTH} characters`);
    }
    const content = input.content;

    const role = optionalString(input.role, "role");
    const session = optionalString(input.session, "session");
    const correlation = optionalString(input.correlation, "correlation");

    // meta is stored as JSON text. Reject anything that can't serialize (a
    // BigInt, a circular object, ...) at the door rather than discovering it
    // mid-INSERT. `undefined`/`null` means "no meta".
    let metaText: string | null = null;
    if (input.meta !== undefined && input.meta !== null) {
        try {
            const json = JSON.stringify(input.meta);
            // JSON.stringify(undefined) is undefined, and a value that serializes
            // to undefined (e.g. a lone function) carries nothing; treat as none.
            metaText = json ?? null;
        } catch (err) {
            throw new EventError(
                `meta must be JSON-serializable: ${err instanceof Error ? err.message : String(err)}`,
            );
        }
    }

    let ts = input.ts;
    if (ts === undefined) {
        ts = Date.now();
    } else if (typeof ts !== "number" || !Number.isFinite(ts)) {
        throw new EventError("ts must be a finite number");
    }

    return { kind, content, role, metaText, session, correlation, ts };
}

/** Validate an optional string field: undefined/null pass through as null, a
 *  non-string is rejected, a string is kept verbatim (callers may want
 *  whitespace-significant ids, so we do not trim here). */
function optionalString(value: unknown, field: string): string | null {
    if (value === undefined || value === null) return null;
    if (typeof value !== "string") {
        throw new EventError(`${field} must be a string`);
    }
    return value;
}

/** The image MIME types an attachment may carry. Narrow on purpose: only what
 *  the provider bridge accepts (mirrors the core {@link ImagePart}). */
const ATTACHMENT_MEDIA_TYPES: ReadonlySet<string> = new Set(["image/jpeg", "image/png"]);

/** Hard ceiling on one attachment's byte length, so a runaway upload can't bloat
 *  the database. Generous enough for a phone photo, well under what the wire
 *  caps allow once base64-expanded. */
export const MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024;

/** Validate an attachment before it reaches the database. Throws {@link
 *  EventError} on anything we refuse to store (mirrors normalizeEventInput's
 *  loud-on-bad-data posture). Returns the bytes as a Buffer node:sqlite accepts. */
function normalizeAttachmentInput(input: AttachmentInput): {
    mediaType: string;
    filename: string | null;
    data: Uint8Array;
} {
    if (input === null || typeof input !== "object") {
        throw new EventError("attachment input must be an object");
    }
    if (typeof input.mediaType !== "string" || !ATTACHMENT_MEDIA_TYPES.has(input.mediaType)) {
        throw new EventError("attachment mediaType must be 'image/jpeg' or 'image/png'");
    }
    if (!(input.data instanceof Uint8Array)) {
        throw new EventError("attachment data must be a Uint8Array");
    }
    if (input.data.length === 0) {
        throw new EventError("attachment data must not be empty");
    }
    if (input.data.length > MAX_ATTACHMENT_BYTES) {
        throw new EventError(`attachment exceeds ${MAX_ATTACHMENT_BYTES} bytes`);
    }
    const filename = optionalString(input.filename, "filename");
    return { mediaType: input.mediaType, filename, data: input.data };
}

interface EventRow {
    id: number;
    ts: number;
    kind: string;
    role: string | null;
    content: string;
    meta: string | null;
    session: string | null;
    correlation: string | null;
}

interface AttachmentRow {
    id: number;
    event_id: number;
    media_type: string;
    filename: string | null;
    data: Uint8Array;
}

/**
 * Reconstruct an {@link Event} from a db row. Tolerant of a corrupt `meta`
 * payload: a row whose meta doesn't parse as JSON degrades to `undefined`
 * rather than throwing and taking down a whole query (mirrors memory's
 * tolerant tag parse).
 */
function rowToEvent(row: EventRow): Event {
    return new Event({
        id: row.id,
        ts: row.ts,
        kind: row.kind,
        role: row.role ?? undefined,
        content: row.content,
        meta: parseMeta(row.meta),
        session: row.session ?? undefined,
        correlation: row.correlation ?? undefined,
    });
}

function parseMeta(raw: string | null): unknown {
    if (raw === null) return undefined;
    try {
        return JSON.parse(raw);
    } catch {
        return undefined;
    }
}

/**
 * SQLite-backed append-only event log.
 *
 * Shares one database file (and one schema `user_version`) with
 * {@link MemoryStore}: both call the same {@link migrate}, so opening either
 * store brings the whole schema current. Construct one, append to it, query it,
 * and {@link close} it. File-backed stores run in WAL mode by default so many
 * readers and one writer work concurrently.
 *
 * The public surface is append + read + selective-embedding only. There is no
 * content UPDATE and no event DELETE by contract; the `_vec`/`_fts` UPDATE/DELETE
 * triggers exist for integrity, not as an exposed mutation path.
 */
export class EventStore {
    private readonly db: DatabaseSync;
    private readonly walEnabled: boolean;
    private readonly schemaVersion: number;
    private readonly insertStmt;
    private readonly getStmt;
    private readonly upsertVecStmt;
    private readonly deleteVecStmt;
    private readonly getVecStmt;
    private readonly missingVecStmt;
    private readonly insertAttachmentStmt;
    private readonly attachmentsForStmt;
    private readonly getAttachmentStmt;
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

        // Honor the event_vec → events ON DELETE CASCADE (SQLite leaves foreign
        // keys off per connection by default). The explicit delete trigger is a
        // belt-and-braces backup, but the cascade is the contract.
        this.db.exec(`PRAGMA foreign_keys = ON`);

        if (wantWal) {
            const row = this.db.prepare(`PRAGMA journal_mode = WAL`).get() as {
                journal_mode?: string;
            };
            this.walEnabled = row?.journal_mode?.toLowerCase() === "wal";
            if (this.walEnabled) {
                this.db.exec(`PRAGMA synchronous = NORMAL`);
            }
        } else {
            this.walEnabled = false;
        }

        // The same one migration runner MemoryStore uses: there is a single
        // user_version per file, shared by both stores.
        this.schemaVersion = migrate(this.db);

        this.insertStmt = this.db.prepare(
            `INSERT INTO events (ts, kind, role, content, meta, session, correlation)
             VALUES (?, ?, ?, ?, ?, ?, ?)`,
        );
        this.getStmt = this.db.prepare(`SELECT * FROM events WHERE id = ?`);

        this.upsertVecStmt = this.db.prepare(
            `INSERT INTO event_vec (rowid, dim, vec) VALUES (?, ?, ?)
             ON CONFLICT(rowid) DO UPDATE SET dim = excluded.dim, vec = excluded.vec`,
        );
        this.deleteVecStmt = this.db.prepare(`DELETE FROM event_vec WHERE rowid = ?`);
        this.getVecStmt = this.db.prepare(`SELECT vec FROM event_vec WHERE rowid = ?`);
        // Events with no embedding yet: the backfill work-list. Newest first so
        // recent events become semantically searchable soonest.
        this.missingVecStmt = this.db.prepare(
            `SELECT e.id FROM events e
             LEFT JOIN event_vec v ON v.rowid = e.id
             WHERE v.rowid IS NULL
             ORDER BY e.ts DESC, e.id DESC
             LIMIT ?`,
        );

        this.insertAttachmentStmt = this.db.prepare(
            `INSERT INTO event_attachments (event_id, media_type, filename, data)
             VALUES (?, ?, ?, ?)`,
        );
        // Metadata only (no `data`): listing an event's images mustn't pull their
        // bytes. Oldest-first by id so they render in the order they were attached.
        this.attachmentsForStmt = this.db.prepare(
            `SELECT id, event_id, media_type, filename FROM event_attachments
             WHERE event_id = ? ORDER BY id ASC`,
        );
        this.getAttachmentStmt = this.db.prepare(`SELECT * FROM event_attachments WHERE id = ?`);
    }

    private assertOpen() {
        if (this.closed) throw new EventError("store is closed");
    }

    /** Insert one event, returning it with its assigned id. Append-only. */
    append(input: EventInput): Event {
        this.assertOpen();
        const norm = normalizeEventInput(input);
        const result = this.insertStmt.run(
            norm.ts,
            norm.kind,
            norm.role,
            norm.content,
            norm.metaText,
            norm.session,
            norm.correlation,
        );
        return new Event({
            id: Number(result.lastInsertRowid),
            ts: norm.ts,
            kind: norm.kind,
            role: norm.role ?? undefined,
            content: norm.content,
            meta: parseMeta(norm.metaText),
            session: norm.session ?? undefined,
            correlation: norm.correlation ?? undefined,
        });
    }

    /**
     * Append a batch in one transaction. All-or-nothing: each input is validated
     * and inserted within the transaction, so a single bad input rolls the whole
     * batch back and inserts nothing. Returns the events in input order, each
     * with its assigned id.
     */
    appendMany(inputs: EventInput[]): Event[] {
        this.assertOpen();
        if (!Array.isArray(inputs)) {
            throw new EventError("appendMany expects an array of event inputs");
        }
        if (inputs.length === 0) return [];

        const out: Event[] = [];
        this.db.exec("BEGIN");
        try {
            for (const input of inputs) {
                // Validate-then-insert inside the txn so a bad input (which throws
                // here) rolls back everything appended so far in this batch.
                const norm = normalizeEventInput(input);
                const result = this.insertStmt.run(
                    norm.ts,
                    norm.kind,
                    norm.role,
                    norm.content,
                    norm.metaText,
                    norm.session,
                    norm.correlation,
                );
                out.push(
                    new Event({
                        id: Number(result.lastInsertRowid),
                        ts: norm.ts,
                        kind: norm.kind,
                        role: norm.role ?? undefined,
                        content: norm.content,
                        meta: parseMeta(norm.metaText),
                        session: norm.session ?? undefined,
                        correlation: norm.correlation ?? undefined,
                    }),
                );
            }
            this.db.exec("COMMIT");
        } catch (err) {
            this.db.exec("ROLLBACK");
            throw err;
        }
        return out;
    }

    /** Fetch one event by id, or undefined if no such event exists. */
    get(id: number): Event | undefined {
        this.assertOpen();
        const row = this.getStmt.get(id) as EventRow | undefined;
        return row ? rowToEvent(row) : undefined;
    }

    /**
     * The episodic-replay read: events newest first (ts DESC, id DESC tiebreak so
     * events sharing a timestamp still order by insertion). Supports limit,
     * offset, and the kind/session/since/until filters. Defaults to
     * {@link DEFAULT_LIMIT} rows so a growing log never returns an unbounded set.
     */
    recent(opts: EventQuery = {}): Event[] {
        this.assertOpen();
        const { whereSql, params } = this.buildFilters(opts);
        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));

        const sql =
            `SELECT * FROM events ${whereSql} ` + `ORDER BY ts DESC, id DESC ` + `LIMIT ? OFFSET ?`;
        params.push(limit, offset);

        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as EventRow[];
        return rows.map(rowToEvent);
    }

    /**
     * Rank events by lexical relevance to `text` using the FTS5 index (bm25),
     * with the same kind/session/since/until filters AND-ed in. Matching is
     * token-based (see {@link toFtsQuery}): the query is reduced to word tokens
     * and OR-matched, so a whole sentence still finds events sharing any term. A
     * query with no usable tokens (or no FTS hits) yields an empty array.
     */
    searchRelevant(text: string, opts: EventQuery = {}): Event[] {
        this.assertOpen();
        const match = toFtsQuery(text);
        if (match === null) return [];

        const { conditions, params } = this.filterConditions(opts, "e");
        const where = ["event_fts MATCH ?", ...conditions];
        const matchParams: unknown[] = [match, ...params];

        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));

        // bm25() is ascending (more-negative = better), so order by it directly,
        // then newest first to break ties among comparably-relevant events.
        const sql =
            `SELECT e.* FROM event_fts ` +
            `JOIN events e ON e.id = event_fts.rowid ` +
            `WHERE ${where.join(" AND ")} ` +
            `ORDER BY bm25(event_fts), e.ts DESC, e.id DESC ` +
            `LIMIT ? OFFSET ?`;
        matchParams.push(limit, offset);

        const rows = this.db.prepare(sql).all(...(matchParams as never[])) as unknown as EventRow[];
        return rows.map(rowToEvent);
    }

    /**
     * Rank events by semantic similarity to a query vector (cosine), highest
     * first, with the same filters applied. The meaning-based counterpart to
     * {@link searchRelevant}'s lexical match. Events without an embedding are
     * invisible here (embed them via {@link setEmbedding} / a backfill). A query
     * vector whose dimension differs from a stored one scores 0 for that event
     * and drops out.
     *
     * Implementation note: this scans every stored event vector and ranks in JS.
     * Because the vector index is selective (most events never get a vector),
     * that set stays small; a corpus that outgrows a linear scan would want an
     * ANN index, a future migration, not a change to this signature.
     */
    semanticSearch(query: Float32Array, opts: EventQuery = {}): EventSemanticHit[] {
        this.assertOpen();
        const { conditions, params } = this.filterConditions(opts, "e");
        const whereSql = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

        const sql =
            `SELECT e.*, v.vec AS vec FROM event_vec v ` +
            `JOIN events e ON e.id = v.rowid ${whereSql}`;

        const rows = this.db.prepare(sql).all(...(params as never[])) as unknown as Array<
            EventRow & { vec: Uint8Array }
        >;

        const scored: EventSemanticHit[] = [];
        for (const row of rows) {
            const score = cosineSimilarity(query, blobToVector(row.vec));
            scored.push({ event: rowToEvent(row), score });
        }
        // Highest similarity first; newest event breaks ties among equally-
        // similar events (mirrors the lexical path's secondary sort).
        scored.sort(
            (a, b) => b.score - a.score || b.event.ts - a.event.ts || b.event.id - a.event.id,
        );

        const limit = clampLimit(opts.limit);
        const offset = Math.max(0, Math.floor(opts.offset ?? 0));
        return scored.slice(offset, offset + limit);
    }

    // ── Embeddings (selective) ────────────────────────────────────────────────
    //
    // The store owns storage and comparison of embeddings, never their
    // production. Embedding is opt-in per event: the log is total, the vector
    // index is not.

    /**
     * Store (or replace) the embedding for an event. Returns false if no event
     * with that id exists: we won't keep an orphan vector. The vector is expected
     * to be L2-normalized (as {@link Embedder} guarantees) so semantic ranking
     * can use a plain dot product.
     */
    setEmbedding(id: number, vector: Float32Array): boolean {
        this.assertOpen();
        if (!this.getStmt.get(id)) return false;
        this.upsertVecStmt.run(id, vector.length, vectorToBlob(vector));
        return true;
    }

    /** Drop an event's embedding, e.g. before re-embedding. Returns whether a
     *  row was removed. (Deleting the event does this too, via cascade/trigger.) */
    deleteEmbedding(id: number): boolean {
        this.assertOpen();
        return this.deleteVecStmt.run(id).changes > 0;
    }

    /** Whether an event currently has a stored embedding. */
    hasEmbedding(id: number): boolean {
        this.assertOpen();
        return this.getVecStmt.get(id) !== undefined;
    }

    /**
     * Ids of events with no embedding yet: the backfill work-list. Newest first;
     * bounded by `limit` (default {@link DEFAULT_LIMIT}). Drive a backfill by
     * embedding these and calling {@link setEmbedding} for each.
     */
    idsMissingEmbedding(limit = DEFAULT_LIMIT): number[] {
        this.assertOpen();
        const rows = this.missingVecStmt.all(clampLimit(limit)) as Array<{ id: number }>;
        return rows.map((r) => r.id);
    }

    // ── Attachments (selective) ───────────────────────────────────────────────
    //
    // Image bytes live in a side table, not in `content` (which is text, FTS-
    // indexed, and capped at MAX_CONTENT_LENGTH) and not in `meta` (which is
    // parsed on every read). An attaching event keeps a `[image: name]`
    // placeholder in its text; the bytes are fetched only by an explicit call.

    /**
     * Attach one image to an existing event, returning its assigned id. Throws
     * {@link EventError} if the event doesn't exist (no orphan attachments) or the
     * input fails validation. The bytes are stored as a BLOB (raw, not base64).
     */
    appendAttachment(eventId: number, input: AttachmentInput): number {
        this.assertOpen();
        if (!this.getStmt.get(eventId)) {
            throw new EventError(`cannot attach to unknown event ${eventId}`);
        }
        const norm = normalizeAttachmentInput(input);
        const result = this.insertAttachmentStmt.run(
            eventId,
            norm.mediaType,
            norm.filename,
            // node:sqlite binds a Buffer/Uint8Array straight to a BLOB param.
            norm.data,
        );
        return Number(result.lastInsertRowid);
    }

    /** List an event's attachments as metadata only (no bytes), oldest first. The
     *  cheap read a replay uses to know which images to request and how to label
     *  them; the actual bytes come from {@link getAttachment} per id. */
    attachmentsFor(eventId: number): AttachmentMeta[] {
        this.assertOpen();
        const rows = this.attachmentsForStmt.all(eventId) as unknown as Omit<
            AttachmentRow,
            "data"
        >[];
        return rows.map((r) => ({
            id: r.id,
            eventId: r.event_id,
            mediaType: r.media_type,
            filename: r.filename ?? undefined,
        }));
    }

    /** Fetch one attachment with its bytes, or undefined if no such attachment
     *  exists. The bytes-bearing read, served on demand at the wire boundary. */
    getAttachment(id: number): Attachment | undefined {
        this.assertOpen();
        const row = this.getAttachmentStmt.get(id) as AttachmentRow | undefined;
        if (!row) return undefined;
        return {
            id: row.id,
            eventId: row.event_id,
            mediaType: row.media_type,
            filename: row.filename ?? undefined,
            // node:sqlite hands back a Uint8Array for a BLOB column.
            data: row.data,
        };
    }

    /** Total number of events, or the count matching the given filters. */
    count(opts: EventQuery = {}): number {
        this.assertOpen();
        const { whereSql, params } = this.buildFilters(opts);
        const sql = `SELECT COUNT(*) AS n FROM events ${whereSql}`;
        const row = this.db.prepare(sql).get(...(params as never[])) as { n: number };
        return row.n;
    }

    /**
     * Build the shared kind/session/since/until WHERE conditions for a query on
     * the base `events` table (optionally under alias `alias`). Returns the
     * condition fragments and their bound params, in lockstep.
     */
    private filterConditions(
        opts: EventQuery,
        alias?: string,
    ): { conditions: string[]; params: unknown[] } {
        const col = (name: string) => (alias ? `${alias}.${name}` : name);
        const conditions: string[] = [];
        const params: unknown[] = [];

        if (opts.kind !== undefined) {
            conditions.push(`${col("kind")} = ?`);
            params.push(opts.kind);
        }
        if (opts.session !== undefined) {
            conditions.push(`${col("session")} = ?`);
            params.push(opts.session);
        }
        if (opts.since !== undefined) {
            if (typeof opts.since !== "number" || !Number.isFinite(opts.since)) {
                throw new EventError("since must be a finite number");
            }
            conditions.push(`${col("ts")} >= ?`);
            params.push(opts.since);
        }
        if (opts.until !== undefined) {
            if (typeof opts.until !== "number" || !Number.isFinite(opts.until)) {
                throw new EventError("until must be a finite number");
            }
            conditions.push(`${col("ts")} <= ?`);
            params.push(opts.until);
        }
        return { conditions, params };
    }

    /** filterConditions wrapped into a ready-to-interpolate WHERE clause for
     *  queries over the bare `events` table (no alias). */
    private buildFilters(opts: EventQuery): { whereSql: string; params: unknown[] } {
        const { conditions, params } = this.filterConditions(opts);
        return {
            whereSql: conditions.length ? `WHERE ${conditions.join(" AND ")}` : "",
            params,
        };
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
