/**
 * Bridges the knowledge base into the agentic loop.
 *
 * Mirrors {@link memoryTools}, but for the {@link NotesStore}/{@link NotesService}
 * corpus. The crucial difference in posture: notes are documentation the agent
 * *opts into* reading, not facts auto-injected every turn. There is therefore no
 * `recallContext`-style passive injection here by default; the agent reaches for
 * a note with `note_recall` when it judges the human's docs relevant, which keeps
 * long-form human writing out of every turn's context window.
 *
 * Writes (`note_save`, `note_update`, `note_link`) go through the
 * {@link NotesService}, so an agent's note edit lands a DB row *and* a markdown
 * file on disk, converging to the same state a human's file save would, the
 * single unified write path. Reads (`note_recall`) go straight to the store.
 *
 * Every tool speaks plain JSON in and out (its `run` result drops straight into a
 * `tool_result` part), and translates {@link NoteError} into a clean message the
 * model can read rather than letting an opaque error surface.
 */

import type { ToolDef } from "./types.ts";
import { Note, NoteError, NotesStore, type NoteLink } from "./notes.ts";
import { NotesService } from "./notesService.ts";
import { embedOne, EmbeddingError, type Embedder } from "./embeddings.ts";

/** How many notes `note_recall` returns by default. */
export const DEFAULT_NOTE_RECALL_LIMIT = 8;

/** The serializable view of a note handed back to the model. The body is
 *  included (the agent asked for it), but capped so a single huge note can't
 *  blow the turn's budget; the cap is generous and only trims pathological
 *  notes. */
export interface NoteView {
    id: number;
    uuid: string;
    path: string;
    title: string;
    content: string;
    frontmatter: Note["frontmatter"];
    created: number;
    updated: number;
}

/** Per-note body cap in the recall view, so a recall of several notes stays
 *  bounded. A note longer than this is truncated with an ellipsis marker; the
 *  agent can open the full note via its path/uuid if it needs the rest. */
const RECALL_BODY_CAP = 4_000;

function toView(n: Note, capBody = false): NoteView {
    let content = n.content;
    if (capBody && content.length > RECALL_BODY_CAP) {
        content = content.slice(0, RECALL_BODY_CAP) + "\n…[truncated]";
    }
    return {
        id: n.id,
        uuid: n.uuid,
        path: n.path,
        title: n.title,
        content,
        frontmatter: n.frontmatter,
        created: n.created,
        updated: n.updated,
    };
}

/** Narrow an unknown args bag to a record without trusting its fields yet. */
function asRecord(args: unknown): Record<string, unknown> {
    return typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
}

/**
 * The shared note recall ranking, the notes counterpart to memory's. Order of
 * preference: semantic (cosine) when an embedder is configured and the query
 * embeds, then lexical (FTS/bm25), then recency. Each step falls through so
 * recall is never worse than lexical.
 */
async function recallNotes(
    store: NotesStore,
    embedder: Embedder | undefined,
    query: string,
    opts: { limit: number; pathPrefix?: string },
): Promise<Note[]> {
    const trimmed = query.trim();

    if (trimmed && embedder) {
        try {
            const vec = await embedOne(embedder, trimmed);
            const hits = store.semanticSearch(vec, opts).map((h) => h.note);
            if (hits.length) return hits;
        } catch (err) {
            if (!(err instanceof EmbeddingError)) throw err;
        }
    }

    if (trimmed) {
        const lexical = store.searchRelevant(trimmed, opts);
        if (lexical.length) return lexical;
        const substring = store.search(trimmed, opts);
        if (substring.length) return substring;
    }

    return store.all(opts);
}

/**
 * Build the note tool set over a service (writes) and its store (reads).
 *
 * Pass an {@link Embedder} to enable semantic recall (best-effort: an embedding
 * outage degrades recall to lexical, never failing a tool call). The tools are
 * async because a write touches the filesystem through the {@link NotesService}.
 */
export function noteTools(
    service: NotesService,
    store: NotesStore,
    embedder?: Embedder,
): ToolDef[] {
    const save: ToolDef = {
        name: "note_save",
        description:
            "Create a knowledge-base note: longer-form documentation kept as a " +
            "markdown file the human can also edit. Use for durable docs, runbooks, " +
            "and references, not transient facts (use memory_save for those). " +
            "Returns the created note including its path and uuid.",
        parameters: {
            type: "object",
            properties: {
                title: { type: "string", description: "The note's title." },
                content: { type: "string", description: "The markdown body." },
                path: {
                    type: "string",
                    description:
                        "Optional relative path within the KB folder (e.g. 'ops/deploy.md'). " +
                        "Derived from the title when omitted. Must end in .md.",
                },
                tags: {
                    type: "array",
                    items: { type: "string" },
                    description: "Optional tags, stored in the note's frontmatter.",
                },
            },
            required: ["title", "content"],
        },
        async run(args) {
            const a = asRecord(args);
            try {
                const frontmatter = Array.isArray(a.tags)
                    ? { tags: a.tags.filter((t): t is string => typeof t === "string") }
                    : undefined;
                const result = await service.create({
                    title: a.title as string,
                    content: typeof a.content === "string" ? a.content : "",
                    path: typeof a.path === "string" ? a.path : undefined,
                    frontmatter,
                });
                return { saved: true, note: toView(result.note) };
            } catch (err) {
                if (err instanceof NoteError) return { saved: false, error: err.message };
                throw err;
            }
        },
    };

    const update: ToolDef = {
        name: "note_update",
        description:
            "Update an existing knowledge-base note by its uuid. Only the fields you " +
            "pass change; omitted fields are left as-is. Returns the updated note.",
        parameters: {
            type: "object",
            properties: {
                uuid: { type: "string", description: "The uuid of the note to update." },
                title: { type: "string", description: "New title." },
                content: { type: "string", description: "New markdown body." },
                path: { type: "string", description: "New relative path (moves the file)." },
            },
            required: ["uuid"],
        },
        async run(args) {
            const a = asRecord(args);
            if (typeof a.uuid !== "string") {
                return { updated: false, error: "uuid must be a string" };
            }
            const existing = store.getByUuid(a.uuid);
            if (!existing) return { updated: false, error: "no note with that uuid" };
            try {
                const patch: Parameters<NotesService["update"]>[1] = {};
                if (typeof a.title === "string") patch.title = a.title;
                if (typeof a.content === "string") patch.content = a.content;
                if (typeof a.path === "string") patch.path = a.path;
                const result = await service.update(existing.id, patch);
                if (!result) return { updated: false, error: "note vanished" };
                return { updated: true, note: toView(result.note) };
            } catch (err) {
                if (err instanceof NoteError) return { updated: false, error: err.message };
                throw err;
            }
        },
    };

    const recall: ToolDef = {
        name: "note_recall",
        description:
            "Search the knowledge base. Returns the most relevant notes first " +
            "(by meaning when embeddings are available, else by shared words). " +
            "Omit `query` to list recent notes; filter to a folder with `prefix`.",
        parameters: {
            type: "object",
            properties: {
                query: { type: "string", description: "What to search for." },
                prefix: {
                    type: "string",
                    description: "Optional folder prefix to scope the search (e.g. 'ops/').",
                },
                limit: { type: "number", description: "Max results (default 8)." },
            },
        },
        async run(args) {
            const a = asRecord(args);
            const query = typeof a.query === "string" ? a.query : "";
            const limit = typeof a.limit === "number" ? a.limit : DEFAULT_NOTE_RECALL_LIMIT;
            const pathPrefix = typeof a.prefix === "string" ? a.prefix : undefined;
            const hits = await recallNotes(store, embedder, query, { limit, pathPrefix });
            return { count: hits.length, notes: hits.map((n) => toView(n, true)) };
        },
    };

    const forget: ToolDef = {
        name: "note_forget",
        description:
            "Delete a knowledge-base note by its uuid: removes the row, its " +
            "embedding, any links touching it, and the markdown file on disk. " +
            "Symmetric with memory_forget. Returns whether a note was removed.",
        parameters: {
            type: "object",
            properties: {
                uuid: { type: "string", description: "The uuid of the note to delete." },
            },
            required: ["uuid"],
        },
        async run(args) {
            const a = asRecord(args);
            if (typeof a.uuid !== "string") {
                return { forgotten: false, error: "uuid must be a string" };
            }
            const existing = store.getByUuid(a.uuid);
            if (!existing) return { forgotten: false, error: "no note with that uuid" };
            try {
                // Go through the service so the file is deleted alongside the row
                // (the store's cascade drops the vector and links; the file lives
                // outside the DB and only the service knows the KB root).
                const removed = await service.remove(existing.id);
                return { forgotten: removed };
            } catch (err) {
                if (err instanceof NoteError) return { forgotten: false, error: err.message };
                throw err;
            }
        },
    };

    const links: ToolDef = {
        name: "note_links",
        description:
            "Read a note's links so the graph is traversable, not write-only. By " +
            "default returns the note's outgoing links (what it references); pass " +
            "direction:'in' for the reverse (other notes that reference this one). " +
            "Each link reports its id (for note_unlink), kind, and target.",
        parameters: {
            type: "object",
            properties: {
                uuid: { type: "string", description: "The uuid of the note to inspect." },
                direction: {
                    type: "string",
                    enum: ["out", "in"],
                    description:
                        "'out' (default): links this note makes. 'in': links pointing " +
                        "at this note from other notes.",
                },
            },
            required: ["uuid"],
        },
        async run(args) {
            const a = asRecord(args);
            if (typeof a.uuid !== "string") {
                return { error: "uuid must be a string" };
            }
            const note = store.getByUuid(a.uuid);
            if (!note) return { error: "no note with that uuid" };
            const direction = a.direction === "in" ? "in" : "out";
            const raw = direction === "in" ? store.linksToNote(note.id) : store.linksFrom(note.id);
            const links = raw.map((l) => toLinkView(store, l));
            return { count: links.length, direction, links };
        },
    };

    const unlink: ToolDef = {
        name: "note_unlink",
        description:
            "Remove a single link by its id (get ids from note_links). Returns " +
            "whether a link was removed. Deletes only the edge, never the notes or " +
            "memory it connected.",
        parameters: {
            type: "object",
            properties: {
                id: { type: "number", description: "The id of the link to remove." },
            },
            required: ["id"],
        },
        async run(args) {
            const a = asRecord(args);
            if (typeof a.id !== "number" || !Number.isFinite(a.id)) {
                return { unlinked: false, error: "id must be a finite number" };
            }
            return { unlinked: store.unlink(a.id) };
        },
    };

    const link: ToolDef = {
        name: "note_link",
        description:
            "Record a relation from one note to a memory or another note (e.g. " +
            "'references', 'derived_from'). Identify the source note and the target " +
            "by uuid (note) or id (memory). One target only.",
        parameters: {
            type: "object",
            properties: {
                from: { type: "string", description: "The uuid of the source note." },
                toNote: { type: "string", description: "The uuid of a target note." },
                toMemory: { type: "number", description: "The id of a target memory." },
                kind: {
                    type: "string",
                    description: "Optional relation label, e.g. 'references' or 'derived_from'.",
                },
            },
            required: ["from"],
        },
        async run(args) {
            const a = asRecord(args);
            if (typeof a.from !== "string") {
                return { linked: false, error: "from must be a note uuid" };
            }
            const from = store.getByUuid(a.from);
            if (!from) return { linked: false, error: "no source note with that uuid" };

            const hasNote = typeof a.toNote === "string";
            const hasMemory = typeof a.toMemory === "number" && Number.isFinite(a.toMemory);
            if (hasNote === hasMemory) {
                return { linked: false, error: "specify exactly one of toNote or toMemory" };
            }
            try {
                if (hasNote) {
                    const target = store.getByUuid(a.toNote as string);
                    if (!target) return { linked: false, error: "no target note with that uuid" };
                    const l = store.link(from.id, { toNote: target.id }, asKind(a.kind));
                    return { linked: true, link: { id: l.id, kind: l.kind } };
                }
                const l = store.link(from.id, { toMemory: a.toMemory as number }, asKind(a.kind));
                return { linked: true, link: { id: l.id, kind: l.kind } };
            } catch (err) {
                if (err instanceof NoteError) return { linked: false, error: err.message };
                throw err;
            }
        },
    };

    return [save, update, recall, link, forget, links, unlink];
}

function asKind(v: unknown): string | undefined {
    return typeof v === "string" && v.trim() ? v : undefined;
}

/** The serializable view of a link handed back to the model. We resolve the
 *  target into the same handles the agent uses elsewhere: a note target as its
 *  uuid (its stable id), a memory target as its numeric id. A null `toUuid`/
 *  `toMemory` means the target was forgotten (a memory link survives its target's
 *  deletion via SET NULL); we surface that rather than hiding the dangling edge. */
function toLinkView(store: NotesStore, link: NoteLink) {
    const target =
        link.toNote !== null
            ? { kind: "note" as const, toUuid: store.get(link.toNote)?.uuid ?? null }
            : { kind: "memory" as const, toMemory: link.toMemory };
    return { id: link.id, relation: link.kind, ...target };
}
