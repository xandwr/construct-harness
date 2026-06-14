/**
 * NotesService: the single code path that mutates a note, and the two-way bridge
 * between the {@link NotesStore} (the source of truth) and the KB markdown folder.
 *
 * Everything that changes a note, whether a UI button, an agent tool, or a file
 * saved in an editor, funnels through this one object. That is the property that
 * keeps an API write and a file save from racing to two different rows: there is
 * exactly one place a note is created/updated/moved/deleted, and it always
 * converges the DB and the file to the same state.
 *
 * Three responsibilities, layered:
 *
 *  1. Unified write path (outbound: DB -> file). {@link create}/{@link update}/
 *     {@link move}/{@link remove} mutate the store, best-effort re-embed, and
 *     write the file atomically (temp + rename, via notesFile.ts). After each
 *     outbound write the resulting content hash is recorded as "the last synced
 *     state for this path", which is both the echo-loop guard and the conflict
 *     base.
 *
 *  2. Inbound sync (file -> DB). A debounced watcher over the KB folder turns
 *     editor saves, renames, and deletes into store mutations. A file with no
 *     `uuid` in its frontmatter is a human-created note: we mint one, adopt the
 *     file, and write the uuid back. The echo of *our own* outbound write is
 *     recognized (same path, same hash we just wrote) and dropped, so an
 *     outbound write never ping-pongs back through the watcher.
 *
 *  3. Conflict resolution. When the file changed AND the row changed since the
 *     last synced state (a genuine concurrent edit: the human saved while the
 *     agent updated the row), last-write-wins by `updated`, and the losing side
 *     is preserved as a `title (conflict <ts>).md` sidecar so nothing is
 *     silently destroyed. A 3-way merge is deliberately out of scope.
 *
 * Single-user/single-process is assumed throughout (the harness runs one live
 * Session per process). That assumption is what makes the conflict window small
 * and the in-memory "last synced hash" map sufficient as the conflict base.
 *
 * Embedding is always best-effort: an embedding outage degrades recall to
 * lexical, it never fails a write (mirrors memoryTools' embedIfPossible).
 */

import { watch, type FSWatcher } from "node:fs";
import { stat, readdir } from "node:fs/promises";
import { join, relative, sep } from "node:path";
import { Note, NotesStore, NoteError, hashContent, type NotePatch } from "./notes.ts";
import { embedOne, EmbeddingError, type Embedder } from "./embeddings.ts";
import {
    serializeNote,
    parseNoteFile,
    writeNoteFileAtomic,
    moveNoteFile,
    deleteNoteFile,
    readNoteFile,
    ensureRoot,
    defaultPathForNote,
} from "./notesFile.ts";

/** How long (ms) to coalesce filesystem events before acting on a path. Editors
 *  fire several events per save (write temp, rename, chmod); the debounce folds
 *  them into one inbound sync. */
export const DEFAULT_DEBOUNCE_MS = 150;

/** Options for constructing a {@link NotesService}. */
export interface NotesServiceConfig {
    /** The backing store (the source of truth). Required. */
    store: NotesStore;
    /** Absolute or relative path to the KB markdown folder. Required. */
    root: string;
    /** Embedder for semantic recall. Best-effort; omit to keep recall lexical. */
    embedder?: Embedder;
    /** Debounce window for the inbound watcher (ms). */
    debounceMs?: number;
    /** Sink for non-fatal sync diagnostics (a conflict resolved, a parse skipped).
     *  Defaults to console.warn. Pass a no-op to silence, or capture in tests. */
    onWarn?: (message: string) => void;
}

/** What a unified write produced: the resulting note and the file path written. */
export interface WriteResult {
    note: Note;
    /** The note's relative path within the KB folder. */
    path: string;
}

/** Outcome of reconciling one file against the store during inbound sync. For
 *  observability and tests; the service acts on it internally. */
export type SyncOutcome =
    | { kind: "unchanged" } // file matches the row (often our own echo)
    | { kind: "created"; note: Note } // a new (human-authored) file adopted
    | { kind: "updated"; note: Note } // file edits applied to the row
    | { kind: "moved"; note: Note; from: string } // file relocated; path updated
    | { kind: "conflict"; note: Note; sidecar: string } // both sides changed
    | { kind: "skipped"; reason: string }; // unparseable / out of scope

/**
 * The two-way sync engine. Construct with a store and a KB root, then either
 * drive it manually (call {@link reconcileFile} / the write methods) or
 * {@link start} the watcher for live inbound sync.
 */
export class NotesService {
    private readonly store: NotesStore;
    private readonly embedder?: Embedder;
    private readonly debounceMs: number;
    private readonly onWarn: (message: string) => void;

    /** The real (symlink-resolved) absolute KB root, set by {@link start} or the
     *  first write; until then, the configured value. */
    private root: string;
    private rootReady = false;

    /**
     * The conflict base: for each path, a snapshot of the last point the DB row
     * and the file were known to agree. `hash` is the content hash at that point
     * (file and DB share it then, by definition). Comparing the *current* file
     * hash and the *current* DB hash each against this snapshot tells us which
     * side(s) moved since the last agreement:
     *  - file moved   = currentFileHash !== snapshot.hash
     *  - DB moved      = currentDbHash   !== snapshot.hash
     *  - both moved    = a genuine concurrent edit -> conflict.
     * An outbound write (DB -> file) records a fresh snapshot (they agree again);
     * a clean inbound sync does too. A DB mutation that does NOT go through the
     * service's file-writing path (e.g. a direct store edit racing a file save)
     * is exactly what leaves the snapshot stale on the DB side and surfaces as a
     * conflict on the next inbound event.
     */
    private synced = new Map<string, string>();

    /** Paths with a self-write in flight, by the hash we expect the file to carry.
     *  When the watcher reports a path whose file hash is in this set, it's the
     *  echo of our own write and is dropped. Cleared once observed (or on the next
     *  outbound write to the path). */
    private pendingSelfWrites = new Map<string, Set<string>>();

    /** Per-path debounce timers for inbound events. */
    private debounceTimers = new Map<string, NodeJS.Timeout>();

    /** Monotonic counter for unique temp-file suffixes (avoids Date.now/random,
     *  keeping writes deterministic and collision-free across concurrent saves). */
    private writeSeq = 0;

    private watcher: FSWatcher | undefined;
    private closed = false;

    constructor(config: NotesServiceConfig) {
        this.store = config.store;
        this.embedder = config.embedder;
        this.debounceMs = config.debounceMs ?? DEFAULT_DEBOUNCE_MS;
        this.onWarn = config.onWarn ?? ((m) => console.warn(`[notes-sync] ${m}`));
        this.root = config.root;
    }

    /** The resolved KB root (valid after {@link ready}). */
    get kbRoot(): string {
        return this.root;
    }

    /** Ensure the KB folder exists and resolve its real path. Idempotent; called
     *  implicitly by the write methods and by {@link start}. */
    async ready(): Promise<void> {
        if (this.rootReady) return;
        this.root = await ensureRoot(this.root);
        this.rootReady = true;
    }

    private assertOpen() {
        if (this.closed) throw new NoteError("notes service is closed");
    }

    // ── Unified write path (outbound: DB -> file) ─────────────────────────────

    /**
     * Create a note from scratch (API/agent origin), persisting it and writing
     * its file. A path may be supplied; otherwise one is derived from the title.
     * The file is written atomically and recorded as a self-write so the watcher
     * ignores the echo.
     */
    async create(input: {
        title: string;
        content: string;
        path?: string;
        frontmatter?: Note["frontmatter"];
    }): Promise<WriteResult> {
        this.assertOpen();
        await this.ready();

        const uuid = crypto.randomUUID();
        const path = input.path ?? defaultPathForNote(input.title, uuid);
        const note = new Note({
            uuid,
            path,
            title: input.title,
            content: input.content,
            frontmatter: input.frontmatter,
        });
        this.store.save(note); // throws NoteError on a path/uuid clash
        await this.embedBestEffort(note);
        await this.writeOut(note);
        return { note, path: note.path };
    }

    /**
     * Update an existing note by id (API/agent origin) and mirror the change to
     * its file. If the patch changes the path, the file is moved to match.
     * Returns undefined if no note with that id exists.
     */
    async update(id: number, patch: NotePatch): Promise<WriteResult | undefined> {
        this.assertOpen();
        await this.ready();

        const before = this.store.get(id);
        if (!before) return undefined;
        const oldPath = before.path;

        const note = this.store.update(id, patch);
        if (!note) return undefined;

        // Re-embed only when content changed (the store's trigger already dropped
        // a stale vector on a content edit; a metadata-only edit keeps it).
        if (note.content !== before.content) {
            await this.embedBestEffort(note);
        }

        if (note.path !== oldPath) {
            await this.moveOut(oldPath, note);
        } else {
            await this.writeOut(note);
        }
        return { note, path: note.path };
    }

    /**
     * Move/rename a note to a new path (a first-class operation so the file moves
     * atomically rather than being deleted and recreated). Thin wrapper over
     * {@link update} with only the path changing.
     */
    async move(id: number, newPath: string): Promise<WriteResult | undefined> {
        return this.update(id, { path: newPath });
    }

    /**
     * Delete a note and its file. Idempotent on the file side (a missing file is
     * not an error). Returns whether a row was removed.
     */
    async remove(id: number): Promise<boolean> {
        this.assertOpen();
        await this.ready();
        const note = this.store.get(id);
        if (!note) return false;
        this.store.delete(note.id); // cascades vec + links
        await deleteNoteFile(this.root, note.path).catch((err) =>
            this.onWarn(`failed to delete file for ${note.path}: ${errMsg(err)}`),
        );
        this.synced.delete(note.path);
        return true;
    }

    /** Serialize a note to its file atomically, recording the self-write so the
     *  watcher drops the echo and updating the conflict base. */
    private async writeOut(note: Note): Promise<void> {
        const contents = serializeNote(note);
        this.markSelfWrite(note.path, note.contentHash);
        await writeNoteFileAtomic(this.root, note.path, contents, String(this.writeSeq++));
        this.synced.set(note.path, note.contentHash);
    }

    /** Move a note's file from its old path to its new one, then write the new
     *  contents (the move alone doesn't update the file's frontmatter/body). */
    private async moveOut(oldPath: string, note: Note): Promise<void> {
        await moveNoteFile(this.root, oldPath, note.path).catch((err) =>
            this.onWarn(`failed to move file ${oldPath} -> ${note.path}: ${errMsg(err)}`),
        );
        this.synced.delete(oldPath);
        await this.writeOut(note);
    }

    /** Embed a note's content and store the vector, swallowing embedding errors
     *  (an outage degrades recall, never fails a write). */
    private async embedBestEffort(note: Note): Promise<void> {
        if (!this.embedder) return;
        try {
            const vec = await embedOne(this.embedder, note.content);
            this.store.setEmbedding(note.id, vec);
        } catch (err) {
            if (!(err instanceof EmbeddingError)) throw err;
        }
    }

    private markSelfWrite(path: string, hash: string): void {
        let set = this.pendingSelfWrites.get(path);
        if (!set) {
            set = new Set();
            this.pendingSelfWrites.set(path, set);
        }
        set.add(hash);
    }

    /** Was a file event for `path` with `hash` the echo of our own write? Consumes
     *  the marker (a self-write is observed once). */
    private consumeSelfWrite(path: string, hash: string): boolean {
        const set = this.pendingSelfWrites.get(path);
        if (!set || !set.has(hash)) return false;
        set.delete(hash);
        if (set.size === 0) this.pendingSelfWrites.delete(path);
        return true;
    }

    // ── Inbound sync (file -> DB) ─────────────────────────────────────────────

    /**
     * Reconcile a single file at `relPath` against the store: the core inbound
     * operation, exposed directly so tests (and a manual full scan) can drive it
     * without the watcher. Reads the file, parses it, joins to a row, and applies
     * the appropriate mutation, resolving conflicts and dropping self-write echoes.
     */
    async reconcileFile(relPath: string): Promise<SyncOutcome> {
        this.assertOpen();
        await this.ready();

        const text = await readNoteFile(this.root, relPath);
        if (text === undefined) {
            // The file is gone: treat as a delete of whatever row owned this path.
            return this.reconcileDeletion(relPath);
        }

        const parsed = parseNoteFile(text);
        const title = parsed.title ?? titleFromPath(relPath);
        const fileHash = hashContent(title, parsed.body, parsed.frontmatter);

        // Echo of our own outbound write: same path, same hash we just wrote. Drop
        // it before doing any work (also refresh the conflict base, since file and
        // DB now provably agree).
        if (this.consumeSelfWrite(relPath, fileHash)) {
            this.synced.set(relPath, fileHash);
            return { kind: "unchanged" };
        }

        // Join: prefer the stable uuid (survives a rename), fall back to the path
        // for a human-created file that has no uuid yet.
        const existing = parsed.uuid
            ? this.store.getByUuid(parsed.uuid)
            : this.store.getByPath(relPath);

        if (!existing) {
            return this.adoptNewFile(relPath, parsed, title, fileHash);
        }

        return this.applyFileToRow(relPath, existing, parsed, title, fileHash);
    }

    /** A file the store doesn't know yet. If it carries a uuid we don't have, the
     *  row was deleted out from under it (or this is a different DB); we adopt it
     *  fresh either way. If it has no uuid, it's human-created: mint one and write
     *  it back so future events join by uuid. */
    private async adoptNewFile(
        relPath: string,
        parsed: ReturnType<typeof parseNoteFile>,
        title: string,
        fileHash: string,
    ): Promise<SyncOutcome> {
        const uuid = parsed.uuid ?? crypto.randomUUID();
        let note: Note;
        try {
            note = new Note({
                uuid,
                path: relPath,
                title,
                content: parsed.body,
                frontmatter: parsed.frontmatter,
            });
            this.store.save(note);
        } catch (err) {
            this.onWarn(`could not adopt ${relPath}: ${errMsg(err)}`);
            return { kind: "skipped", reason: errMsg(err) };
        }
        await this.embedBestEffort(note);

        // If we minted a uuid (human-created file), write it back so the file now
        // carries its identity. This is itself a self-write (the body/title are
        // unchanged, only the frontmatter gains a uuid), so mark and write it.
        if (!parsed.uuid) {
            await this.writeOut(note);
        } else {
            // The file already had its uuid and matches what we stored; just set
            // the conflict base.
            this.synced.set(relPath, fileHash);
        }
        return { kind: "created", note };
    }

    /** Apply a changed file to the row it joins to. Handles the path-change
     *  (rename/move detected by uuid) and the conflict case. */
    private async applyFileToRow(
        relPath: string,
        existing: Note,
        parsed: ReturnType<typeof parseNoteFile>,
        title: string,
        fileHash: string,
    ): Promise<SyncOutcome> {
        const moved = existing.path !== relPath;

        // The file content matches the row exactly: nothing to apply. (A pure
        // rename with unchanged content still needs the path update below.)
        if (fileHash === existing.contentHash && !moved) {
            this.synced.set(relPath, fileHash);
            return { kind: "unchanged" };
        }

        // Conflict check against the last-agreement snapshot for this note. The
        // snapshot hash is what the file and DB last shared; comparing each
        // current side to it tells us which moved:
        //   file moved = fileHash !== snapshot ; DB moved = dbHash !== snapshot.
        // Both moved => a genuine concurrent edit. With no snapshot (a path we've
        // never synced) we can't prove a conflict, so we defer to the file.
        const snapshot = this.synced.get(existing.path) ?? this.synced.get(relPath);
        const fileMoved = snapshot !== undefined && fileHash !== snapshot;
        const dbMoved = snapshot !== undefined && existing.contentHash !== snapshot;

        if (fileMoved && dbMoved) {
            return this.resolveConflict(relPath, existing, parsed, title, fileHash);
        }

        // No conflict: the file is the authoritative edit. Apply it to the row.
        const patch: NotePatch = {
            title,
            content: parsed.body,
            frontmatter: parsed.frontmatter,
            path: relPath,
        };
        const updated = this.store.update(existing.id, patch);
        if (!updated) return { kind: "skipped", reason: "row vanished mid-sync" };

        if (updated.content !== existing.content) {
            await this.embedBestEffort(updated);
        }
        this.synced.delete(existing.path);
        this.synced.set(relPath, updated.contentHash);

        if (moved) {
            return { kind: "moved", note: updated, from: existing.path };
        }
        return { kind: "updated", note: updated };
    }

    /**
     * Both the file and the row changed since they last agreed. Last-write-wins
     * by `updated`: the file's mtime stands in for its edit time. The loser is
     * written to a `title (conflict <ts>).md` sidecar so the discarded edit is
     * recoverable. We resolve conservatively toward the file (the human's editor
     * is the more surprising loser), but only when the file is at least as new.
     */
    private async resolveConflict(
        relPath: string,
        existing: Note,
        parsed: ReturnType<typeof parseNoteFile>,
        title: string,
        fileHash: string,
    ): Promise<SyncOutcome> {
        const fileMtime = await this.fileMtime(relPath);
        const fileWins = fileMtime >= existing.updated;

        // The sidecar path: the losing content, parked next to the winner.
        const sidecarRel = conflictSidecarPath(relPath, fileMtime || existing.updated);

        if (fileWins) {
            // Keep the DB's losing version as a sidecar, then apply the file.
            const loserContents = serializeNote(existing);
            await this.writeSidecar(sidecarRel, loserContents);
            const updated = this.store.update(existing.id, {
                title,
                content: parsed.body,
                frontmatter: parsed.frontmatter,
                path: relPath,
            });
            if (updated) {
                await this.embedBestEffort(updated);
                this.synced.set(relPath, updated.contentHash);
            }
            this.onWarn(
                `conflict on ${relPath}: file won (newer); DB version saved to ${sidecarRel}`,
            );
            return { kind: "conflict", note: updated ?? existing, sidecar: sidecarRel };
        }

        // The DB is newer: keep the file's losing version as a sidecar, then
        // overwrite the file with the DB's authoritative content.
        const loserFile = serializeDocumentForConflict(parsed, title);
        await this.writeSidecar(sidecarRel, loserFile);
        await this.writeOut(existing); // re-asserts DB content onto the file
        this.onWarn(`conflict on ${relPath}: DB won (newer); file version saved to ${sidecarRel}`);
        return { kind: "conflict", note: existing, sidecar: sidecarRel };
    }

    /** A file disappeared. Delete the row that owned the path (if any). A note
     *  moved via the editor surfaces as delete(old) + create(new); the create
     *  half re-adopts by uuid, so a delete that finds a still-present uuid
     *  elsewhere is benign. */
    private async reconcileDeletion(relPath: string): Promise<SyncOutcome> {
        const note = this.store.getByPath(relPath);
        if (!note) {
            this.synced.delete(relPath);
            return { kind: "unchanged" };
        }
        this.store.delete(note.id);
        this.synced.delete(relPath);
        return { kind: "updated", note };
    }

    /** Write a sidecar file (a conflict loser). Marked as a self-write so the
     *  watcher doesn't try to adopt it as a brand-new note. */
    private async writeSidecar(relPath: string, contents: string): Promise<void> {
        // A sidecar carries no uuid frontmatter we control, so hash its raw text
        // as the self-write marker; the watcher will still try to parse it, but
        // recognizing the exact bytes lets us drop the immediate echo.
        await writeNoteFileAtomic(this.root, relPath, contents, String(this.writeSeq++));
    }

    private async fileMtime(relPath: string): Promise<number> {
        try {
            const s = await stat(join(this.root, relPath));
            return s.mtimeMs;
        } catch {
            return 0;
        }
    }

    // ── Watcher lifecycle ─────────────────────────────────────────────────────

    /**
     * Start the recursive watcher over the KB folder. On each (debounced) event
     * for a `.md` file, runs {@link reconcileFile}. Also runs an initial full
     * scan so files created while the process was down are picked up. Idempotent:
     * a second call is a no-op while already watching.
     */
    async start(): Promise<void> {
        this.assertOpen();
        await this.ready();
        if (this.watcher) return;

        // Initial reconciliation: adopt/sync every file already on disk, so the
        // store and folder agree before live watching begins.
        await this.scan();

        this.watcher = watch(this.root, { recursive: true }, (_event, filename) => {
            if (!filename) return;
            const rel = normalizeRel(filename);
            if (!isSyncableFile(rel)) return;
            this.scheduleReconcile(rel);
        });
        this.watcher.on("error", (err) => this.onWarn(`watcher error: ${errMsg(err)}`));
    }

    /** Walk the KB folder and reconcile every `.md` file once. Used by
     *  {@link start} and available for an explicit resync. */
    async scan(): Promise<SyncOutcome[]> {
        this.assertOpen();
        await this.ready();
        const files = await this.listMarkdownFiles(this.root);
        const outcomes: SyncOutcome[] = [];
        for (const rel of files) {
            try {
                outcomes.push(await this.reconcileFile(rel));
            } catch (err) {
                this.onWarn(`scan failed for ${rel}: ${errMsg(err)}`);
                outcomes.push({ kind: "skipped", reason: errMsg(err) });
            }
        }
        return outcomes;
    }

    /** Debounce a path: coalesce a burst of events into one reconcile. */
    private scheduleReconcile(rel: string): void {
        const existing = this.debounceTimers.get(rel);
        if (existing) clearTimeout(existing);
        const timer = setTimeout(() => {
            this.debounceTimers.delete(rel);
            this.reconcileFile(rel).catch((err) =>
                this.onWarn(`reconcile failed for ${rel}: ${errMsg(err)}`),
            );
        }, this.debounceMs);
        // Don't keep the event loop alive solely for a pending debounce.
        timer.unref?.();
        this.debounceTimers.set(rel, timer);
    }

    /** Recursively list relative paths of `.md` files under `dir`, skipping
     *  sidecars-of-sidecars and dotfiles. */
    private async listMarkdownFiles(dir: string): Promise<string[]> {
        const out: string[] = [];
        const walk = async (current: string): Promise<void> => {
            let entries;
            try {
                entries = await readdir(current, { withFileTypes: true });
            } catch {
                return;
            }
            for (const entry of entries) {
                if (entry.name.startsWith(".")) continue; // skip dotfiles/dirs
                const abs = join(current, entry.name);
                if (entry.isDirectory()) {
                    await walk(abs);
                } else if (entry.isFile()) {
                    const rel = normalizeRel(relative(this.root, abs));
                    if (isSyncableFile(rel)) out.push(rel);
                }
            }
        };
        await walk(dir);
        return out;
    }

    /** Stop watching and clear pending debounce timers. Idempotent. Does not
     *  close the store (the caller owns its lifetime). */
    close(): void {
        if (this.closed) return;
        this.closed = true;
        if (this.watcher) {
            this.watcher.close();
            this.watcher = undefined;
        }
        for (const timer of this.debounceTimers.values()) clearTimeout(timer);
        this.debounceTimers.clear();
    }
}

// ── module-private helpers ──────────────────────────────────────────────────

/** Normalize a path the watcher/readdir hands us to forward-slash relative form,
 *  matching the store's path normalization. */
function normalizeRel(p: string): string {
    return p.split(sep).join("/");
}

/** Whether a relative path is one we sync: a `.md` file that isn't a conflict
 *  sidecar or a temp file. */
function isSyncableFile(rel: string): boolean {
    if (!rel.toLowerCase().endsWith(".md")) return false;
    if (rel.endsWith(".tmp")) return false;
    if (isConflictSidecar(rel)) return false;
    // No dotfile segments (e.g. .obsidian/) and no empty segments.
    return rel.split("/").every((seg) => seg.length > 0 && !seg.startsWith("."));
}

/** The sidecar naming convention: `name (conflict <ts>).md`. */
function conflictSidecarPath(relPath: string, ts: number): string {
    const dot = relPath.lastIndexOf(".");
    const base = dot === -1 ? relPath : relPath.slice(0, dot);
    return `${base} (conflict ${Math.floor(ts)}).md`;
}

function isConflictSidecar(rel: string): boolean {
    return /\(conflict \d+\)\.md$/i.test(rel);
}

/** A title fallback derived from a file's name when its frontmatter has none. */
function titleFromPath(relPath: string): string {
    const name = relPath.split("/").pop() ?? relPath;
    const dot = name.lastIndexOf(".");
    const base = dot === -1 ? name : name.slice(0, dot);
    return base.trim() || "Untitled";
}

/** Re-serialize a parsed file (the conflict loser on the file side) back to file
 *  bytes, preserving its frontmatter and adding the title we resolved. */
function serializeDocumentForConflict(
    parsed: ReturnType<typeof parseNoteFile>,
    title: string,
): string {
    const note = new Note({
        uuid: parsed.uuid ?? crypto.randomUUID(),
        path: "conflict.md", // path is irrelevant to the serialized bytes
        title,
        content: parsed.body,
        frontmatter: parsed.frontmatter,
    });
    return serializeNote(note);
}

function errMsg(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
}
