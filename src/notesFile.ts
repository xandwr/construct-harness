/**
 * The file half of the knowledge base: turning a {@link Note} into a markdown
 * file on disk and back, with the path-safety and atomicity the two-way sync
 * engine relies on.
 *
 * This is the lowest layer of the sync subsystem and the *only* place the harness
 * does file I/O. It is deliberately small and free of any watcher/conflict logic
 * (those live in {@link NotesService} / the sync engine above it): here we only
 *  - map a note's modeled fields to/from a file's frontmatter + body,
 *  - resolve a note's relative `path` against the KB root *safely* (no escaping
 *    the root via `..`, symlinks, or absolute paths), and
 *  - write/rename/delete files *atomically*, so the watcher and any editor never
 *    observe a half-written file.
 *
 * Atomicity matters specifically because of the echo loop: an outbound write
 * fires the watcher. A torn read by the watcher (or an editor) of a partially
 * written file would be observed as a spurious "external change". Writing to a
 * temp file in the same directory and `rename`-ing it into place makes the
 * replacement a single atomic step on every POSIX filesystem.
 */

import { mkdir, readFile, rename, rm, writeFile, realpath } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import { Note, type NoteFrontmatter, normalizePath } from "./notes.ts";
import { parseDocument, serializeDocument, type FrontmatterValue } from "./frontmatter.ts";

/** Thrown when a file operation can't be performed safely (a path that escapes
 *  the KB root) or a file's contents can't be reconciled with a note. */
export class NotesFileError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "NotesFileError";
    }
}

/** The order modeled keys appear in a file's frontmatter, so a written file
 *  reads predictably (identity first, then human-facing metadata) and diffs
 *  stay stable across writes. Unmodeled human keys follow, in their own order. */
const MODELED_KEY_ORDER = ["uuid", "title"] as const;

/**
 * Serialize a note to the exact bytes its file should contain: a frontmatter
 * block (uuid + title first, then the note's other frontmatter keys) followed by
 * the markdown body. Inverse of {@link parseNoteFile} for the fields we model.
 *
 * The `uuid` is always written: it is the stable join key the inbound watcher
 * reads back to recognize the file as this note rather than a new one. `title`
 * is written too (so a human editing the file sees and can change it), even
 * though it's also encoded in the filename, because the filename is not
 * authoritative: the frontmatter is.
 */
export function serializeNote(note: Note): string {
    const fm: NoteFrontmatter = {};
    fm.uuid = note.uuid;
    fm.title = note.title;
    // The note's own frontmatter keys, minus any that collide with the modeled
    // ones (which would have been stripped on the way in, but be defensive).
    for (const [key, value] of Object.entries(note.frontmatter)) {
        if ((MODELED_KEY_ORDER as readonly string[]).includes(key)) continue;
        fm[key] = value;
    }
    return serializeDocument(fm, note.content);
}

/** The fields recovered from a markdown file: the modeled identity/title, the
 *  leftover (unmodeled) frontmatter, and the body. `uuid` is absent for a
 *  human-created file that has never been synced. */
export interface ParsedNoteFile {
    uuid: string | undefined;
    title: string | undefined;
    frontmatter: NoteFrontmatter;
    body: string;
}

/**
 * Parse a markdown file's text into the parts a note needs. Pulls `uuid` and
 * `title` out of the frontmatter (the modeled fields), and returns the rest of
 * the frontmatter as the note's `frontmatter` blob. Tolerant by contract (it
 * leans on {@link parseDocument}'s tolerant parse): a file with no frontmatter
 * yields `uuid: undefined` and an empty frontmatter map, which the caller treats
 * as a freshly-created human file to adopt.
 */
export function parseNoteFile(text: string): ParsedNoteFile {
    const { frontmatter, body } = parseDocument(text);

    const uuid = readScalarString(frontmatter.uuid);
    const title = readScalarString(frontmatter.title);

    // The note's frontmatter blob is everything except the modeled keys.
    const rest: NoteFrontmatter = {};
    for (const [key, value] of Object.entries(frontmatter)) {
        if ((MODELED_KEY_ORDER as readonly string[]).includes(key)) continue;
        rest[key] = value;
    }

    return { uuid, title, frontmatter: rest, body };
}

/** Coerce a frontmatter value to a trimmed non-empty string, or undefined. Used
 *  for uuid/title where only a string makes sense. */
function readScalarString(v: FrontmatterValue | undefined): string | undefined {
    if (typeof v !== "string") return undefined;
    const s = v.trim();
    return s.length ? s : undefined;
}

/**
 * Resolve a note's relative `path` against the KB root, guaranteeing the result
 * stays inside the root. Two layers of defense:
 *  1. {@link normalizePath} has already rejected `..` segments, absolute paths,
 *     and control chars at the store boundary; we re-run it here so this function
 *     is safe to call on any string, not just an already-validated note path.
 *  2. We resolve the joined path and assert it is still prefixed by the resolved
 *     root, catching anything normalization missed (defense in depth).
 *
 * Returns the absolute filesystem path. Throws {@link NotesFileError} if the
 * path would escape the root.
 */
export function resolveInRoot(root: string, relativePath: string): string {
    const safeRel = normalizePath(relativePath); // throws NoteError on traversal
    const absRoot = resolve(root);
    const abs = resolve(absRoot, safeRel);
    // The resolved path must be the root itself or strictly under it.
    const rootWithSep = absRoot.endsWith(sep) ? absRoot : absRoot + sep;
    if (abs !== absRoot && !abs.startsWith(rootWithSep)) {
        throw new NotesFileError(`path escapes the knowledge-base root: ${relativePath}`);
    }
    return abs;
}

/**
 * Write `contents` to the file for `relativePath` under `root`, atomically.
 *
 * Creates any missing parent directories, writes to a sibling temp file, then
 * renames it into place. The rename is atomic on POSIX, so a reader (the watcher
 * or an editor) never sees a partially written file. Returns the absolute path
 * written.
 *
 * The temp file is a sibling (same directory) so the rename never crosses a
 * filesystem boundary (which would make it a copy, not an atomic move).
 */
export async function writeNoteFileAtomic(
    root: string,
    relativePath: string,
    contents: string,
    tempSuffix: string,
): Promise<string> {
    const abs = resolveInRoot(root, relativePath);
    await mkdir(dirname(abs), { recursive: true });

    // A unique-enough temp name in the same directory. The suffix is supplied by
    // the caller (e.g. a per-write counter) so concurrent writes to the same path
    // don't collide on the temp file; Date.now()/random are avoided to keep the
    // module deterministic and testable.
    const tmp = `${abs}.${tempSuffix}.tmp`;
    try {
        await writeFile(tmp, contents, { encoding: "utf8", mode: 0o644 });
        await rename(tmp, abs);
    } catch (err) {
        // Best-effort cleanup of the temp file on failure, then surface the error.
        await rm(tmp, { force: true }).catch(() => {});
        throw err;
    }
    return abs;
}

/**
 * Move the file for `fromRel` to `toRel` under `root`, atomically, creating the
 * destination's parent directories. Used when a note's path changes (the DB-side
 * rename mirrors out to disk). A no-op (and not an error) when the source does
 * not exist, so an outbound rename for a note whose file was never written still
 * converges. Returns the absolute destination path.
 */
export async function moveNoteFile(root: string, fromRel: string, toRel: string): Promise<string> {
    const fromAbs = resolveInRoot(root, fromRel);
    const toAbs = resolveInRoot(root, toRel);
    if (fromAbs === toAbs) return toAbs;
    await mkdir(dirname(toAbs), { recursive: true });
    if (!existsSync(fromAbs)) return toAbs; // nothing to move; destination is the target
    await rename(fromAbs, toAbs);
    return toAbs;
}

/**
 * Delete the file for `relativePath` under `root`. Idempotent: a missing file is
 * not an error (the DB row and the file can briefly disagree during a delete;
 * converging to "absent" is the goal). Returns whether a file was removed.
 */
export async function deleteNoteFile(root: string, relativePath: string): Promise<boolean> {
    const abs = resolveInRoot(root, relativePath);
    if (!existsSync(abs)) return false;
    await rm(abs, { force: true });
    return true;
}

/** Read the raw text of a note file under `root`, or undefined if it's gone. */
export async function readNoteFile(
    root: string,
    relativePath: string,
): Promise<string | undefined> {
    const abs = resolveInRoot(root, relativePath);
    try {
        return await readFile(abs, "utf8");
    } catch (err) {
        if ((err as NodeJS.ErrnoException).code === "ENOENT") return undefined;
        throw err;
    }
}

/**
 * Ensure the KB root directory exists, returning its *real* (symlink-resolved)
 * absolute path. The sync engine resolves note paths against this real root, so
 * a symlinked KB folder can't be used to smuggle a write outside it. Safe to
 * call repeatedly.
 */
export async function ensureRoot(root: string): Promise<string> {
    await mkdir(root, { recursive: true });
    // realpath after mkdir: resolves any symlink in the root itself so the
    // resolveInRoot prefix check compares real path to real path.
    return await realpath(root);
}

/** Derive a default relative path for a note that has none yet, from its title:
 *  a filesystem-safe slug plus `.md`. Used when minting a file for a note created
 *  via the API/agent without an explicit path. Falls back to the uuid when the
 *  title slugs to nothing (e.g. a title of only punctuation). */
export function defaultPathForNote(title: string, uuid: string): string {
    const slug = title
        .toLowerCase()
        .normalize("NFKD")
        .replace(/[^\p{L}\p{N}]+/gu, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 80);
    const base = slug.length ? slug : uuid;
    return `${base}.md`;
}

/** Join helper re-exported so callers can build absolute paths without importing
 *  node:path directly (keeps the file-path concern in one module). */
export { join as joinPath };
