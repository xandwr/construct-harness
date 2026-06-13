/**
 * Small SQLite query helpers shared across stores.
 *
 * These are the pure, store-agnostic primitives that both {@link MemoryStore}
 * and {@link EventStore} need verbatim: turning free user text into a safe FTS5
 * MATCH query, escaping LIKE wildcards, and clamping caller-supplied limits. They
 * live here, not duplicated per store, so the two substrates can never drift in
 * how they sanitize input or bound a result set: a security and correctness
 * property, not just tidiness (an FTS-escaping bug fixed in one place is fixed
 * everywhere).
 *
 * Everything here is a pure function or a constant: no database handle, no I/O,
 * no provider. That keeps it trivially testable and safe to import from any
 * layer.
 */

/** Default rows a bounded query returns when the caller names no limit. */
export const DEFAULT_LIMIT = 100;
/** Hard ceiling on a single query's row count, so a runaway limit can't ask
 *  the database for an unbounded result set. */
export const MAX_LIMIT = 1000;

/** Hard ceiling on stored textual content, so a runaway write can't bloat the
 *  database. Shared by every store that persists a free-text payload. */
export const MAX_CONTENT_LENGTH = 100_000;

/** Cap on tokens fed into a single FTS MATCH, to bound a giant prompt. */
export const MAX_FTS_TOKENS = 32;

/**
 * Clamp a caller-supplied limit into a sane, bounded range.
 *
 * `undefined`, non-finite, or non-positive limits fall back to
 * {@link DEFAULT_LIMIT}; anything larger than {@link MAX_LIMIT} is capped to it.
 * The result is always a positive integer, so a query can pass it straight to
 * `LIMIT ?` without a runaway or a zero-row surprise.
 */
export function clampLimit(limit: number | undefined): number {
    if (limit === undefined) return DEFAULT_LIMIT;
    if (!Number.isFinite(limit) || limit <= 0) return DEFAULT_LIMIT;
    return Math.min(Math.floor(limit), MAX_LIMIT);
}

/**
 * Escape LIKE wildcards so user text is matched literally (with `ESCAPE '\'`).
 *
 * Without this, a memory containing `%` or `_` (or `\`) could turn a substring
 * search into an unintended wildcard match. Callers MUST pair the escaped string
 * with `LIKE ? ESCAPE '\'` in their SQL for the escape character to take effect.
 */
export function escapeLike(s: string): string {
    return s.replace(/[\\%_]/g, (c) => `\\${c}`);
}

/**
 * Turn free text into a safe FTS5 MATCH query.
 *
 * Raw user text can't go straight into MATCH: characters like `"`, `*`, `:`,
 * `(`, `-`, and the bareword `AND`/`OR`/`NOT` are FTS operators and would
 * either throw a syntax error or change the query's meaning. So we extract
 * alphanumeric word tokens, wrap each in double quotes (which makes it a
 * literal string token, neutralizing operators), and OR them together: any
 * shared term makes a row a candidate, and bm25 sorts by how well it matches.
 *
 * Returns null when the text has no usable tokens, so the caller can treat that
 * as "no query" rather than issuing an empty MATCH (which is itself an error).
 */
export function toFtsQuery(text: string): string | null {
    if (typeof text !== "string") return null;
    const tokens = text.toLowerCase().match(/[\p{L}\p{N}]+/gu);
    if (!tokens || tokens.length === 0) return null;
    // Dedupe to keep the query compact; cap to bound pathological inputs.
    const unique = [...new Set(tokens)].slice(0, MAX_FTS_TOKENS);
    return unique.map((t) => `"${t}"`).join(" OR ");
}
