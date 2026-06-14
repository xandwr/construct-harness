/**
 * A small, honest frontmatter (de)serializer for the knowledge base.
 *
 * A KB note on disk is a markdown file with an optional YAML frontmatter block
 * delimited by `---` lines:
 *
 *     ---
 *     uuid: 0c5f...-...
 *     title: My note
 *     tags: [work, urgent]
 *     importance: 0.8
 *     ---
 *     The markdown body starts here.
 *
 * We deliberately do NOT depend on a full YAML library (`gray-matter` and
 * friends). The harness is zero-runtime-dependency by design: it hand-rolls its
 * own embedding serialization, FTS query building, and SSE framing, and this is
 * the same call. A note's frontmatter is *our* format first (we write the keys
 * the KB models), and a human's arbitrary scalars second; both fit a tiny,
 * well-understood subset of YAML. A subset we control end-to-end is safer here
 * than a general parser whose edge cases we'd have to reason about anyway.
 *
 * The supported subset, chosen to round-trip everything the KB writes and to
 * tolerate what a human is likely to add by hand:
 *  - `key: value` scalars: strings (quoted or bare), numbers, booleans, null.
 *  - Flow-sequence string arrays: `tags: [a, b, c]`.
 *  - Block-sequence string arrays:  `tags:` then `  - a` / `  - b` lines.
 *  - `#` line comments and blank lines inside the block (ignored).
 *
 * Anything outside the subset (nested maps, multi-line scalars, anchors) is
 * tolerated on read by being kept as its raw string value rather than throwing:
 * a corrupt or exotic block degrades to "best-effort scalars" and never takes
 * down a sync. This mirrors the tolerant-read posture of the stores (a corrupt
 * `tags`/`meta` payload degrades instead of crashing a query).
 *
 * Pure functions only: no I/O, no provider, trivially testable.
 */

/** A parsed frontmatter value. Arrays are always string arrays in our subset;
 *  scalars are string/number/boolean/null. */
export type FrontmatterValue = string | number | boolean | null | string[];

/** The parsed shape of a markdown document with optional frontmatter. */
export interface ParsedDocument {
    /** The frontmatter key/value map (empty when there was no block). */
    frontmatter: Record<string, FrontmatterValue>;
    /** The markdown body with the frontmatter block (and its delimiters) removed.
     *  Canonicalized so parse/serialize is a true fixpoint: leading blank lines
     *  immediately after the closing `---` are dropped, and trailing newlines are
     *  trimmed (serialization re-adds the single trailing newline a file wants).
     *  So `parseDocument(serializeDocument(fm, body)).body === body` for any body
     *  the parser itself produced. */
    body: string;
}

/** The line that opens and closes a frontmatter block. Must be exactly `---`
 *  (after trimming trailing whitespace) on its own line. */
const DELIM = "---";

/**
 * Split a markdown document into its frontmatter map and its body.
 *
 * A frontmatter block is recognized only when the very first line is `---` and a
 * later `---` line closes it. Without an opening delimiter the whole input is the
 * body and `frontmatter` is empty: a plain markdown file is valid input.
 *
 * Tolerant by contract: a malformed block (e.g. never closed) is treated as
 * "no frontmatter, all body" rather than an error, so a half-written or hand-
 * mangled file still parses to *something* the sync can act on.
 */
export function parseDocument(input: string): ParsedDocument {
    // Normalize CRLF so a file saved by a Windows editor parses identically.
    const text = input.replace(/\r\n/g, "\n");
    const lines = text.split("\n");

    // The opening delimiter must be the first line (allowing a leading BOM).
    const first = lines[0]?.replace(/^﻿/, "");
    if (first?.trimEnd() !== DELIM) {
        return { frontmatter: {}, body: canonicalizeBody(stripLeadingBom(text)) };
    }

    // Find the closing delimiter.
    let close = -1;
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trimEnd() === DELIM) {
            close = i;
            break;
        }
    }
    // Unterminated block: treat the whole thing as body (degrade, don't throw).
    if (close === -1) {
        return { frontmatter: {}, body: canonicalizeBody(stripLeadingBom(text)) };
    }

    const fmLines = lines.slice(1, close);
    const bodyLines = lines.slice(close + 1);
    // Drop the single blank line conventionally left between the block and the
    // body, so a round-trip doesn't accrete leading newlines.
    while (bodyLines.length && bodyLines[0].trim() === "") bodyLines.shift();

    return {
        frontmatter: parseFrontmatterLines(fmLines),
        body: canonicalizeBody(bodyLines.join("\n")),
    };
}

/** Strip trailing newlines so the stored body is canonical (serialization
 *  re-adds the one a file wants); makes parse/serialize a true fixpoint. */
function canonicalizeBody(body: string): string {
    return body.replace(/\n+$/, "");
}

/** Drop a leading byte-order mark if present, so a body never carries one. */
function stripLeadingBom(s: string): string {
    return s.replace(/^﻿/, "");
}

/** Parse the lines *between* the delimiters into a key/value map. */
function parseFrontmatterLines(lines: string[]): Record<string, FrontmatterValue> {
    const out: Record<string, FrontmatterValue> = {};

    for (let i = 0; i < lines.length; i++) {
        const raw = lines[i];
        const line = stripComment(raw);
        if (line.trim() === "") continue;

        // A block-sequence item (`- x`) with no preceding key is malformed; skip
        // it rather than crash. (Handled in-context below when it follows a key.)
        const kv = splitKeyValue(line);
        if (!kv) continue;
        const { key, value } = kv;

        if (value === "") {
            // A bare `key:` may introduce a block sequence on the following
            // indented `- item` lines. Collect them; if none follow, it's an
            // empty value, which we record as an empty string.
            const items: string[] = [];
            let j = i + 1;
            while (j < lines.length) {
                const itemLine = stripComment(lines[j]);
                if (itemLine.trim() === "") {
                    j++;
                    continue;
                }
                const item = parseBlockSequenceItem(itemLine);
                if (item === null) break;
                items.push(item);
                j++;
            }
            if (items.length > 0) {
                out[key] = items;
                i = j - 1; // resume after the consumed sequence
            } else {
                out[key] = "";
            }
            continue;
        }

        out[key] = parseScalarOrFlowArray(value);
    }

    return out;
}

/** Strip a trailing `# comment`, but only when the `#` is not inside quotes.
 *  Keeps `title: "a # b"` intact while dropping `tags: [a] # note`. */
function stripComment(line: string): string {
    let inSingle = false;
    let inDouble = false;
    for (let i = 0; i < line.length; i++) {
        const c = line[i];
        if (inDouble && c === "\\" && i + 1 < line.length) {
            i++; // skip the escaped char so an escaped quote can't end the run
            continue;
        }
        if (c === "'" && !inDouble) inSingle = !inSingle;
        else if (c === '"' && !inSingle) inDouble = !inDouble;
        else if (c === "#" && !inSingle && !inDouble) {
            // A `#` only starts a comment when preceded by whitespace or at the
            // start of the (trimmed) content, matching YAML's rule loosely.
            if (i === 0 || /\s/.test(line[i - 1])) return line.slice(0, i);
        }
    }
    return line;
}

/** Split `key: value` into its parts, or null if there's no top-level colon.
 *  Only the first colon separates; the rest belongs to the value (so a value may
 *  itself contain colons, e.g. a URL or timestamp). Block-sequence lines (`- x`)
 *  return null here. */
function splitKeyValue(line: string): { key: string; value: string } | null {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ") || trimmed === "-") return null;
    const colon = trimmed.indexOf(":");
    if (colon === -1) return null;
    const key = trimmed.slice(0, colon).trim();
    if (key === "") return null;
    const value = trimmed.slice(colon + 1).trim();
    return { key, value };
}

/** Parse one block-sequence line (`  - item`) into its string item, or null if
 *  the line isn't a sequence item (which ends the sequence). */
function parseBlockSequenceItem(line: string): string | null {
    const trimmed = line.trim();
    if (trimmed === "-") return "";
    if (!trimmed.startsWith("- ")) return null;
    // Sequence items are strings in our subset; coerce so `- 42` joins a string
    // array as "42" rather than mixing types (mirrors the flow-array handling).
    const v = unquoteScalar(trimmed.slice(2).trim());
    return typeof v === "string" ? v : String(v);
}

/** Parse a scalar value, or a flow-sequence array `[a, b, c]` of scalars. */
function parseScalarOrFlowArray(value: string): FrontmatterValue {
    if (value.startsWith("[") && value.endsWith("]")) {
        const inner = value.slice(1, -1).trim();
        if (inner === "") return [];
        return splitFlowItems(inner).map((item) => {
            const v = unquoteScalar(item.trim());
            // Flow arrays in our subset are string arrays; coerce so `[1, 2]`
            // round-trips as strings rather than mixing types in one column.
            return typeof v === "string" ? v : String(v);
        });
    }
    // A quoted scalar decodes via unquoteScalar (preserving its literal string);
    // a bare scalar coerces to bool/number/null where unambiguous.
    return unquoteScalar(value);
}

/** Split the inside of a flow array on commas that aren't inside quotes, so
 *  `["a, b", c]` yields two items, not three. A backslash inside a double-quoted
 *  run escapes the next character, so `"c\"d"` is one quoted item, not two. */
function splitFlowItems(inner: string): string[] {
    const items: string[] = [];
    let current = "";
    let inSingle = false;
    let inDouble = false;
    for (let i = 0; i < inner.length; i++) {
        const c = inner[i];
        if (inDouble && c === "\\" && i + 1 < inner.length) {
            // Keep the escape pair intact for unquoteScalar to decode later.
            current += c + inner[i + 1];
            i++;
            continue;
        }
        if (c === "'" && !inDouble) inSingle = !inSingle;
        else if (c === '"' && !inSingle) inDouble = !inDouble;
        if (c === "," && !inSingle && !inDouble) {
            items.push(current);
            current = "";
        } else {
            current += c;
        }
    }
    if (current.trim() !== "" || items.length > 0) items.push(current);
    return items;
}

/** Remove surrounding quotes from a scalar, decoding the few escapes a quoted
 *  YAML scalar uses; a bare (unquoted) scalar passes through to {@link coerceScalar}. */
function unquoteScalar(value: string): FrontmatterValue {
    if (value.length >= 2 && value.startsWith('"') && value.endsWith('"')) {
        // Double-quoted: decode \" \\ \n \t.
        return value
            .slice(1, -1)
            .replace(/\\(["\\nt])/g, (_, ch) =>
                ch === "n" ? "\n" : ch === "t" ? "\t" : ch === '"' ? '"' : "\\",
            );
    }
    if (value.length >= 2 && value.startsWith("'") && value.endsWith("'")) {
        // Single-quoted: only `''` -> `'` is special.
        return value.slice(1, -1).replace(/''/g, "'");
    }
    return coerceScalar(value);
}

/** Coerce a bare scalar to a boolean/number/null where unambiguous, else keep
 *  it as the verbatim string. Conservative: only the canonical literals coerce,
 *  so `title: 12 angry men` stays a string. */
function coerceScalar(value: string): FrontmatterValue {
    if (value === "") return "";
    if (value === "true") return true;
    if (value === "false") return false;
    if (value === "null" || value === "~") return null;
    // A number only if the whole string is one (no trailing prose).
    if (/^-?\d+(\.\d+)?$/.test(value)) {
        const n = Number(value);
        if (Number.isFinite(n)) return n;
    }
    return value;
}

/**
 * Serialize a frontmatter map and a body back into a markdown document.
 *
 * Inverse of {@link parseDocument} for the subset we control: a document we
 * write and read back yields the same `frontmatter` and `body`. Keys are emitted
 * in insertion order (so callers control field order: uuid first, then title,
 * etc.), values are quoted only when necessary, and string arrays use the
 * compact flow form `[a, b]`.
 *
 * An empty frontmatter map yields just the body (no empty `---` block), so a
 * note with no metadata is a plain markdown file.
 */
export function serializeDocument(
    frontmatter: Record<string, FrontmatterValue>,
    body: string,
): string {
    const keys = Object.keys(frontmatter);
    if (keys.length === 0) return ensureTrailingNewline(body);

    const lines: string[] = [DELIM];
    for (const key of keys) {
        lines.push(serializeEntry(key, frontmatter[key]));
    }
    lines.push(DELIM);
    // One blank line between the block and the body, matching what parse trims.
    return `${lines.join("\n")}\n\n${ensureTrailingNewline(body)}`;
}

/** Serialize one `key: value` (or `key: [a, b]`) entry. */
function serializeEntry(key: string, value: FrontmatterValue): string {
    if (Array.isArray(value)) {
        const items = value.map((v) => serializeScalar(v, true)).join(", ");
        return `${key}: [${items}]`;
    }
    return `${key}: ${serializeScalar(value, false)}`;
}

/** Serialize a scalar, quoting it when a bare form would be ambiguous or would
 *  not round-trip (contains a colon-space, a comment `#`, leading/trailing
 *  space, brackets, or looks like a different type than it is). `inFlow` adds
 *  comma/bracket to the quote triggers. */
function serializeScalar(value: string | number | boolean | null, inFlow: boolean): string {
    if (value === null) return "null";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return String(value);

    const s = value;
    if (s === "") return '""';

    const needsQuote =
        s !== s.trim() || // leading/trailing whitespace
        /[:#"'\[\]{}]/.test(s) || // structural / quote chars
        /\n/.test(s) || // newline
        (inFlow && s.includes(",")) ||
        s === "true" ||
        s === "false" ||
        s === "null" ||
        s === "~" ||
        /^-?\d+(\.\d+)?$/.test(s) || // would coerce to a number
        s.startsWith("- ") || // would look like a sequence item
        s.startsWith("-"); // leading dash is ambiguous

    if (!needsQuote) return s;
    // Double-quote and escape the few chars that need it.
    const escaped = s
        .replace(/\\/g, "\\\\")
        .replace(/"/g, '\\"')
        .replace(/\n/g, "\\n")
        .replace(/\t/g, "\\t");
    return `"${escaped}"`;
}

/** Guarantee exactly one trailing newline, so files end cleanly and editors
 *  don't flag a missing final newline (a common spurious diff). */
function ensureTrailingNewline(body: string): string {
    const trimmed = body.replace(/\n+$/, "");
    return trimmed.length === 0 ? "" : `${trimmed}\n`;
}
