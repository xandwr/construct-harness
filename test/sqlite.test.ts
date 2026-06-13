/**
 * Tests for the shared, pure SQLite helpers ({@link toFtsQuery},
 * {@link escapeLike}, {@link clampLimit}). These were extracted from memory.ts so
 * both stores share one sanitizer; they have no database and no I/O, so the
 * tests are plain input/output assertions.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
    toFtsQuery,
    escapeLike,
    clampLimit,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_FTS_TOKENS,
} from "../src/sqlite.ts";

// ---------------------------------------------------------------------------
// toFtsQuery: tokenization, operator neutralization, caps
// ---------------------------------------------------------------------------

test("toFtsQuery extracts word tokens and OR-joins them, each quoted", () => {
    assert.equal(toFtsQuery("hello world"), '"hello" OR "world"');
});

test("toFtsQuery lowercases and dedupes tokens", () => {
    assert.equal(toFtsQuery("Milk milk MILK"), '"milk"');
});

test("toFtsQuery neutralizes FTS operators and punctuation as literal tokens", () => {
    // AND/OR/NOT, quotes, parens, colons, stars would all be operators raw; here
    // they must reduce to harmless quoted word tokens and never appear bare.
    const q = toFtsQuery('oranges AND "(*: NOT apples');
    assert.ok(q !== null);
    // No bare operator survives: every emitted token is double-quoted.
    assert.equal(q, '"oranges" OR "and" OR "not" OR "apples"');
});

test("toFtsQuery preserves Unicode letters and digits", () => {
    assert.equal(toFtsQuery("café 2024 naïve"), '"café" OR "2024" OR "naïve"');
});

test("toFtsQuery returns null when there are no usable tokens", () => {
    assert.equal(toFtsQuery(""), null);
    assert.equal(toFtsQuery("   "), null);
    assert.equal(toFtsQuery("!!! ??? ..."), null);
    assert.equal(toFtsQuery("- * : ( )"), null);
});

test("toFtsQuery returns null for a non-string input", () => {
    // @ts-expect-error deliberately wrong type
    assert.equal(toFtsQuery(123), null);
    // @ts-expect-error deliberately wrong type
    assert.equal(toFtsQuery(null), null);
});

test("toFtsQuery caps the number of tokens at MAX_FTS_TOKENS", () => {
    // Far more distinct tokens than the cap; the result must hold exactly the cap.
    const many = Array.from({ length: MAX_FTS_TOKENS + 50 }, (_, i) => `t${i}`).join(" ");
    const q = toFtsQuery(many);
    assert.ok(q !== null);
    const emitted = q.split(" OR ");
    assert.equal(emitted.length, MAX_FTS_TOKENS);
});

// ---------------------------------------------------------------------------
// escapeLike: wildcard escaping
// ---------------------------------------------------------------------------

test("escapeLike escapes the LIKE wildcards % and _ and the escape char itself", () => {
    assert.equal(escapeLike("100%"), "100\\%");
    assert.equal(escapeLike("a_b"), "a\\_b");
    assert.equal(escapeLike("back\\slash"), "back\\\\slash");
    assert.equal(escapeLike("%_\\"), "\\%\\_\\\\");
});

test("escapeLike leaves ordinary text untouched", () => {
    assert.equal(escapeLike("just words 42"), "just words 42");
});

// ---------------------------------------------------------------------------
// clampLimit: bounding caller limits
// ---------------------------------------------------------------------------

test("clampLimit falls back to DEFAULT_LIMIT for undefined and junk", () => {
    assert.equal(clampLimit(undefined), DEFAULT_LIMIT);
    assert.equal(clampLimit(0), DEFAULT_LIMIT);
    assert.equal(clampLimit(-5), DEFAULT_LIMIT);
    assert.equal(clampLimit(NaN), DEFAULT_LIMIT);
    assert.equal(clampLimit(Infinity), DEFAULT_LIMIT);
});

test("clampLimit caps at MAX_LIMIT and floors fractional limits", () => {
    assert.equal(clampLimit(MAX_LIMIT + 1000), MAX_LIMIT);
    assert.equal(clampLimit(3.9), 3);
    assert.equal(clampLimit(MAX_LIMIT), MAX_LIMIT);
    assert.equal(clampLimit(1), 1);
});
