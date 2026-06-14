/**
 * Tests for the zero-dependency frontmatter (de)serializer ({@link parseDocument}
 * / {@link serializeDocument}).
 *
 * The load-bearing property is round-tripping: anything the KB writes parses
 * back to the same frontmatter map and body. The rest pins down the tolerant-
 * read behavior (a malformed block degrades to "all body" instead of throwing)
 * and the quoting rules that keep a value from changing type across a round trip.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { parseDocument, serializeDocument, type FrontmatterValue } from "../src/frontmatter.ts";

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

test("parses a basic block with mixed scalar types", () => {
    const doc = parseDocument(
        [
            "---",
            "uuid: abc-123",
            "title: My Note",
            "importance: 0.8",
            "draft: true",
            "---",
            "Body here.",
        ].join("\n"),
    );
    assert.equal(doc.frontmatter.uuid, "abc-123");
    assert.equal(doc.frontmatter.title, "My Note");
    assert.equal(doc.frontmatter.importance, 0.8);
    assert.equal(doc.frontmatter.draft, true);
    assert.equal(doc.body, "Body here.");
});

test("a file with no frontmatter is all body", () => {
    const doc = parseDocument("# Just markdown\n\nNo block.");
    assert.deepEqual(doc.frontmatter, {});
    assert.equal(doc.body, "# Just markdown\n\nNo block.");
});

test("an unterminated block degrades to all-body, not an error", () => {
    const input = "---\nuuid: x\ntitle: never closed\nstill going";
    const doc = parseDocument(input);
    assert.deepEqual(doc.frontmatter, {});
    assert.equal(doc.body, input);
});

test("parses a flow-sequence string array", () => {
    const doc = parseDocument("---\ntags: [work, urgent, deploy]\n---\nx");
    assert.deepEqual(doc.frontmatter.tags, ["work", "urgent", "deploy"]);
});

test("an empty flow array parses to []", () => {
    const doc = parseDocument("---\ntags: []\n---\nx");
    assert.deepEqual(doc.frontmatter.tags, []);
});

test("parses a block-sequence string array", () => {
    const doc = parseDocument(["---", "tags:", "  - work", "  - urgent", "---", "body"].join("\n"));
    assert.deepEqual(doc.frontmatter.tags, ["work", "urgent"]);
});

test("a block sequence ends at the next key", () => {
    const doc = parseDocument(
        ["---", "tags:", "  - a", "  - b", "title: after", "---", "body"].join("\n"),
    );
    assert.deepEqual(doc.frontmatter.tags, ["a", "b"]);
    assert.equal(doc.frontmatter.title, "after");
});

test("quoted scalars keep colons, hashes, and commas verbatim", () => {
    const doc = parseDocument(
        ["---", 'title: "a: b, c # d"', "url: 'http://x.test/p'", "---", "body"].join("\n"),
    );
    assert.equal(doc.frontmatter.title, "a: b, c # d");
    assert.equal(doc.frontmatter.url, "http://x.test/p");
});

test("a value may contain colons when only the first splits the key", () => {
    const doc = parseDocument("---\nwhen: 2026-06-13T10:00:00\n---\nx");
    assert.equal(doc.frontmatter.when, "2026-06-13T10:00:00");
});

test("a trailing comment is stripped but a quoted hash is kept", () => {
    const doc = parseDocument(
        ["---", "tags: [a, b] # a note", 'title: "issue #42"', "---", "x"].join("\n"),
    );
    assert.deepEqual(doc.frontmatter.tags, ["a", "b"]);
    assert.equal(doc.frontmatter.title, "issue #42");
});

test("blank lines and comment lines inside the block are ignored", () => {
    const doc = parseDocument(["---", "# a comment", "", "title: x", "", "---", "body"].join("\n"));
    assert.equal(doc.frontmatter.title, "x");
    assert.equal(Object.keys(doc.frontmatter).length, 1);
});

test("CRLF line endings parse identically to LF", () => {
    const doc = parseDocument("---\r\ntitle: x\r\n---\r\nbody line\r\nsecond");
    assert.equal(doc.frontmatter.title, "x");
    assert.equal(doc.body, "body line\nsecond");
});

test("leading blank lines after the block are trimmed from the body", () => {
    const doc = parseDocument("---\ntitle: x\n---\n\n\nthe body");
    assert.equal(doc.body, "the body");
});

test("number coercion is conservative: prose stays a string", () => {
    const doc = parseDocument(
        ["---", "n: 42", "f: 3.14", "title: 12 angry men", "ver: v2", "---", "x"].join("\n"),
    );
    assert.equal(doc.frontmatter.n, 42);
    assert.equal(doc.frontmatter.f, 3.14);
    assert.equal(doc.frontmatter.title, "12 angry men");
    assert.equal(doc.frontmatter.ver, "v2");
});

test("null literals parse to null", () => {
    const doc = parseDocument("---\na: null\nb: ~\n---\nx");
    assert.equal(doc.frontmatter.a, null);
    assert.equal(doc.frontmatter.b, null);
});

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

test("an empty frontmatter map serializes to just the body", () => {
    assert.equal(serializeDocument({}, "hello"), "hello\n");
});

test("serializes scalars and a flow array in insertion order", () => {
    const out = serializeDocument(
        { uuid: "abc", title: "My Note", tags: ["a", "b"], importance: 0.5 },
        "Body.",
    );
    assert.equal(
        out,
        [
            "---",
            "uuid: abc",
            "title: My Note",
            "tags: [a, b]",
            "importance: 0.5",
            "---",
            "",
            "Body.",
            "",
        ].join("\n"),
    );
});

test("serialization quotes values that would otherwise change type or break parsing", () => {
    const out = serializeDocument(
        { a: "true", b: "42", c: "has: colon", d: "trailing ", e: "a, b" },
        "x",
    );
    // Re-parse and confirm each survived as the original string.
    const back = parseDocument(out);
    assert.equal(back.frontmatter.a, "true");
    assert.equal(back.frontmatter.b, "42");
    assert.equal(back.frontmatter.c, "has: colon");
    assert.equal(back.frontmatter.d, "trailing ");
    assert.equal(back.frontmatter.e, "a, b");
});

test("body always ends with exactly one newline", () => {
    assert.ok(serializeDocument({ a: "x" }, "no newline").endsWith("no newline\n"));
    assert.ok(serializeDocument({ a: "x" }, "many\n\n\n").endsWith("many\n"));
});

// ---------------------------------------------------------------------------
// Round-trip (the load-bearing property)
// ---------------------------------------------------------------------------

test("round-trips the full KB frontmatter shape", () => {
    const fm: Record<string, FrontmatterValue> = {
        uuid: "0c5f1d2e-aaaa-bbbb-cccc-1234567890ab",
        title: "Deploy runbook",
        tags: ["ops", "deploy", "on-call"],
        importance: 0.9,
        archived: false,
        owner: null,
    };
    const body = "# Deploy runbook\n\nStep 1.\nStep 2.";
    const round = parseDocument(serializeDocument(fm, body));
    assert.deepEqual(round.frontmatter, fm);
    assert.equal(round.body, body);
});

test("round-trips a value containing every awkward character", () => {
    const fm: Record<string, FrontmatterValue> = {
        title: 'colon: comma, hash # brace } quote " bracket ]',
    };
    const round = parseDocument(serializeDocument(fm, "b"));
    assert.equal(round.frontmatter.title, fm.title);
});

test("round-trips an array whose items contain commas and quotes", () => {
    const fm: Record<string, FrontmatterValue> = { tags: ["a, b", 'c"d', "plain"] };
    const round = parseDocument(serializeDocument(fm, "b"));
    assert.deepEqual(round.frontmatter.tags, ["a, b", 'c"d', "plain"]);
});
