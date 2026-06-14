/**
 * Tests for the terminal Markdown + LaTeX renderer ({@link makeMarkdownRenderer},
 * {@link latexToUnicode}).
 *
 * Two output modes matter: plain (non-TTY — punctuation stripped, no ANSI, what
 * a pipe or a test buffer sees) and styled (ANSI escapes present). Most asserts
 * use plain mode so they read clearly; a few check that styling actually wraps
 * the right content.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { makeMarkdownRenderer, latexToUnicode } from "../src/markdown.ts";

/** Render one line in plain (non-styled) mode. A fresh renderer each call, since
 *  callers that need fence state build their own. */
const plain = (src: string): string => makeMarkdownRenderer(false).line(src);
/** Render one line in styled (ANSI) mode. */
const styled = (src: string): string => makeMarkdownRenderer(true).line(src);

const ESC = "\x1b[";

// ── Inline spans (plain) ──────────────────────────────────────────────────────

test("bold and italic punctuation is stripped in plain mode", () => {
    assert.equal(plain("a **bold** word"), "a bold word");
    assert.equal(plain("a __bold__ word"), "a bold word");
    assert.equal(plain("an *italic* word"), "an italic word");
    assert.equal(plain("an _italic_ word"), "an italic word");
});

test("italic underscores inside a word are left alone", () => {
    // snake_case must not become snakecase.
    assert.equal(plain("call some_function_name now"), "call some_function_name now");
});

test("inline code keeps its content, drops the backticks", () => {
    assert.equal(plain("run `npm test` please"), "run npm test please");
});

test("inline code is protected from emphasis parsing", () => {
    // The * inside the code span must not start an italic run.
    assert.equal(plain("use `a*b` here"), "use a*b here");
});

test("adjacent code spans both survive (distinct placeholders)", () => {
    assert.equal(plain("`one``two`"), "onetwo");
    assert.equal(plain("`a` and `b`"), "a and b");
});

test("links keep the label and show the url", () => {
    assert.equal(plain("see [the docs](https://x.io)"), "see the docs (https://x.io)");
});

// ── Block constructs (plain) ──────────────────────────────────────────────────

test("ATX headings drop the hashes", () => {
    assert.equal(plain("# Title"), "Title");
    assert.equal(plain("### Sub heading"), "Sub heading");
});

test("unordered list markers become bullets", () => {
    assert.equal(plain("- first"), "• first");
    assert.equal(plain("* second"), "• second");
    assert.equal(plain("  + nested"), "  • nested");
});

test("ordered list keeps the number", () => {
    assert.equal(plain("1. first"), "1. first");
    assert.equal(plain("  3. third"), "  3. third");
});

test("blockquote gets a bar and renders its body", () => {
    assert.equal(plain("> quoted **text**"), "│ quoted text");
});

test("horizontal rule becomes a line", () => {
    assert.match(plain("---"), /^─+$/);
    assert.match(plain("***"), /^─+$/);
});

test("a bare number-dot in prose is not treated as a list", () => {
    // Only a line that *starts* with "N. " is a list item.
    assert.equal(plain("version 2. final"), "version 2. final");
});

// ── Fenced code blocks (stateful) ─────────────────────────────────────────────

test("fenced code suppresses inline formatting between the fences", () => {
    const md = makeMarkdownRenderer(false);
    assert.equal(md.line("```ts"), "```ts"); // opening fence, plain
    assert.equal(md.line("const x = `a`;"), "const x = `a`;"); // verbatim, backticks kept
    assert.equal(md.line("y **not bold**"), "y **not bold**"); // verbatim
    assert.equal(md.line("```"), "```"); // closing fence
    assert.equal(md.line("now **bold**"), "now bold"); // formatting resumes
});

// ── LaTeX → Unicode ───────────────────────────────────────────────────────────

test("inline math: greek and superscripts", () => {
    assert.equal(plain("energy is $E = mc^2$ today"), "energy is E = mc² today");
    assert.equal(plain("angle $\\alpha + \\beta$"), "angle α + β");
});

test("block math $$…$$ renders too", () => {
    // \sum → ∑, _{i=0} → ᵢ₌₀ (all chars have subscript glyphs), ^n → ⁿ.
    assert.equal(plain("$$\\sum_{i=0}^n i$$"), "∑ᵢ₌₀ⁿ i");
});

test("latexToUnicode: operators and relations", () => {
    assert.equal(latexToUnicode("a \\leq b \\to c"), "a ≤ b → c");
    assert.equal(latexToUnicode("x \\in \\mathbb{R}"), "x ∈ ℝ");
    assert.equal(latexToUnicode("\\forall x \\exists y"), "∀ x ∃ y");
});

test("latexToUnicode: fractions, parenthesizing compound parts", () => {
    assert.equal(latexToUnicode("\\frac{1}{2}"), "1/2");
    assert.equal(latexToUnicode("\\frac{a+b}{c}"), "(a+b)/c");
});

test("latexToUnicode: sqrt", () => {
    assert.equal(latexToUnicode("\\sqrt{x}"), "√(x)");
});

test("latexToUnicode: unknown commands are left as readable source", () => {
    // No table entry for \widehat — keep it rather than mangle it.
    assert.equal(latexToUnicode("\\widehat{x}"), "\\widehat x");
});

test("latexToUnicode: unmapped script chars fall back to readable form", () => {
    // 'z' has no superscript glyph, so x^z stays legible.
    assert.equal(latexToUnicode("x^{z}"), "x^(z)");
});

test("an escaped dollar is a literal, not a math delimiter", () => {
    assert.equal(plain("it costs \\$5 not $\\alpha$ dollars"), "it costs $5 not α dollars");
});

// ── Styled (ANSI) mode ────────────────────────────────────────────────────────

test("styled bold wraps the content in an ANSI code", () => {
    const out = styled("a **b** c");
    assert.ok(out.includes(`${ESC}1m`), "bold SGR present");
    assert.ok(out.includes("b"), "content present");
    assert.ok(out.includes(`${ESC}0m`), "reset present");
});

test("styled output of a plain line has no escapes", () => {
    assert.equal(styled("just prose"), "just prose");
});

test("styled headings carry bold+underline", () => {
    assert.match(styled("# Hi"), new RegExp(`${ESC.replace("[", "\\[")}1;4m`));
});
