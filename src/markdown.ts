/**
 * A small, dependency-free Markdown + LaTeX renderer for the terminal.
 *
 * This is deliberately *not* a Markdown parser. The REPL streams replies
 * line-by-line, so we only need to render constructs that fit on (or are
 * delimited by) a single line: ATX headings, list items, blockquotes, fenced
 * code blocks, horizontal rules, and the inline spans within a line (bold,
 * italic, inline code, links, and `$…$` / `$$…$$` math). That keeps the whole
 * thing to a few focused passes instead of a tree, and it composes naturally
 * with line-buffered streaming: feed each completed line to {@link
 * MarkdownRenderer.line} as its newline arrives.
 *
 * The one piece of cross-line state is the fenced code block (```), which spans
 * lines and suppresses inline formatting in between; {@link makeMarkdownRenderer}
 * owns that bit.
 *
 * LaTeX in a terminal can only ever be an approximation: there is no 2D layout,
 * so `\frac{a}{b}` becomes `a/b` and `\sum` becomes `∑`. {@link latexToUnicode}
 * does best-effort token substitution (Greek, super/subscripts, common
 * operators, simple fractions) and leaves anything it doesn't recognize as
 * readable source. This trades fidelity for zero dependencies, on purpose.
 *
 * All styling is gated by the `styled` flag the renderer is constructed with, so
 * a non-TTY sink (a pipe, or a test buffer) gets clean plain text with the
 * Markdown punctuation stripped but no ANSI escapes.
 */

// ── ANSI ────────────────────────────────────────────────────────────────────

const ESC = "\x1b[";
const RESET = `${ESC}0m`;

/** ANSI SGR codes we use. Kept tiny and named so the renderer reads clearly. */
const SGR = {
    bold: 1,
    dim: 2,
    italic: 3,
    underline: 4,
} as const;

/** Wrap `text` in an SGR code, but only when styling is on. The reset is a full
 *  `0m` so nested spans never leak attributes past their close. */
function sgr(on: boolean, code: number, text: string): string {
    return on ? `${ESC}${code}m${text}${RESET}` : text;
}

// ── LaTeX → Unicode ──────────────────────────────────────────────────────────

/** `\command` → Unicode. Longest names first doesn't matter here since we match
 *  on word boundaries, but greek + common math operators cover the bulk of what
 *  shows up in prose. Kept as a dense table; one entry per line would be noise. */
// prettier-ignore
const LATEX_COMMANDS: Record<string, string> = {
    // Lowercase Greek
    alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", zeta: "ζ",
    eta: "η", theta: "θ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ",
    nu: "ν", xi: "ξ", pi: "π", rho: "ρ", sigma: "σ", tau: "τ",
    upsilon: "υ", phi: "φ", chi: "χ", psi: "ψ", omega: "ω", varphi: "ϕ",
    varepsilon: "ε", vartheta: "ϑ",
    // Uppercase Greek
    Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π",
    Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
    // Operators and relations
    sum: "∑", prod: "∏", int: "∫", oint: "∮", partial: "∂", nabla: "∇",
    infty: "∞", pm: "±", mp: "∓", times: "×", div: "÷", cdot: "·",
    ast: "∗", star: "⋆", circ: "∘", bullet: "•",
    leq: "≤", le: "≤", geq: "≥", ge: "≥", neq: "≠", ne: "≠",
    approx: "≈", equiv: "≡", sim: "∼", simeq: "≃", cong: "≅", propto: "∝",
    ll: "≪", gg: "≫",
    to: "→", rightarrow: "→", leftarrow: "←", leftrightarrow: "↔",
    Rightarrow: "⇒", Leftarrow: "⇐", Leftrightarrow: "⇔", mapsto: "↦",
    uparrow: "↑", downarrow: "↓",
    in: "∈", notin: "∉", ni: "∋", subset: "⊂", supset: "⊃",
    subseteq: "⊆", supseteq: "⊇", cup: "∪", cap: "∩",
    emptyset: "∅", varnothing: "∅", setminus: "∖",
    forall: "∀", exists: "∃", nexists: "∄", neg: "¬", lnot: "¬",
    land: "∧", wedge: "∧", lor: "∨", vee: "∨",
    therefore: "∴", because: "∵",
    sqrt: "√", angle: "∠", perp: "⊥", parallel: "∥",
    cdots: "⋯", ldots: "…", dots: "…", vdots: "⋮", ddots: "⋱",
    aleph: "ℵ", hbar: "ℏ", ell: "ℓ", Re: "ℜ", Im: "ℑ", wp: "℘",
    nabla_: "∇", deg: "°", prime: "′",
    // Blackboard-ish (single letters; \mathbb{...} handled separately)
    mathbb: "", // placeholder, real handling below
};

/** Superscript glyphs for digits and a few symbols, for `^{…}` / `^x`. */
// prettier-ignore
const SUPERSCRIPT: Record<string, string> = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵",
    "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻",
    "=": "⁼", "(": "⁽", ")": "⁾", n: "ⁿ", i: "ⁱ", a: "ᵃ", b: "ᵇ",
    c: "ᶜ", d: "ᵈ", e: "ᵉ", x: "ˣ", y: "ʸ",
};

/** Subscript glyphs for digits and a few symbols, for `_{…}` / `_x`. */
// prettier-ignore
const SUBSCRIPT: Record<string, string> = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅",
    "6": "₆", "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋",
    "=": "₌", "(": "₍", ")": "₎", a: "ₐ", e: "ₑ", i: "ᵢ", j: "ⱼ",
    o: "ₒ", x: "ₓ", n: "ₙ", t: "ₜ",
};

/** Blackboard-bold letters for `\mathbb{R}` etc. */
// prettier-ignore
const BLACKBOARD: Record<string, string> = {
    A: "𝔸", B: "𝔹", C: "ℂ", D: "𝔻", E: "𝔼", F: "𝔽", G: "𝔾", H: "ℍ",
    I: "𝕀", J: "𝕁", K: "𝕂", L: "𝕃", M: "𝕄", N: "ℕ", O: "𝕆", P: "ℙ",
    Q: "ℚ", R: "ℝ", S: "𝕊", T: "𝕋", U: "𝕌", V: "𝕍", W: "𝕎", X: "𝕏",
    Y: "𝕐", Z: "ℤ",
};

/** Map each char of `s` through a sub/superscript table, or null if any char is
 *  unmapped (so the caller can fall back to readable source like `x^(ij)`). */
function mapScript(s: string, table: Record<string, string>): string | null {
    let out = "";
    for (const ch of s) {
        const g = table[ch];
        if (g === undefined) return null;
        out += g;
    }
    return out;
}

/**
 * Best-effort conversion of a LaTeX math fragment to Unicode. Lossy by nature:
 * recognized tokens become glyphs, everything else is left as readable source.
 * Operates on the inside of `$…$` / `$$…$$` (delimiters already stripped).
 */
export function latexToUnicode(src: string): string {
    let s = src;

    // \mathbb{R} → ℝ (and \mathbb R as a fallback).
    s = s.replace(/\\mathbb\s*\{([A-Z])\}/g, (_, c) => BLACKBOARD[c] ?? c);
    s = s.replace(/\\mathbb\s+([A-Z])/g, (_, c) => BLACKBOARD[c] ?? c);

    // \frac{a}{b} → a/b, parenthesizing compound numerators/denominators so the
    // result stays unambiguous (\frac{a+b}{c} → (a+b)/c).
    s = s.replace(/\\(?:t|d)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, (_, a, b) => {
        const wrap = (x: string) => (/[+\-*/ ]/.test(x.trim()) ? `(${x.trim()})` : x.trim());
        return `${wrap(a)}/${wrap(b)}`;
    });

    // \sqrt{x} → √(x), \sqrt x → √x.
    s = s.replace(/\\sqrt\s*\{([^{}]*)\}/g, (_, x) => `√(${x})`);

    // Named commands: \alpha, \sum, … . Word-boundary so \tau isn't eaten by \t.
    // An unrecognized command keeps its name, and if it has a `{arg}` we unwrap
    // it to `\name arg` so the later brace-strip can't glue them (\widehat{x} →
    // \widehat x, not \widehatx).
    s = s.replace(/\\([A-Za-z]+)(\s*\{([^{}]*)\})?/g, (whole, name: string, _g, arg) => {
        const glyph = LATEX_COMMANDS[name];
        if (glyph !== undefined && glyph !== "")
            return glyph + (arg !== undefined ? ` ${arg}` : "");
        return arg !== undefined ? `\\${name} ${arg}` : `\\${name}`;
    });

    // Superscripts: ^{...} or ^x. Fall back to ^(...) when a glyph is missing.
    // The single-char form excludes `(` / `{` so it never re-eats a braced
    // fallback's own `^(` (which would corrupt x^{z} → x^(z) into x⁽z)).
    s = s.replace(/\^\{([^{}]*)\}/g, (_, body) => mapScript(body, SUPERSCRIPT) ?? `^(${body})`);
    s = s.replace(/\^([^\s({])/g, (_, ch) => mapScript(ch, SUPERSCRIPT) ?? `^${ch}`);

    // Subscripts: _{...} or _x.
    s = s.replace(/_\{([^{}]*)\}/g, (_, body) => mapScript(body, SUBSCRIPT) ?? `_(${body})`);
    s = s.replace(/_([^\s({])/g, (_, ch) => mapScript(ch, SUBSCRIPT) ?? `_${ch}`);

    // Drop any remaining grouping braces and collapse the runs of spaces LaTeX
    // ignores anyway.
    s = s.replace(/[{}]/g, "");
    s = s.replace(/\s+/g, " ").trim();
    return s;
}

// ── Inline spans ─────────────────────────────────────────────────────────────

/**
 * Render inline Markdown + math within a single line. Order matters: math is
 * extracted first (so `$a_b$` isn't mauled by emphasis rules), then code spans
 * (which suppress all other formatting), then emphasis and links.
 */
function renderInline(text: string, styled: boolean): string {
    // 1. Math: $$…$$ first (greedy would cross $…$ boundaries), then $…$.
    //    A `\$` is a literal dollar, not a delimiter.
    let s = text.replace(/(?<!\\)\$\$([^$]+?)\$\$/g, (_, m) => latexToUnicode(m));
    s = s.replace(/(?<!\\)\$([^$\n]+?)\$/g, (_, m) => latexToUnicode(m));
    s = s.replace(/\\\$/g, "$");

    // 2. Inline code `…`: dim, and protected from later passes via placeholders.
    //    The sentinels are control chars that can't occur in a line of prose, so
    //    a placeholder never collides with real text and adjacent spans (\x00 0
    //    \x01\x00 1 \x01) split cleanly.
    const codes: string[] = [];
    s = s.replace(/`([^`]+)`/g, (_, code) => {
        codes.push(sgr(styled, SGR.dim, code));
        return ` ${codes.length - 1}`;
    });

    // 3. Links [text](url): keep the text, drop the URL (dim it when styled).
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) =>
        styled ? `${label} ${sgr(true, SGR.dim, `(${url})`)}` : `${label} (${url})`,
    );

    // 4. Bold then italic. Bold first so ** isn't consumed as two * .
    s = s.replace(/\*\*([^*]+)\*\*/g, (_, m) => sgr(styled, SGR.bold, m));
    s = s.replace(/__([^_]+)__/g, (_, m) => sgr(styled, SGR.bold, m));
    s = s.replace(/(?<![*\w])\*([^*\n]+)\*(?!\w)/g, (_, m) => sgr(styled, SGR.italic, m));
    s = s.replace(/(?<![_\w])_([^_\n]+)_(?!\w)/g, (_, m) => sgr(styled, SGR.italic, m));

    // 5. Restore code spans.
    s = s.replace(/ (\d+)/g, (_, i) => codes[Number(i)]);
    return s;
}

// ── Block (per-line) ─────────────────────────────────────────────────────────

/** A renderer with one bit of cross-line state: whether we're inside a fenced
 *  code block, where inline formatting is suppressed. */
export interface MarkdownRenderer {
    /** Render one source line (no trailing newline) to a display line. */
    line(src: string): string;
}

/** Construct a per-turn Markdown renderer. `styled` enables ANSI; pass the
 *  TTY-ness of the sink. */
export function makeMarkdownRenderer(styled: boolean): MarkdownRenderer {
    let inFence = false;

    return {
        line(src: string): string {
            // Fenced code: ``` toggles the block; content in between is dimmed
            // verbatim, with no inline parsing.
            const fence = /^\s*```/.test(src);
            if (fence) {
                inFence = !inFence;
                // The fence line itself: show the (optional) language label dim.
                return sgr(styled, SGR.dim, src);
            }
            if (inFence) return sgr(styled, SGR.dim, src);

            // ATX heading: #..###### . Render bold + underline, drop the hashes.
            const heading = /^(#{1,6})\s+(.*)$/.exec(src);
            if (heading) {
                const body = renderInline(heading[2], styled);
                return styled ? `${ESC}${SGR.bold};${SGR.underline}m${body}${RESET}` : body;
            }

            // Horizontal rule: ---, ***, ___ (3+).
            if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(src)) {
                return sgr(styled, SGR.dim, "─".repeat(40));
            }

            // Blockquote: > text → dim bar + rendered text.
            const quote = /^(\s*)>\s?(.*)$/.exec(src);
            if (quote) {
                return `${quote[1]}${sgr(styled, SGR.dim, "│")} ${renderInline(quote[2], styled)}`;
            }

            // Unordered list: -, *, + → • , preserving indentation.
            const ul = /^(\s*)[-*+]\s+(.*)$/.exec(src);
            if (ul) {
                return `${ul[1]}${sgr(styled, SGR.bold, "•")} ${renderInline(ul[2], styled)}`;
            }

            // Ordered list: 1. text → keep the number, render the body.
            const ol = /^(\s*)(\d+)\.\s+(.*)$/.exec(src);
            if (ol) {
                return `${ol[1]}${sgr(styled, SGR.bold, `${ol[2]}.`)} ${renderInline(ol[3], styled)}`;
            }

            // Plain prose (including blank lines).
            return renderInline(src, styled);
        },
    };
}
