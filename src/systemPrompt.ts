/**
 * The Construct's base system prompt, loaded from a markdown file at the repo
 * root rather than spelled out inline in an entrypoint.
 *
 * The prompt is the Construct's persona and the contract for how it uses its
 * tools (memory, goals, transcript, dreams, the knowledge base, the shell). That
 * is content a human reads and edits, not a string literal worth burying in
 * code, so it lives in `SYSTEM.md` (the full persona, every tool) and
 * `SYSTEM.cli.md` (the leaner CLI persona). Each entrypoint loads the one that
 * matches the tools it actually ships, so a prompt never advertises a tool the
 * surface doesn't wire in.
 *
 * Resolution is anchored to this module's own location via `import.meta.url`, not
 * the process CWD, so the prompt loads the same whether the harness is started
 * from the repo root, a parent directory, or as an installed dependency. The read
 * is synchronous and done once at module load: the prompt is small, needed before
 * the first turn, and never changes within a run.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

/**
 * Read a prompt file sitting one level up from `src/` (the repo root) and reflow
 * it from legible, hard-wrapped markdown into the prose the model sees.
 *
 * The files are wrapped at ~80-100 chars so they read cleanly on disk and diff
 * sanely, but those line breaks are presentational, not semantic. We apply the
 * standard markdown convention: a single newline is a soft wrap (it rejoins into
 * a space) and a blank line is a real paragraph break (preserved). So the prompt
 * the model receives is a handful of flowing paragraphs regardless of where the
 * source happens to wrap, and rewrapping the file never changes what it sees.
 */
function loadPrompt(filename: string): string {
    const path = fileURLToPath(new URL(`../${filename}`, import.meta.url));
    return readFileSync(path, "utf8")
        .trim()
        // Split on blank lines into paragraphs, unwrap each paragraph's soft
        // line breaks back into single spaces, then rejoin with blank lines.
        .split(/\n\s*\n/)
        .map((para) => para.replace(/\s*\n\s*/g, " ").trim())
        .join("\n\n");
}

/** The full Construct persona: every tool group the server wires in (memory,
 *  goals, transcript_recall, dream_recall, the knowledge base, and both the
 *  sandboxed and local code paths). Used by the HTTP server. */
export const SYSTEM_PROMPT = loadPrompt("SYSTEM.md");

/** The leaner persona for the CLI, which ships only memory, goals, and the local
 *  shell. Kept honest about its smaller tool set rather than reusing the full
 *  prompt and naming tools the CLI doesn't have. */
export const CLI_SYSTEM_PROMPT = loadPrompt("SYSTEM.cli.md");
