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

/** Read a prompt file sitting one level up from `src/` (the repo root) and
 *  return it trimmed, so a trailing newline in the file doesn't ride into the
 *  system turn. */
function loadPrompt(filename: string): string {
    const path = fileURLToPath(new URL(`../${filename}`, import.meta.url));
    return readFileSync(path, "utf8").trim();
}

/** The full Construct persona: every tool group the server wires in (memory,
 *  goals, transcript_recall, dream_recall, the knowledge base, and both the
 *  sandboxed and local code paths). Used by the HTTP server. */
export const SYSTEM_PROMPT = loadPrompt("SYSTEM.md");

/** The leaner persona for the CLI, which ships only memory, goals, and the local
 *  shell. Kept honest about its smaller tool set rather than reusing the full
 *  prompt and naming tools the CLI doesn't have. */
export const CLI_SYSTEM_PROMPT = loadPrompt("SYSTEM.cli.md");
