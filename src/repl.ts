/**
 * An interactive REPL over a {@link Session}: the harness's "talk to the
 * Construct" entrypoint.
 *
 * Reads lines from stdin, streams each reply to stdout token-by-token, and shows
 * tool activity and per-turn accounting as it happens. Slash commands handle the
 * session-level actions a transcript can't: `/reset`, `/history`, `/exit`.
 *
 * This is deliberately thin: all the conversation, memory, and streaming logic
 * lives in {@link Session}. The REPL only owns the terminal: reading input and
 * rendering the {@link LoopEvent} stream.
 */

import * as readline from "node:readline";
import { stdin as processStdin, stdout as processStdout } from "node:process";
import type { Readable } from "node:stream";
import { Session } from "./session.ts";
import type { LoopEvent } from "./bridge/loop.ts";
import { HarnessError } from "./bridge/errors.ts";
import { makeMarkdownRenderer } from "./markdown.ts";
import { BUILTIN_COMMANDS, commandSignature, findCommand } from "./commands.ts";

/** The output sink the REPL writes to. `process.stdout` satisfies this; tests
 *  pass a buffer. `isTTY` gates ANSI styling so piped/redirected output stays
 *  plain. */
export interface ReplOutput {
    write(text: string): void;
    readonly isTTY?: boolean;
}

/** What {@link runRepl} needs from its environment, injectable for tests. */
export interface ReplDeps {
    /** Line source. Defaults to `process.stdin`. */
    input?: Readable;
    /** Output sink. Defaults to `process.stdout`. */
    output?: ReplOutput;
}

/** ANSI dim, for the unobtrusive status/tool lines. Disabled when not a TTY so
 *  piped output stays clean. */
function dim(out: ReplOutput, text: string): string {
    return out.isTTY ? `\x1b[2m${text}\x1b[0m` : text;
}

/**
 * A stateful renderer for one turn's {@link LoopEvent} stream.
 *
 * Text deltas are Markdown, rendered (with inline LaTeX) as the reply arrives;
 * tool and compaction activity print as dim status lines. Two problems get
 * solved here, both about boundaries:
 *
 *  1. **Markdown can't render mid-token.** You can't style `**bold**` until the
 *     closing `**` arrives. So we render *line-buffered*: text deltas accumulate
 *     in `pending`, and each time a `\n` completes a line we hand that line to
 *     the {@link makeMarkdownRenderer} and print the styled result. The partial
 *     last line stays buffered until its newline (or a boundary) arrives. This
 *     matches how the model streams — token by token, but lines settle quickly —
 *     so the reply still appears live, just a line at a time rather than a
 *     half-formatted character at a time.
 *
 *  2. **Segments must not run together.** A turn interleaves prose and tool
 *     blocks; without care they form one unspaced wall. This closure owns every
 *     boundary: entering a tool/compaction block first flushes and closes the
 *     current prose line, and resuming prose after a block emits a blank line so
 *     each segment reads as its own paragraph. {@link RenderState.finish} drains
 *     the buffer and leaves the cursor on a fresh line for the footer.
 */
interface RenderState {
    render(event: LoopEvent): void;
    /** Flush any buffered text and ensure the cursor is on its own line (call
     *  before the footer). */
    finish(): void;
}

function makeRenderer(out: ReplOutput): RenderState {
    const md = makeMarkdownRenderer(out.isTTY ?? false);
    // Buffered text for the line currently being streamed (no newline yet).
    let pending = "";
    // True while we have emitted prose with no trailing newline since the last
    // boundary, i.e. either `pending` is non-empty or a rendered line is open.
    let inText = false;
    // True once any text has been seen this turn, so the separating blank line
    // goes *between* paragraphs, not before the first one.
    let sawText = false;

    /** Render and emit a single completed source line (its `\n` is added). */
    function emitLine(src: string): void {
        out.write(md.line(src) + "\n");
    }

    /** Flush whole lines out of `pending`, keeping any trailing partial line
     *  buffered. */
    function flushCompleteLines(): void {
        let nl: number;
        while ((nl = pending.indexOf("\n")) !== -1) {
            emitLine(pending.slice(0, nl));
            pending = pending.slice(nl + 1);
            inText = false; // a full line was just terminated with its own \n
        }
        if (pending.length > 0) inText = true;
    }

    /** Flush a trailing partial line and close it, so a status block starts on a
     *  clean line. */
    function endText(): void {
        if (pending.length > 0) {
            emitLine(pending);
            pending = "";
        } else if (inText) {
            out.write("\n");
        }
        inText = false;
    }

    return {
        render(event: LoopEvent): void {
            switch (event.kind) {
                case "text":
                    // Resuming prose after a tool/compaction block: separate it
                    // from the block above with a blank line.
                    if (!inText && pending.length === 0 && sawText) out.write("\n");
                    sawText = true;
                    pending += event.text;
                    flushCompleteLines();
                    break;
                case "tool_start":
                    endText();
                    out.write(dim(out, `\n  ↳ ${event.name}(${compactArgs(event.args)})\n`));
                    break;
                case "tool_end":
                    endText();
                    out.write(dim(out, `  ↳ ${event.name} ${event.isError ? "errored" : "ok"}\n`));
                    break;
                case "compacted":
                    endText();
                    out.write(dim(out, `\n  [compacted history]\n`));
                    break;
                // thinking, tool_call_start/args, turn_start, loop_done: not
                // surfaced in the basic REPL view.
            }
        },
        finish(): void {
            endText();
        },
    };
}

/** One-line, length-bounded rendering of tool args for the status line. */
function compactArgs(args: unknown): string {
    let s: string;
    try {
        s = typeof args === "string" ? args : (JSON.stringify(args) ?? "");
    } catch {
        s = "…";
    }
    return s.length > 60 ? `${s.slice(0, 57)}…` : s;
}

/**
 * Run an interactive REPL until EOF or `/exit`. Resolves when the loop ends.
 *
 * Streams are injectable ({@link ReplDeps}) so the REPL is testable without a
 * real terminal; both default to the process std streams. The prompt is written
 * to the output sink rather than handed to readline so a non-TTY output (a test
 * buffer) still captures it.
 */
export async function runRepl(session: Session, deps: ReplDeps = {}): Promise<void> {
    const input = deps.input ?? processStdin;
    const out: ReplOutput = deps.output ?? processStdout;

    const rl = readline.createInterface({ input });
    out.write(dim(out, "Construct ready. Type a message, or /help.\n"));
    out.write("› ");

    for await (const line of rl) {
        const text = line.trim();
        if (text.length === 0) {
            out.write("› ");
            continue;
        }

        if (text.startsWith("/")) {
            const done = handleCommand(out, text, session);
            if (done) break;
            out.write("› ");
            continue;
        }

        try {
            // Drive the turn. The generator yields events (rendered live) and
            // returns the TurnResult, which carries the accounting footer.
            const turn = session.send(text);
            const renderer = makeRenderer(out);
            let next = await turn.next();
            while (!next.done) {
                renderer.render(next.value);
                next = await turn.next();
            }
            renderer.finish();
            const result = next.value;
            const u = result.usage;
            out.write(
                dim(
                    out,
                    `\n${footer(result.modelTurns, u.inputTokens, u.outputTokens, result.compactions, result.stoppedAtMaxTurns)}\n`,
                ),
            );
        } catch (err) {
            out.write(`\n[error] ${describeError(err)}\n`);
        }

        out.write("› ");
    }

    rl.close();
    out.write(dim(out, "\nBye.\n"));
}

/** A readable one-liner for a failed turn. A {@link HarnessError} surfaces its
 *  neutral kind so the user can tell a rate-limit from an auth failure; for a
 *  retryable kind that reached us, retries were already exhausted. */
function describeError(err: unknown): string {
    if (err instanceof HarnessError) {
        const exhausted = err.retryable ? " (retries exhausted)" : "";
        return `${err.kind}${exhausted}: ${err.message}`;
    }
    return err instanceof Error ? err.message : String(err);
}

function footer(
    turns: number,
    inTok: number,
    outTok: number,
    compactions: number,
    cutOff: boolean,
): string {
    const parts = [`${turns} turn(s)`, `${inTok} in / ${outTok} out tokens`];
    if (compactions) parts.push(`${compactions} compaction(s)`);
    if (cutOff) parts.push("cut off at maxTurns");
    return parts.join(" · ");
}

/**
 * Handle a slash command. Returns true when the REPL should exit.
 *
 * The set of commands and their descriptions live in {@link BUILTIN_COMMANDS}
 * (the same catalogue the web client lists in its `/` menu); the REPL resolves
 * the typed word against it so an unknown command is reported rather than run,
 * and `/help` prints the registry instead of a hand-kept string. What each
 * command *does* still lives here — execution is the surface's job — switched on
 * the resolved command's stable `name`.
 */
function handleCommand(out: ReplOutput, input: string, session: Session): boolean {
    const [word] = input.slice(1).split(/\s+/);
    const cmd = findCommand(word);
    if (!cmd) {
        out.write(dim(out, `Unknown command: /${word}. Try /help.\n`));
        return false;
    }
    switch (cmd.name) {
        case "exit":
            return true;
        case "reset":
            session.reset();
            out.write(dim(out, "History cleared.\n"));
            return false;
        case "history": {
            const turns = session.history();
            out.write(dim(out, `${turns.length} message(s) in history.\n`));
            return false;
        }
        case "help":
            out.write(dim(out, `${helpText()}\n`));
            return false;
        default:
            // A command in the registry the REPL doesn't act on (none today).
            out.write(dim(out, `/${cmd.name} isn't available here.\n`));
            return false;
    }
}

/** One-line help listing every built-in command's signature and description,
 *  rendered from {@link BUILTIN_COMMANDS} so the REPL and the registry never
 *  drift. */
function helpText(): string {
    return BUILTIN_COMMANDS.map((c) => `${commandSignature(c)} — ${c.description}`).join(" · ");
}
