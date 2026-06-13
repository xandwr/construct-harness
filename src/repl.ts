/**
 * An interactive REPL over a {@link Session} — the harness's "talk to the
 * Construct" entrypoint.
 *
 * Reads lines from stdin, streams each reply to stdout token-by-token, and shows
 * tool activity and per-turn accounting as it happens. Slash commands handle the
 * session-level actions a transcript can't: `/reset`, `/history`, `/exit`.
 *
 * This is deliberately thin: all the conversation, memory, and streaming logic
 * lives in {@link Session}. The REPL only owns the terminal — reading input and
 * rendering the {@link LoopEvent} stream.
 */

import * as readline from "node:readline";
import { stdin as processStdin, stdout as processStdout } from "node:process";
import type { Readable } from "node:stream";
import { Session } from "./session.ts";
import type { LoopEvent } from "./bridge/loop.ts";
import { HarnessError } from "./bridge/errors.ts";

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

/** Render one streamed event. Text deltas print inline (the reply, as it
 *  arrives); everything else prints as a dim status line. */
function render(out: ReplOutput, event: LoopEvent): void {
    switch (event.kind) {
        case "text":
            out.write(event.text);
            break;
        case "tool_start":
            out.write(dim(out, `\n  ↳ ${event.name}(${compactArgs(event.args)})\n`));
            break;
        case "tool_end":
            out.write(dim(out, `  ↳ ${event.name} ${event.isError ? "errored" : "ok"}\n`));
            break;
        case "compacted":
            out.write(dim(out, `\n  [compacted history]\n`));
            break;
        // thinking, tool_call_start/args, turn_start, loop_done: not surfaced
        // in the basic REPL view.
    }
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
            let next = await turn.next();
            while (!next.done) {
                render(out, next.value);
                next = await turn.next();
            }
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

/** Handle a slash command. Returns true when the REPL should exit. */
function handleCommand(out: ReplOutput, input: string, session: Session): boolean {
    const [cmd] = input.slice(1).split(/\s+/);
    switch (cmd) {
        case "exit":
        case "quit":
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
            out.write(dim(out, "/reset clear history · /history count · /exit quit\n"));
            return false;
        default:
            out.write(dim(out, `Unknown command: /${cmd}. Try /help.\n`));
            return false;
    }
}
