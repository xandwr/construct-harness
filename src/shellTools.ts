/**
 * Bridges the *local user shell* into the agentic loop.
 *
 * The harness already gives a Construct a sandboxed scratchpad: Anthropic's
 * provider-hosted `code_execution` tool (see {@link ServerToolName}), which runs
 * server-side in a throwaway container with no reach back to the human's machine.
 * This is the opposite primitive: a custom {@link ToolDef} the *loop* dispatches,
 * which runs a command on the real local shell, in the real working directory,
 * with the harness process's own privileges. Where `code_execution` is isolated
 * by design, this is deliberately unguarded: the Construct can read the user's
 * files, run their tooling, and change their machine, exactly as the user could
 * at their own prompt.
 *
 * The two coexist: a Construct can reach for the sandbox when it just needs to
 * compute something disposably, and for {@link shellTools}'s `use__user__shell`
 * when the work has to touch the actual environment (run the project's tests,
 * inspect a real file, drive a CLI). One is server-hosted and stateless; this one
 * is local and load-bearing.
 *
 * Like every other tool module here, it speaks plain JSON in and out (its `run`
 * result drops straight into a `tool_result` part) and never throws out of `run`:
 * a command that fails, times out, or can't be spawned comes back as a structured
 * result the model can read and react to, not an exception that crashes the loop.
 */

import { spawn } from "node:child_process";
import type { ToolDef } from "./types.ts";

/** The tool name the model sees. The doubled underscores read as a namespace
 *  ("use the user's shell") and keep it visually distinct from the snake_case
 *  store tools (memory_save, goal_set, …): this one reaches *outside* the
 *  harness, and the name should look like it. */
export const USER_SHELL_TOOL = "use__user__shell";

/** Default ceiling on how long one command may run before it's killed, in ms.
 *  A command that hangs (waits on stdin, polls forever) must not wedge the turn;
 *  the model can raise it per-call via `timeout_ms` for a known-slow build. */
export const DEFAULT_SHELL_TIMEOUT_MS = 120_000;

/** Cap on captured stdout/stderr returned to the model, in characters. A command
 *  that prints a whole file or a megabyte of logs would otherwise blow the turn's
 *  context budget; we keep the tail (where errors and a command's final output
 *  usually are) and mark the truncation so the model knows it isn't the whole
 *  stream. The process still runs to completion; only the *returned* text is
 *  bounded. */
export const SHELL_OUTPUT_CAP = 30_000;

/** The structured result `use__user__shell` returns. Plain data so it serializes
 *  straight into a `tool_result`; `ran` is the single bit a caller keys off to
 *  tell "the command executed (whatever its exit code)" from "it never started". */
export interface ShellResult {
    /** True when the command was spawned and ran to completion (or was killed by
     *  the timeout); false when it couldn't be launched at all (e.g. the shell
     *  binary is missing). On false, `error` explains and the streams are empty. */
    ran: boolean;
    /** The command's exit code, or null when it was terminated by a signal
     *  (including the timeout kill) or never started. */
    exitCode: number | null;
    /** The signal that terminated the command, if any (e.g. "SIGTERM" from the
     *  timeout, "SIGINT"). Null on a normal exit. */
    signal: string | null;
    /** Captured standard output, tail-truncated to {@link SHELL_OUTPUT_CAP}. */
    stdout: string;
    /** Captured standard error, tail-truncated to {@link SHELL_OUTPUT_CAP}. */
    stderr: string;
    /** True when the command was killed for exceeding its timeout, so the model
     *  reads a non-zero exit as "too slow", not "it failed". */
    timedOut: boolean;
    /** A spawn-level failure message (shell missing, cwd doesn't exist), set only
     *  when `ran` is false. A non-zero exit is not an error here: it's a normal
     *  result with `ran: true` and the code, the way a shell reports it. */
    error?: string;
}

/** Narrow an unknown args bag to a record without trusting its fields yet, the
 *  same guard the other tool modules use. */
function asRecord(args: unknown): Record<string, unknown> {
    return typeof args === "object" && args !== null ? (args as Record<string, unknown>) : {};
}

/** Keep the last {@link SHELL_OUTPUT_CAP} characters of a stream, prefixing a
 *  marker when we dropped anything so the model never mistakes a tail for the
 *  whole output. The tail (not the head) is kept because a command's verdict
 *  (the failing assertion, the final summary line) lands at the end. */
function capTail(text: string): string {
    if (text.length <= SHELL_OUTPUT_CAP) return text;
    const kept = text.slice(text.length - SHELL_OUTPUT_CAP);
    const dropped = text.length - SHELL_OUTPUT_CAP;
    return `…[${dropped} earlier character${dropped === 1 ? "" : "s"} truncated]\n${kept}`;
}

/** The shell to run commands through, so pipes, redirects, globs, `&&`, and the
 *  user's own shell builtins all work as typed. Honors $SHELL (the user's login
 *  shell) when set, falling back to a POSIX `sh` that's present essentially
 *  everywhere. */
function loginShell(): string {
    return process.env.SHELL || "/bin/sh";
}

/** Options for {@link shellTools}. */
export interface ShellToolsOptions {
    /** Default timeout in ms for a command that doesn't set its own. Defaults to
     *  {@link DEFAULT_SHELL_TIMEOUT_MS}. A per-call `timeout_ms` overrides it. */
    defaultTimeoutMs?: number;
    /** Working directory commands run in when a call doesn't pass `cwd`. Defaults
     *  to the harness process's cwd, so a command runs where the user launched the
     *  Construct: the same place their own shell would start. */
    defaultCwd?: string;
}

/**
 * Build the local-shell tool set: a single `use__user__shell` tool the loop
 * dispatches to run a command on the user's actual machine.
 *
 * This is intentionally the unguarded counterpart to the sandboxed
 * `code_execution` server tool: no allow-list, no confinement, the harness
 * process's full privileges. Wire it into a {@link Session}'s `tools` (alongside
 * the store tools and, if you like, the `code_execution` server tool) to give a
 * Construct both a disposable sandbox and a real shell.
 *
 * Returns an array (one tool today) to match the shape of every other `*Tools`
 * factory here, so a caller spreads it into `tools` the same way.
 */
export function shellTools(options: ShellToolsOptions = {}): ToolDef[] {
    const defaultTimeoutMs = options.defaultTimeoutMs ?? DEFAULT_SHELL_TIMEOUT_MS;
    const defaultCwd = options.defaultCwd;

    const shell: ToolDef = {
        name: USER_SHELL_TOOL,
        description:
            "Run a shell command on the user's LOCAL machine and return its stdout, " +
            "stderr, and exit code. This is the user's real environment (their " +
            "files, their tools, their working directory), not a sandbox: use it to " +
            "run the project's tests, read or edit real files, drive a CLI, or inspect " +
            "the actual system. The command runs through the user's shell, so pipes, " +
            "redirects, globs, and `&&` work as written. For disposable computation " +
            "that doesn't need to touch this machine, prefer the sandboxed code " +
            "execution tool instead. A non-zero exit code is returned as data, not an " +
            "error: read it and react.",
        parameters: {
            type: "object",
            properties: {
                command: {
                    type: "string",
                    description:
                        "The shell command to run, exactly as you'd type it at a " +
                        "prompt (a full command line, e.g. 'npm test' or " +
                        "'grep -n TODO src/*.ts | head').",
                },
                cwd: {
                    type: "string",
                    description:
                        "Optional working directory to run the command in. Defaults " +
                        "to where the Construct was launched.",
                },
                timeout_ms: {
                    type: "number",
                    description:
                        "Optional per-command timeout in milliseconds. The command " +
                        "is killed if it runs longer (its partial output is still " +
                        `returned). Defaults to ${defaultTimeoutMs}.`,
                },
            },
            required: ["command"],
        },
        async run(args) {
            const a = asRecord(args);
            const command = a.command;
            if (typeof command !== "string" || command.trim() === "") {
                return shellError("command must be a non-empty string");
            }
            const cwd = typeof a.cwd === "string" && a.cwd.trim() ? a.cwd : defaultCwd;
            const timeoutMs =
                typeof a.timeout_ms === "number" &&
                Number.isFinite(a.timeout_ms) &&
                a.timeout_ms > 0
                    ? a.timeout_ms
                    : defaultTimeoutMs;
            return runCommand(command, { cwd, timeoutMs });
        },
    };

    return [shell];
}

/** Build the spawn-failure result shape, the one path where `ran` is false. */
function shellError(message: string): ShellResult {
    return {
        ran: false,
        exitCode: null,
        signal: null,
        stdout: "",
        stderr: "",
        timedOut: false,
        error: message,
    };
}

/**
 * Spawn one command through the user's shell and resolve a {@link ShellResult}
 * when it finishes (or is killed by the timeout). Never rejects: a spawn failure
 * resolves as `ran: false` so {@link ToolDef.run} hands the model a structured
 * result rather than throwing into the loop.
 *
 * The timeout sends SIGTERM and, after a short grace period, SIGKILL, so a
 * process ignoring the polite signal can't keep the turn alive; the partial
 * output captured up to that point is still returned.
 */
function runCommand(
    command: string,
    opts: { cwd?: string; timeoutMs: number },
): Promise<ShellResult> {
    return new Promise((resolve) => {
        // `-c` runs the command string through the shell, so the model's pipes,
        // redirects, and builtins behave as typed. We inherit the process env so
        // PATH and the user's exports are exactly what their own shell would see.
        const child = spawn(loginShell(), ["-c", command], {
            cwd: opts.cwd,
            env: process.env,
            // We capture stdout/stderr; stdin is closed (no interactive prompt can
            // wedge the turn waiting on input that will never come).
            stdio: ["ignore", "pipe", "pipe"],
        });

        let stdout = "";
        let stderr = "";
        let timedOut = false;
        let settled = false;

        child.stdout?.on("data", (chunk: Buffer) => {
            stdout += chunk.toString("utf8");
        });
        child.stderr?.on("data", (chunk: Buffer) => {
            stderr += chunk.toString("utf8");
        });

        // Escalate on timeout: a polite SIGTERM, then SIGKILL if it's still alive
        // a moment later. `.unref()` so this timer can't itself keep the harness
        // process from exiting.
        const killTimer = setTimeout(() => {
            timedOut = true;
            child.kill("SIGTERM");
            const hard = setTimeout(() => child.kill("SIGKILL"), 2_000);
            hard.unref?.();
        }, opts.timeoutMs);
        killTimer.unref?.();

        const finish = (result: ShellResult) => {
            if (settled) return;
            settled = true;
            clearTimeout(killTimer);
            resolve(result);
        };

        // A spawn-level failure (shell binary missing, cwd doesn't exist) fires
        // `error` and never `close`; surface it as the one `ran: false` path.
        child.on("error", (err) => {
            finish(shellError(err instanceof Error ? err.message : String(err)));
        });

        child.on("close", (code, signal) => {
            finish({
                ran: true,
                exitCode: code,
                signal: signal ?? null,
                stdout: capTail(stdout),
                stderr: capTail(stderr),
                timedOut,
            });
        });
    });
}
