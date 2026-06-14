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
 *
 * Because commands run through the user's *actual* login shell ($SHELL), the
 * dialect that works isn't always bash/POSIX: a fish user needs `set -x X 1` and
 * `a; and b`, not `export X=1` and `a && b`. So the tool description names the
 * live shell (and OS) and tells the model to write in that shell's syntax. We
 * don't translate or teach each dialect; we just hand the model the one fact it's
 * missing and let it infer the syntax it already knows. The name comes from the
 * same {@link loginShell} the command runs through, so guidance and execution
 * can't drift apart, and it adapts automatically as the user's $SHELL changes.
 */

import { spawn } from "node:child_process";
import { basename, resolve as resolvePath } from "node:path";
import { platform } from "node:os";
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
    /** The policy decision for this call: always present so every shell result
     *  carries an audit trail (the mode it ran under, and on a block, why). A
     *  blocked call has `ran: false`, `blocked: true`, and `error` set to the
     *  reason; an allowed one carries the mode it was permitted under. The loop
     *  logs the whole result in the `tool_result` event's meta, so this is the
     *  structured audit record of every local shell call. */
    policy: ShellPolicyDecision;
}

/** The audit record attached to every {@link ShellResult}: what the policy
 *  decided and under which mode. */
export interface ShellPolicyDecision {
    /** The effective policy mode the call ran (or was refused) under. */
    mode: ShellPolicyMode;
    /** True when the policy refused the command before it could run. */
    blocked: boolean;
    /** The human-readable reason a command was blocked (a denied pattern, a cwd
     *  outside the allowed roots, read-only mode). Undefined when allowed. */
    reason?: string;
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
function capTail(text: string, cap = SHELL_OUTPUT_CAP): string {
    if (text.length <= cap) return text;
    const kept = text.slice(text.length - cap);
    const dropped = text.length - cap;
    return `…[${dropped} earlier character${dropped === 1 ? "" : "s"} truncated]\n${kept}`;
}

/** The shell to run commands through, so pipes, redirects, globs, `&&`, and the
 *  user's own shell builtins all work as typed. Honors $SHELL (the user's login
 *  shell) when set, falling back to a POSIX `sh` that's present essentially
 *  everywhere. */
function loginShell(): string {
    return process.env.SHELL || "/bin/sh";
}

/** The shell's short name (its binary's basename, e.g. "fish", "zsh", "bash"),
 *  which is all the model needs to pick the right dialect. Derived from the same
 *  {@link loginShell} the command actually runs through, so what we *tell* the
 *  model and what we *execute on* can never drift apart. */
function shellName(): string {
    // basename strips the path and any version suffix is rare enough to ignore;
    // "/usr/bin/fish" → "fish", "/bin/sh" → "sh".
    return basename(loginShell());
}

/**
 * Describe the live shell environment for the tool description, so the model
 * authors commands in the *right* dialect instead of defaulting to bash/POSIX
 * and tripping over a non-POSIX shell. The classic trap is fish: `export X=1`,
 * `a && b`, and `$(...)` all differ (`set -x X 1`, `a; and b`, `(...)`), so a
 * bash-shaped command silently does the wrong thing or errors. We don't try to
 * teach the model each dialect; we just name the shell and OS and let it infer
 * the syntax it already knows, which is the whole point of this goal.
 *
 * Returns a sentence ready to append to the tool description, kept to one line
 * so it adds context without bloating the prompt.
 */
function shellEnvNote(): string {
    const name = shellName();
    const os = platform(); // "linux", "darwin", "win32", …
    return (
        `The user's shell is ${name} on ${os}; write the command in ${name} syntax ` +
        `(its own quoting, variables, and control operators), not assuming bash/POSIX. ` +
        `For example, ${name === "fish" ? "fish uses 'set -x VAR value' and 'a; and b', not 'export VAR=value' and 'a && b'" : "constructs like variable assignment, exports, and conditionals can differ between shells"}.`
    );
}

// ── Policy: make local shell power governable ─────────────────────────────────

/**
 * How permissive the local shell is. The default (`unrestricted`) is the
 * harness's historical behavior — the Construct can do anything the user could at
 * their own prompt. The other modes let a cautious operator dial that back without
 * removing the tool:
 *  - `unrestricted` — no policy gate beyond the deny patterns; the original power.
 *  - `restricted`   — deny patterns plus cwd-root confinement and the caps; run
 *                     real commands, but only within the allowed roots.
 *  - `read-only`    — additionally refuse any command that isn't recognizably a
 *                     read. The blunt-but-honest mode for "let it look, not touch".
 */
export type ShellPolicyMode = "unrestricted" | "restricted" | "read-only";

/**
 * The governance applied to every `use__user__shell` call before it spawns. All
 * fields have safe defaults; an unconfigured policy is `unrestricted` with the
 * built-in deny patterns and the standard caps. A blocked call never throws — it
 * returns a structured {@link ShellResult} with `blocked: true` and a reason, so
 * the model reads the refusal as data and the loop logs the audit record.
 */
export interface ShellPolicy {
    /** The permissiveness mode. Defaults to `unrestricted`. */
    mode?: ShellPolicyMode;
    /** Regexes that, if any matches the command, block it outright (in *every*
     *  mode, including unrestricted). Defaults to {@link DEFAULT_DENY_PATTERNS}: a
     *  short list of unambiguously destructive shapes (rm -rf /, fork bombs, raw
     *  disk writes). Pass your own to replace the defaults, or `[]` to disable. */
    denyPatterns?: RegExp[];
    /** Absolute directory roots a command's working directory must fall within
     *  (in `restricted`/`read-only` mode). A `cwd` outside every root is blocked.
     *  Empty/undefined means "no cwd confinement" (the default). */
    allowedCwdRoots?: string[];
    /** Hard ceiling on a call's timeout in ms: a per-call `timeout_ms` larger than
     *  this is clamped down to it, so the policy can cap how long any one command
     *  may wedge a turn. Undefined means no extra ceiling beyond the default. */
    maxTimeoutMs?: number;
    /** Override for the returned-output cap in characters. Undefined uses
     *  {@link SHELL_OUTPUT_CAP}. Lets a policy tighten how much a command can pour
     *  into the turn's context. */
    outputCap?: number;
}

/** The destructive command shapes blocked by default, in every mode. Deliberately
 *  conservative — a few unambiguous catastrophes (recursive root delete, fork
 *  bomb, raw disk overwrite, mkfs), not a sprawling blocklist that would give a
 *  false sense of safety while breaking legitimate commands. The real safety
 *  control is `read-only`/`restricted` mode and the cwd roots; this is a backstop
 *  against the worst typos and prompt-injections even when unrestricted. */
export const DEFAULT_DENY_PATTERNS: RegExp[] = [
    // rm -rf / (and /*), with flags in any order.
    /\brm\s+(-[a-z]*\s+)*-?[a-z]*[rf][a-z]*\s+(-[a-z]*\s+)*\/(\s|\*|$)/i,
    // A classic fork bomb.
    /:\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:/,
    // Raw write over a whole disk device, or formatting one.
    /\b(dd)\b[^\n]*\bof=\/dev\/(sd|nvme|hd|disk)/i,
    /\bmkfs(\.\w+)?\s+\/dev\//i,
    // Overwrite a block device from /dev/zero or /dev/urandom.
    />\s*\/dev\/(sd|nvme|hd|disk)/i,
];

/** Heuristic allow-list of read-only command leaders for `read-only` mode. A
 *  command whose first word (after any leading `sudo`/`env`) isn't one of these is
 *  refused. Blunt by design: it's a denylist-of-everything-else, so it errs toward
 *  refusing an unfamiliar command rather than guessing it's safe. The model can
 *  still be told what's allowed and pick a read instead. */
const READ_ONLY_COMMANDS = new Set([
    "ls",
    "cat",
    "head",
    "tail",
    "grep",
    "rg",
    "find",
    "fd",
    "stat",
    "file",
    "wc",
    "echo",
    "pwd",
    "which",
    "type",
    "env",
    "printenv",
    "date",
    "whoami",
    "id",
    "ps",
    "top",
    "df",
    "du",
    "uname",
    "hostname",
    "uptime",
    "git",
    "diff",
    "tree",
    "sort",
    "uniq",
    "cut",
    "awk",
    "sed",
    "jq",
    "less",
    "more",
    "node",
    "python",
    "python3",
    "npm",
    "cargo",
    "go",
    "tsc",
]);

/** Git subcommands that mutate, so `read-only` mode can allow `git status`/`log`
 *  but refuse `git push`/`commit`/`reset`. */
const GIT_WRITE_SUBCOMMANDS = new Set([
    "push",
    "commit",
    "merge",
    "rebase",
    "reset",
    "clean",
    "checkout",
    "switch",
    "restore",
    "rm",
    "mv",
    "add",
    "stash",
    "tag",
    "branch",
    "fetch",
    "pull",
    "clone",
    "apply",
    "cherry-pick",
    "revert",
    "gc",
    "prune",
    "config",
]);

/**
 * Decide whether a command is permitted under a policy, returning the audit
 * decision. Pure and synchronous: no spawn, no I/O. The order is deny-first (a
 * destructive pattern is refused in every mode), then mode-specific checks
 * (read-only's read allow-list, restricted's cwd confinement).
 */
export function evaluateShellPolicy(
    command: string,
    cwd: string | undefined,
    policy: ShellPolicy,
): ShellPolicyDecision {
    const mode = policy.mode ?? "unrestricted";
    const block = (reason: string): ShellPolicyDecision => ({ mode, blocked: true, reason });

    // 1. Deny patterns: the catastrophe backstop, enforced in every mode.
    const denyPatterns = policy.denyPatterns ?? DEFAULT_DENY_PATTERNS;
    for (const re of denyPatterns) {
        if (re.test(command)) {
            return block(`command matches a blocked pattern (${re.source})`);
        }
    }

    // 2. cwd confinement (restricted / read-only): the resolved cwd must sit
    //    within one of the allowed roots.
    if (mode !== "unrestricted" && policy.allowedCwdRoots && policy.allowedCwdRoots.length) {
        const target = resolvePath(cwd ?? process.cwd());
        const ok = policy.allowedCwdRoots.some((root) => isWithin(target, resolvePath(root)));
        if (!ok) {
            return block(`working directory ${target} is outside the allowed roots`);
        }
    }

    // 3. read-only: every command in the pipeline must read, not write.
    if (mode === "read-only" && !isReadOnlyCommand(command)) {
        return block("read-only mode: this command is not a recognized read");
    }

    return { mode, blocked: false };
}

/** True when `child` is the same as, or nested under, `root`. Both must already
 *  be absolute. Guards the path-prefix check against a sibling that merely shares
 *  a name prefix ("/srv/app" must not count "/srv/app-other" as within). */
function isWithin(child: string, root: string): boolean {
    if (child === root) return true;
    const base = root.endsWith("/") ? root : root + "/";
    return child.startsWith(base);
}

/**
 * Whether a command line is recognizably read-only: every command segment (split
 * on the shell operators that chain commands) leads with an allowed read. Blunt:
 * it refuses anything it doesn't recognize, and refuses outright on a redirect
 * that writes a file (`>`/`>>` to a path). Not a security boundary on its own (a
 * determined `read` tool can still mutate), but an honest "look, don't touch"
 * default that catches the obvious writes.
 */
export function isReadOnlyCommand(command: string): boolean {
    // A write redirect to a file is a mutation regardless of the leader. (We allow
    // `2>&1` and `>/dev/null` style fd/devnull redirects, which don't write files.)
    if (/(^|[^0-9>])>>?\s*(?!\/dev\/null|&)/.test(command)) return false;

    // Split into command segments on ; | && || and `and`/`or` (fish), then check
    // each leads with a read.
    const segments = command
        .split(/;|\|\||\||&&|(?:^|\s)and\s|(?:^|\s)or\s|&/)
        .map((s) => s.trim());
    for (const seg of segments) {
        if (!seg) continue;
        const words = seg.split(/\s+/);
        let i = 0;
        // Skip a leading sudo/env/command wrapper and inline VAR=val assignments.
        while (i < words.length && (/^[A-Za-z_]\w*=/.test(words[i]) || words[i] === "sudo")) i++;
        const leader = basename(words[i] ?? "");
        if (!READ_ONLY_COMMANDS.has(leader)) return false;
        // git is read-only only for non-mutating subcommands.
        if (leader === "git") {
            const sub = words.slice(i + 1).find((w) => !w.startsWith("-"));
            if (sub && GIT_WRITE_SUBCOMMANDS.has(sub)) return false;
        }
    }
    return true;
}

/**
 * Resolve a {@link ShellPolicy} from environment variables, for the server's
 * default wiring. Unset means `unrestricted` (the historical behavior — a change
 * here would silently break existing setups). Recognized vars:
 *  - `SHELL_POLICY`        = unrestricted | restricted | read-only
 *  - `SHELL_ALLOWED_ROOTS` = colon-separated absolute dirs (cwd confinement)
 *  - `SHELL_MAX_TIMEOUT_MS`= number, clamps any per-call timeout
 *  - `SHELL_OUTPUT_CAP`    = number, overrides the returned-output cap
 * A bad mode value degrades to `unrestricted` with a warning rather than
 * crashing, so a typo never takes the server down.
 */
export function resolveShellPolicy(env: NodeJS.ProcessEnv = process.env): ShellPolicy {
    const raw = (env.SHELL_POLICY ?? "").trim().toLowerCase();
    let mode: ShellPolicyMode = "unrestricted";
    if (raw === "restricted" || raw === "read-only" || raw === "readonly") {
        mode = raw === "readonly" ? "read-only" : (raw as ShellPolicyMode);
    } else if (raw !== "" && raw !== "unrestricted") {
        console.warn(`SHELL_POLICY: unknown mode "${raw}", defaulting to unrestricted`);
    }
    const roots = (env.SHELL_ALLOWED_ROOTS ?? "")
        .split(":")
        .map((s) => s.trim())
        .filter(Boolean);
    const maxTimeout = Number(env.SHELL_MAX_TIMEOUT_MS);
    const cap = Number(env.SHELL_OUTPUT_CAP);
    return {
        mode,
        allowedCwdRoots: roots.length ? roots : undefined,
        maxTimeoutMs: Number.isFinite(maxTimeout) && maxTimeout > 0 ? maxTimeout : undefined,
        outputCap: Number.isFinite(cap) && cap > 0 ? cap : undefined,
    };
}

/** A sentence for the tool description naming the active policy mode, so the model
 *  doesn't waste a turn discovering the restriction by being refused. Empty in
 *  unrestricted mode (the default), where there's nothing extra to say. */
function policyNote(mode: ShellPolicyMode): string {
    if (mode === "read-only") {
        return (
            " NOTE: this shell is in READ-ONLY mode — commands that write, delete, or " +
            "otherwise mutate the machine are refused. Use it only to inspect."
        );
    }
    if (mode === "restricted") {
        return (
            " NOTE: this shell runs under a restricted policy — commands must stay " +
            "within the allowed working directories, and some destructive commands are " +
            "blocked. A refused command comes back with the reason."
        );
    }
    return "";
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
    /** The governance applied before every command runs (see {@link ShellPolicy}).
     *  Omit for the historical unrestricted behavior. */
    policy?: ShellPolicy;
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
    const policy = options.policy ?? {};
    const mode = policy.mode ?? "unrestricted";
    const outputCap = policy.outputCap ?? SHELL_OUTPUT_CAP;

    const shell: ToolDef = {
        name: USER_SHELL_TOOL,
        description:
            "Run a shell command on the user's LOCAL machine and return its stdout, " +
            "stderr, and exit code. This is the user's real environment (their " +
            "files, their tools, their working directory), not a sandbox: use it to " +
            "run the project's tests, read or edit real files, drive a CLI, or inspect " +
            "the actual system. The command runs through the user's shell, so pipes, " +
            "redirects, globs, and control operators work as written. For disposable " +
            "computation that doesn't need to touch this machine, prefer the sandboxed " +
            "code execution tool instead. A non-zero exit code is returned as data, " +
            "not an error: read it and react. " +
            shellEnvNote() +
            policyNote(mode),
        parameters: {
            type: "object",
            properties: {
                command: {
                    type: "string",
                    description:
                        "The shell command to run, exactly as you'd type it at the " +
                        `user's ${shellName()} prompt (a full command line, e.g. ` +
                        "'npm test' or 'grep -n TODO src/*.ts | head'), in that " +
                        "shell's syntax.",
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
            const decision: ShellPolicyDecision = { mode, blocked: false };
            if (typeof command !== "string" || command.trim() === "") {
                return shellError("command must be a non-empty string", decision);
            }
            const cwd = typeof a.cwd === "string" && a.cwd.trim() ? a.cwd : defaultCwd;

            // Govern the call before it can spawn. A blocked command never runs and
            // never throws: it returns a structured result the model reads as a
            // refusal, and the loop logs the whole result (decision included) as the
            // tool_result event's audit metadata.
            const verdict = evaluateShellPolicy(command, cwd, policy);
            if (verdict.blocked) {
                return shellError(verdict.reason ?? "blocked by shell policy", verdict);
            }

            // Clamp the per-call timeout down to the policy ceiling, if one is set.
            let timeoutMs =
                typeof a.timeout_ms === "number" &&
                Number.isFinite(a.timeout_ms) &&
                a.timeout_ms > 0
                    ? a.timeout_ms
                    : defaultTimeoutMs;
            if (policy.maxTimeoutMs !== undefined) {
                timeoutMs = Math.min(timeoutMs, policy.maxTimeoutMs);
            }
            return runCommand(command, { cwd, timeoutMs, outputCap, decision: verdict });
        },
    };

    return [shell];
}

/** Build the spawn-failure (or policy-block) result shape: the paths where `ran`
 *  is false. Carries the policy decision so even a refusal is auditable. */
function shellError(message: string, policy: ShellPolicyDecision): ShellResult {
    return {
        ran: false,
        exitCode: null,
        signal: null,
        stdout: "",
        stderr: "",
        timedOut: false,
        error: message,
        policy,
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
    opts: { cwd?: string; timeoutMs: number; outputCap: number; decision: ShellPolicyDecision },
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
            finish(shellError(err instanceof Error ? err.message : String(err), opts.decision));
        });

        child.on("close", (code, signal) => {
            finish({
                ran: true,
                exitCode: code,
                signal: signal ?? null,
                stdout: capTail(stdout, opts.outputCap),
                stderr: capTail(stderr, opts.outputCap),
                timedOut,
                policy: opts.decision,
            });
        });
    });
}
