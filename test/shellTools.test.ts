/**
 * Tests for the local-shell tool bridge ({@link shellTools}).
 *
 * Exercises the one tool two ways: directly (calling `ToolDef.run`, the way the
 * loop would) and through {@link runLoop} driven by the scripted
 * {@link FakeClient}, plus its edges: a non-zero exit returned as data not an
 * error, stdout/stderr capture, output capping, a custom cwd, the per-call
 * timeout, bad args, and a spawn failure.
 *
 * The commands run a real shell (this is what the tool *is*), but only trivial,
 * side-effect-free ones (echo, exit, a bounded sleep), so the suite stays
 * hermetic and fast.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import os from "node:os";

import {
    shellTools,
    USER_SHELL_TOOL,
    SHELL_OUTPUT_CAP,
    type ShellResult,
} from "../src/shellTools.ts";
import { runLoop } from "../src/bridge/loop.ts";
import { RoleType } from "../src/types.ts";
import type { Message, ToolDef, ToolResultPart } from "../src/types.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

function tool(tools: ToolDef[], name: string): ToolDef {
    const t = tools.find((x) => x.name === name);
    assert.ok(t, `expected a tool named ${name}`);
    return t;
}

/** Run the shell tool's `run` and narrow the result to {@link ShellResult}. */
async function runShell(args: unknown, opts = {}): Promise<ShellResult> {
    const shell = tool(shellTools(opts), USER_SHELL_TOOL);
    return (await shell.run(args)) as ShellResult;
}

const user = (text: string): Message => ({
    sender: { role: RoleType.User },
    timestamp: 0,
    content: [{ kind: "text", text }],
});

// ---------------------------------------------------------------------------
// Tool shape
// ---------------------------------------------------------------------------

test("shellTools exposes a single use__user__shell tool", () => {
    assert.deepEqual(
        shellTools().map((t) => t.name),
        [USER_SHELL_TOOL],
    );
    const shell = tool(shellTools(), USER_SHELL_TOOL);
    const schema = shell.parameters as { required?: string[] };
    assert.deepEqual(schema.required, ["command"]);
});

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

test("runs a command and returns its stdout and exit code", async () => {
    const r = await runShell({ command: "echo hello" });
    assert.equal(r.ran, true);
    assert.equal(r.exitCode, 0);
    assert.equal(r.signal, null);
    assert.equal(r.timedOut, false);
    assert.equal(r.stdout.trim(), "hello");
    assert.equal(r.stderr, "");
    assert.equal(r.error, undefined);
});

test("captures stderr separately from stdout", async () => {
    const r = await runShell({ command: "echo out; echo err 1>&2" });
    assert.equal(r.ran, true);
    assert.equal(r.exitCode, 0);
    assert.equal(r.stdout.trim(), "out");
    assert.equal(r.stderr.trim(), "err");
});

test("honors shell features (pipes) since it runs through a shell", async () => {
    const r = await runShell({ command: "printf 'a\\nb\\nc\\n' | wc -l" });
    assert.equal(r.ran, true);
    assert.equal(r.exitCode, 0);
    assert.equal(r.stdout.trim(), "3");
});

// ---------------------------------------------------------------------------
// A non-zero exit is data, not an error
// ---------------------------------------------------------------------------

test("a non-zero exit comes back as ran:true with the code, not an error", async () => {
    const r = await runShell({ command: "exit 3" });
    assert.equal(r.ran, true);
    assert.equal(r.exitCode, 3);
    assert.equal(r.error, undefined);
});

// ---------------------------------------------------------------------------
// Working directory
// ---------------------------------------------------------------------------

test("runs in a per-call cwd when given", async () => {
    const dir = os.tmpdir();
    const r = await runShell({ command: "pwd", cwd: dir });
    assert.equal(r.ran, true);
    // macOS symlinks /tmp → /private/tmp; compare on the basename to stay portable.
    assert.ok(
        r.stdout.trim().endsWith(dir.replace(/^.*\//, "")) || r.stdout.trim() === dir,
        `expected pwd to reflect ${dir}, got ${r.stdout.trim()}`,
    );
});

test("falls back to the configured defaultCwd", async () => {
    const dir = os.tmpdir();
    const r = await runShell({ command: "pwd" }, { defaultCwd: dir });
    assert.equal(r.ran, true);
    assert.ok(r.stdout.includes(dir.replace(/^.*\//, "")));
});

// ---------------------------------------------------------------------------
// Output capping
// ---------------------------------------------------------------------------

test("tail-truncates output past the cap and marks it", async () => {
    // Emit well over the cap; the tail (and a truncation marker) should survive.
    const n = SHELL_OUTPUT_CAP + 5_000;
    const r = await runShell({ command: `yes x | head -c ${n}` });
    assert.equal(r.ran, true);
    assert.ok(r.stdout.length <= SHELL_OUTPUT_CAP + 100, "kept length is bounded by the cap");
    assert.match(r.stdout, /truncated/);
    // The very end of the stream is preserved (`yes x` emits "x\n" repeated, so
    // the tail ends on a newline after the final x).
    assert.ok(r.stdout.trimEnd().endsWith("x"));
});

// ---------------------------------------------------------------------------
// Timeout
// ---------------------------------------------------------------------------

test("kills a command that exceeds its timeout and reports timedOut", async () => {
    const r = await runShell({ command: "sleep 5", timeout_ms: 150 });
    assert.equal(r.ran, true);
    assert.equal(r.timedOut, true);
    // Terminated by a signal, so the exit code is null.
    assert.equal(r.exitCode, null);
    assert.ok(r.signal, "a killed process reports the signal");
});

// ---------------------------------------------------------------------------
// Bad args and spawn failures (the ran:false path)
// ---------------------------------------------------------------------------

test("rejects an empty command without spawning anything", async () => {
    const r = await runShell({ command: "   " });
    assert.equal(r.ran, false);
    assert.match(r.error ?? "", /non-empty/);
});

test("a missing cwd surfaces as ran:false with an error, never a throw", async () => {
    const r = await runShell({ command: "echo hi", cwd: "/no/such/dir/xyzzy" });
    assert.equal(r.ran, false);
    assert.ok(r.error, "a spawn failure carries an explanation");
});

// ---------------------------------------------------------------------------
// Through the loop
// ---------------------------------------------------------------------------

test("the loop dispatches use__user__shell and feeds the result back", async () => {
    const client = new FakeClient([
        callTurn("c1", USER_SHELL_TOOL, { command: "echo from-the-loop" }),
        textTurn("done"),
    ]);
    const result = await runLoop(client, {
        messages: [user("run echo")],
        tools: shellTools(),
    });

    // The second model turn saw a tool_result for our call, carrying the stdout.
    const toolTurn = result.messages.find((m) => m.content.some((p) => p.kind === "tool_result"));
    assert.ok(toolTurn, "a tool_result turn was appended");
    const part = toolTurn!.content.find((p): p is ToolResultPart => p.kind === "tool_result")!;
    assert.equal(part.callId, "c1");
    assert.equal(part.isError, undefined);
    const payload = part.result as ShellResult;
    assert.equal(payload.ran, true);
    assert.equal(payload.exitCode, 0);
    assert.equal(payload.stdout.trim(), "from-the-loop");
    assert.equal(result.final.message.content[0]?.kind, "text");
});
