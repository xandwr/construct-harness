/**
 * Tests for the interactive REPL ({@link runRepl}).
 *
 * The REPL owns the terminal, so we inject a scripted line source and a string
 * buffer for output (no real TTY) and assert on what gets rendered: the live
 * reply text, tool/compaction status lines, the accounting footer, and slash
 * commands.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { Readable } from "node:stream";

import { runRepl } from "../src/repl.ts";
import type { ReplOutput } from "../src/repl.ts";
import { Session } from "../src/session.ts";
import { HarnessError } from "../src/bridge/errors.ts";
import type { ModelClient } from "../src/bridge/types.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

/** A line source that yields each input line then ends (EOF closes the REPL). */
function inputOf(lines: string[]): Readable {
    return Readable.from(
        (async function* () {
            for (const l of lines) yield `${l}\n`;
        })(),
    );
}

/** A string-collecting output sink. Non-TTY, so output is plain (no ANSI). */
function bufferOutput(): ReplOutput & { text(): string } {
    let buf = "";
    return {
        isTTY: false,
        write(s: string) {
            buf += s;
        },
        text: () => buf,
    };
}

test("renders the streamed reply and an accounting footer", async () => {
    const client = new FakeClient([textTurn("hello from the construct")]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["hi", "/exit"]), output: out });

    const text = out.text();
    assert.match(text, /hello from the construct/);
    assert.match(text, /1 turn\(s\)/);
    assert.match(text, /1 in \/ 1 out tokens/);
});

test("/reset clears history and reports it", async () => {
    const client = new FakeClient([textTurn("a")]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["one", "/reset", "/exit"]), output: out });

    assert.equal(session.history().length, 0);
    assert.match(out.text(), /History cleared/);
});

test("/history reports the message count", async () => {
    const client = new FakeClient([textTurn("a")]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["one", "/history", "/exit"]), output: out });
    // After one exchange: user + assistant = 2 messages.
    assert.match(out.text(), /2 message\(s\) in history/);
});

test("shows tool activity as it runs", async () => {
    const noop = {
        name: "noop",
        description: "does nothing",
        parameters: { type: "object" },
        async run() {
            return "ok";
        },
    };
    const client = new FakeClient([callTurn("c1", "noop", { a: 1 }), textTurn("done")]);
    const session = new Session({ client, system: "S", tools: [noop] });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["go", "/exit"]), output: out });

    const text = out.text();
    assert.match(text, /↳ noop/, "tool_start line missing");
    assert.match(text, /noop ok/, "tool_end line missing");
    assert.match(text, /done/);
});

test("renders markdown in the streamed reply (line-buffered)", async () => {
    const client = new FakeClient([textTurn("# Title\n\nuse **bold** and `code` here\n")]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["hi", "/exit"]), output: out });

    const text = out.text();
    // Non-TTY buffer → plain rendering: hashes/asterisks/backticks stripped.
    // (The heading shares a line with the "› " prompt, hence not anchored to ^.)
    assert.match(text, /\bTitle\b/, "heading should render");
    assert.doesNotMatch(text, /# Title/, "heading hashes should be stripped");
    assert.match(text, /use bold and code here/, "inline markers should be stripped");
    assert.doesNotMatch(text, /\*\*bold\*\*/, "raw bold markers should not survive");
});

test("renders LaTeX math approximated to Unicode", async () => {
    const client = new FakeClient([textTurn("energy $E = mc^2$ and $\\alpha$\n")]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["hi", "/exit"]), output: out });

    assert.match(out.text(), /energy E = mc² and α/);
});

test("markdown spanning multiple stream deltas renders once the line completes", async () => {
    // A construct split across deltas: the renderer must buffer until the line's
    // newline arrives, then render the whole line.
    const split = {
        content: [
            { kind: "text", text: "a **bo" },
            { kind: "text", text: "ld** word\n" },
        ],
        stopReason: "end_turn",
    } as const;
    const client = new FakeClient([split]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["hi", "/exit"]), output: out });

    const text = out.text();
    assert.match(text, /a bold word/, "split bold should still render");
    assert.doesNotMatch(text, /\*\*/, "no raw markers should leak");
});

test("separates text segments around a tool block with a blank line", async () => {
    // The pattern that used to render as one unspaced blob: a turn that emits
    // prose *and* calls a tool, followed by a turn with more prose. The tool
    // status lines must not let the two prose segments butt together.
    const noop = {
        name: "noop",
        description: "does nothing",
        parameters: { type: "object" },
        async run() {
            return "ok";
        },
    };
    const turnWithText = {
        content: [
            { kind: "text", text: "before the tool" },
            { kind: "tool_call", id: "c1", name: "noop", args: {} },
        ],
        stopReason: "tool_use",
    } as const;
    const client = new FakeClient([turnWithText, textTurn("after the tool")]);
    const session = new Session({ client, system: "S", tools: [noop] });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["go", "/exit"]), output: out });

    const text = out.text();
    // First prose ends on its own line before the tool block; second prose is
    // separated from the tool's "ok" line by a blank line.
    assert.match(text, /before the tool\n\n  ↳ noop\(/, "text should close before the tool block");
    assert.match(text, /noop ok\n\nafter the tool/, "a blank line should precede resumed prose");
    // And no run-together of the two prose segments.
    assert.doesNotMatch(text, /before the toolafter the tool/);
});

test("an unknown slash command is reported, not fatal", async () => {
    const client = new FakeClient([textTurn("a")]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["/frobnicate", "ok now", "/exit"]), output: out });
    assert.match(out.text(), /Unknown command: \/frobnicate/);
    // It kept going and still answered the real message.
    assert.match(out.text(), /^[\s\S]*a[\s\S]*Bye/);
});

test("blank lines are ignored", async () => {
    const client = new FakeClient([textTurn("answer")]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    // Two blank lines then one real message; only one generate should happen.
    await runRepl(session, { input: inputOf(["", "  ", "real", "/exit"]), output: out });
    assert.equal(client.calls.length, 1, "blank lines should not trigger a turn");
    assert.match(out.text(), /answer/);
});

test("ends cleanly on EOF without /exit", async () => {
    const client = new FakeClient([textTurn("hi")]);
    const session = new Session({ client, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["hi"]), output: out });
    assert.match(out.text(), /Bye/);
});

test("renders a HarnessError with its kind and keeps the REPL alive", async () => {
    // A client whose stream throws a classified error: the REPL should report
    // the kind, not crash, and continue to the next prompt.
    const throwing: ModelClient = {
        provider: "x",
        model: "x",
        capabilities: {
            thinking: false,
            effort: false,
            promptCaching: false,
            serverTools: false,
            streaming: true,
        },
        async generate() {
            throw new HarnessError("nope", { kind: "rate_limit", retryable: true });
        },
        // eslint-disable-next-line require-yield
        async *stream() {
            throw new HarnessError("slow down", { kind: "rate_limit", retryable: true });
        },
    };
    const session = new Session({ client: throwing, system: "S" });
    const out = bufferOutput();

    await runRepl(session, { input: inputOf(["go", "/exit"]), output: out });
    const text = out.text();
    assert.match(text, /\[error\] rate_limit \(retries exhausted\): slow down/);
    assert.match(text, /Bye/, "REPL should survive the error and exit cleanly");
});
