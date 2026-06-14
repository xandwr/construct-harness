/**
 * Tests for the interactive {@link Session}.
 *
 * A Session is a long-lived conversation: history must persist across sends, the
 * per-turn system prompt (with recall) must never leak into that durable
 * history, and memory tools/recall must be wired when a store is present. Driven
 * by the streaming {@link FakeClient}.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Session } from "../src/session.ts";
import type { LoopEvent } from "../src/bridge/loop.ts";
import type { TurnResult } from "../src/session.ts";
import { RoleType } from "../src/types.ts";
import { MemoryStore } from "../src/memory.ts";
import { EventStore } from "../src/events.ts";
import { GoalStore } from "../src/goals.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

/** Drive one send to completion, returning streamed text and the TurnResult. */
async function send(
    session: Session,
    text: string,
): Promise<{ streamed: string; result: TurnResult }> {
    const gen = session.send(text);
    let streamed = "";
    let next = await gen.next();
    while (!next.done) {
        const e: LoopEvent = next.value;
        if (e.kind === "text") streamed += e.text;
        next = await gen.next();
    }
    return { streamed, result: next.value };
}

test("streams the reply and returns it in the TurnResult", async () => {
    const client = new FakeClient([textTurn("hi there")]);
    const session = new Session({ client, system: "be nice" });

    const { streamed, result } = await send(session, "hello");
    assert.equal(streamed, "hi there");
    assert.equal(result.text, "hi there");
    assert.equal(result.modelTurns, 1);
});

test("persists conversation across sends, without the system turn", async () => {
    const client = new FakeClient([textTurn("first"), textTurn("second")]);
    const session = new Session({ client, system: "SYS-GUIDANCE" });

    await send(session, "one");
    await send(session, "two");

    const history = session.history();
    // user, assistant, user, assistant: no system turn persisted.
    assert.equal(history.length, 4);
    assert.ok(
        history.every((m) => m.sender.role !== RoleType.System),
        "system turn leaked into history",
    );
    assert.equal(history[0]!.content[0]!.kind === "text" && history[0]!.content[0].text, "one");
});

test("the second send sees the first turn as prior context", async () => {
    const client = new FakeClient([textTurn("a1"), textTurn("a2")]);
    const session = new Session({ client, system: "S" });

    await send(session, "u1");
    await send(session, "u2");

    // The second generate call's messages must include the first exchange.
    const secondCall = client.calls[1]!.messages;
    const texts = secondCall
        .flatMap((m) => m.content)
        .filter(
            (p): p is Extract<(typeof m.content)[number], { kind: "text" }> => p.kind === "text",
        )
        .map((p) => p.text);
    assert.ok(texts.includes("u1"), "first user turn missing from second send");
    assert.ok(texts.includes("a1"), "first assistant turn missing from second send");
    assert.ok(texts.includes("u2"));
});

test("system guidance is present on the wire but never in history", async () => {
    const client = new FakeClient([textTurn("ok")]);
    const session = new Session({ client, system: "SECRET-SYSTEM" });

    await send(session, "hi");

    const wireSystem = client.calls[0]!.messages.filter((m) => m.sender.role === RoleType.System)
        .flatMap((m) => m.content)
        .filter(
            (p): p is Extract<(typeof m.content)[number], { kind: "text" }> => p.kind === "text",
        )
        .map((p) => p.text)
        .join(" ");
    assert.match(wireSystem, /SECRET-SYSTEM/);
    assert.ok(session.history().every((m) => m.sender.role !== RoleType.System));
});

test("reset clears history", async () => {
    const client = new FakeClient([textTurn("a"), textTurn("b")]);
    const session = new Session({ client, system: "S" });
    await send(session, "u1");
    assert.equal(session.history().length, 2);
    session.reset();
    assert.equal(session.history().length, 0);
});

test("a store wires memory tools and the model can save", async () => {
    const store = new MemoryStore(":memory:");
    try {
        // Turn 1: the model calls memory_save; turn 2: it replies.
        const client = new FakeClient([
            callTurn("c1", "memory_save", { content: "user likes tea" }),
            textTurn("noted"),
        ]);
        const session = new Session({ client, system: "S", store });

        const { result } = await send(session, "remember I like tea");
        assert.equal(result.text, "noted");
        assert.equal(result.modelTurns, 2);
        assert.equal(store.count(), 1, "memory_save did not persist");
    } finally {
        store.close();
    }
});

test("recalled memory is injected into the system prompt", async () => {
    const store = new MemoryStore(":memory:");
    try {
        store.save(
            new (await import("../src/memory.ts")).Memory({ content: "user's name is Ada" }),
        );
        const client = new FakeClient([textTurn("hello Ada")]);
        const session = new Session({ client, system: "S", store });

        await send(session, "what's my name");

        const wireSystem = client.calls[0]!.messages.filter(
            (m) => m.sender.role === RoleType.System,
        )
            .flatMap((m) => m.content)
            .filter(
                (p): p is Extract<(typeof m.content)[number], { kind: "text" }> =>
                    p.kind === "text",
            )
            .map((p) => p.text)
            .join(" ");
        assert.match(wireSystem, /Ada/, "recalled memory was not injected");
    } finally {
        store.close();
    }
});

test("TurnResult carries usage and compaction accounting", async () => {
    const client = new FakeClient([callTurn("c1", "noop", {}), textTurn("done")]);
    const noop = {
        name: "noop",
        description: "does nothing",
        parameters: { type: "object" },
        async run() {
            return "ok";
        },
    };
    const session = new Session({ client, system: "S", tools: [noop] });

    const { result } = await send(session, "go");
    assert.equal(result.modelTurns, 2);
    assert.equal(result.usage.outputTokens, 2);
    assert.equal(result.compactions, 0);
    assert.equal(result.stoppedAtMaxTurns, false);
});

// ── Event log wiring ──────────────────────────────────────────────────────────

test("a plain text turn appends a user message and an agent reply to the log", async () => {
    const events = new EventStore(":memory:");
    try {
        const client = new FakeClient([textTurn("hi there")]);
        const session = new Session({ client, system: "S", events });

        await send(session, "hello");

        const log = events.recent({ session: session.id }).reverse();
        assert.equal(log.length, 2);
        assert.equal(log[0]!.kind, "message");
        assert.equal(log[0]!.role, "user");
        assert.equal(log[0]!.content, "hello");
        assert.equal(log[1]!.kind, "message");
        assert.equal(log[1]!.role, "agent");
        assert.equal(log[1]!.content, "hi there");
    } finally {
        events.close();
    }
});

test("a tool turn logs the call and its result, correlated by id", async () => {
    const events = new EventStore(":memory:");
    try {
        const client = new FakeClient([callTurn("c1", "noop", { x: 1 }), textTurn("done")]);
        const noop = {
            name: "noop",
            description: "does nothing",
            parameters: { type: "object" },
            async run() {
                return { ok: true };
            },
        };
        const session = new Session({ client, system: "S", events, tools: [noop] });

        await send(session, "go");

        const log = events.recent({ session: session.id }).reverse();
        // user message, tool_call, tool_result, agent message.
        assert.deepEqual(
            log.map((e) => e.kind),
            ["message", "tool_call", "tool_result", "message"],
        );
        const call = log[1]!;
        const result = log[2]!;
        assert.equal(call.kind, "tool_call");
        assert.equal(call.content, "noop");
        assert.deepEqual((call.meta as { args: unknown }).args, { x: 1 });
        // The call and its result share a correlation id so a reader can thread them.
        assert.ok(call.correlation);
        assert.equal(call.correlation, result.correlation);
        assert.equal(result.kind, "tool_result");
        assert.match(result.content, /"ok":true/);
    } finally {
        events.close();
    }
});

test("transcript() reads this session's turns back in reading order", async () => {
    const events = new EventStore(":memory:");
    try {
        const client = new FakeClient([textTurn("a1"), textTurn("a2")]);
        const session = new Session({ client, system: "S", events });

        await send(session, "u1");
        await send(session, "u2");

        const transcript = session.transcript();
        assert.deepEqual(
            transcript.map((e) => e.content),
            ["u1", "a1", "u2", "a2"],
        );
        // Survives a reset: the log keeps what the in-memory history forgot.
        session.reset();
        assert.equal(session.history().length, 0);
        assert.equal(session.transcript().length, 4);
    } finally {
        events.close();
    }
});

// ── Goal wiring ────────────────────────────────────────────────────────────

test("a goal store wires goal tools and injects active goals each turn", async () => {
    const goals = new GoalStore(":memory:");
    try {
        // Turn 1: the model sets a goal. Turn 2: it replies. The goal it set this
        // turn is scoped to the session and then stands in the system prompt.
        const client = new FakeClient([
            callTurn("c1", "goal_set", { content: "finish the audit" }),
            textTurn("on it"),
        ]);
        const session = new Session({ client, system: "S", goals });

        await send(session, "please finish the audit");

        // The goal landed, scoped to this Session's id.
        const stored = goals.list({ session: session.id });
        assert.equal(stored.length, 1);
        assert.equal(stored[0]!.content, "finish the audit");

        // On the next turn, goalContext injects it: assert via the system prompt
        // the loop builds. We can observe it by sending again and checking the
        // client saw a system turn naming the goal.
        const client2 = new FakeClient([textTurn("still on it")]);
        const s2 = new Session({ client: client2, system: "S", goals, sessionId: session.id });
        await send(s2, "status?");
        const sentSystem = client2.calls
            .flatMap((c) => c.messages)
            .filter((m) => m.sender.role === RoleType.System)
            .flatMap((m) => m.content)
            .map((p) => (p.kind === "text" ? p.text : ""))
            .join("\n");
        assert.match(sentSystem, /active goals/i);
        assert.match(sentSystem, /finish the audit/);
    } finally {
        goals.close();
    }
});

test("a log wires transcript_recall, scoped to this session's own turns", async () => {
    const events = new EventStore(":memory:");
    try {
        // Session A records a turn; Session B (sharing the log) recalls and must
        // see only its own transcript, not A's.
        const a = new Session({
            client: new FakeClient([textTurn("we picked sqlite")]),
            system: "S",
            events,
        });
        await send(a, "what storage?");

        const b = new Session({
            client: new FakeClient([callTurn("c1", "transcript_recall", {}), textTurn("checked")]),
            system: "S",
            events,
        });
        const gen = b.send("recall my history");
        let toolPayload: { count: number; events: { content: string }[] } | undefined;
        let next = await gen.next();
        while (!next.done) {
            const e: LoopEvent = next.value;
            if (e.kind === "tool_end" && e.name === "transcript_recall") {
                toolPayload = e.result as { count: number; events: { content: string }[] };
            }
            next = await gen.next();
        }
        // B's transcript holds only its own turn so far (its user message, plus
        // the in-flight tool_call logged before the tool ran). Crucially, A's
        // "we picked sqlite" turn is invisible: recall is scoped to B's session.
        assert.ok(toolPayload, "transcript_recall did not run");
        assert.ok(
            !toolPayload!.events.some((e) => /sqlite/.test(e.content)),
            "recall leaked another session's turns",
        );
        assert.ok(
            toolPayload!.events.some((e) => /recall my history/.test(e.content)),
            "recall should see this session's own user message",
        );
    } finally {
        events.close();
    }
});

test("transcriptRecall: false withholds the tool; no log means no tool", async () => {
    const events = new EventStore(":memory:");
    try {
        // With a log but the flag off, the model calling transcript_recall hits
        // the loop's "No such tool" path rather than a registered handler.
        const off = new Session({
            client: new FakeClient([callTurn("c1", "transcript_recall", {}), textTurn("done")]),
            system: "S",
            events,
            transcriptRecall: false,
        });
        const gen = off.send("go");
        let errored = false;
        let next = await gen.next();
        while (!next.done) {
            const e: LoopEvent = next.value;
            if (e.kind === "tool_end" && e.name === "transcript_recall") errored = e.isError;
            next = await gen.next();
        }
        assert.ok(errored, "withheld tool should resolve as an error result");
    } finally {
        events.close();
    }
});

test("two sessions on one shared log stay separable by id", async () => {
    const events = new EventStore(":memory:");
    try {
        const a = new Session({ client: new FakeClient([textTurn("ra")]), system: "S", events });
        const b = new Session({ client: new FakeClient([textTurn("rb")]), system: "S", events });
        assert.notEqual(a.id, b.id);

        await send(a, "qa");
        await send(b, "qb");

        assert.deepEqual(
            a.transcript().map((e) => e.content),
            ["qa", "ra"],
        );
        assert.deepEqual(
            b.transcript().map((e) => e.content),
            ["qb", "rb"],
        );
    } finally {
        events.close();
    }
});

test("a pinned sessionId resumes appending to the same transcript", async () => {
    const events = new EventStore(":memory:");
    try {
        const first = new Session({
            client: new FakeClient([textTurn("r1")]),
            system: "S",
            events,
            sessionId: "fixed",
        });
        await send(first, "q1");

        // A second Session pinned to the same id continues the same transcript.
        const second = new Session({
            client: new FakeClient([textTurn("r2")]),
            system: "S",
            events,
            sessionId: "fixed",
        });
        await send(second, "q2");

        assert.deepEqual(
            second.transcript().map((e) => e.content),
            ["q1", "r1", "q2", "r2"],
        );
    } finally {
        events.close();
    }
});

test("no log configured: send still works and transcript is empty", async () => {
    const client = new FakeClient([textTurn("ok")]);
    const session = new Session({ client, system: "S" });
    const { result } = await send(session, "hi");
    assert.equal(result.text, "ok");
    assert.deepEqual(session.transcript(), []);
});

test("a memory saved during a turn is linked to the prompting user event", async () => {
    // Provenance spans both tables, so memory + log must share one db file.
    const dir = mkdtempSync(join(tmpdir(), "session-prov-"));
    const path = join(dir, "shared.sqlite");
    const store = new MemoryStore(path);
    const events = new EventStore(path);
    try {
        const client = new FakeClient([
            callTurn("c1", "memory_save", { content: "user likes tea" }),
            textTurn("noted"),
        ]);
        const session = new Session({ client, system: "S", store, events });

        await send(session, "remember I like tea");

        // The save persisted exactly one memory, and its provenance points at the
        // user-message event that opened the turn.
        assert.equal(store.count(), 1);
        const saved = store.all()[0]!;
        const eventId = store.provenanceOf(saved.id);
        assert.ok(eventId, "saved memory has no provenance");
        const sourceEvent = events.get(eventId!);
        assert.equal(sourceEvent?.kind, "message");
        assert.equal(sourceEvent?.role, "user");
        assert.equal(sourceEvent?.content, "remember I like tea");
        // And the reverse lookup goes from that event back to the memory.
        assert.deepEqual(
            store.memoriesFromEvent(eventId!).map((m) => m.content),
            ["user likes tea"],
        );
    } finally {
        events.close();
        store.close();
        rmSync(dir, { recursive: true, force: true });
    }
});
