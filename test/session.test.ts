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

import { Session } from "../src/session.ts";
import type { LoopEvent } from "../src/bridge/loop.ts";
import type { TurnResult } from "../src/session.ts";
import { RoleType } from "../src/types.ts";
import { MemoryStore } from "../src/memory.ts";
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
