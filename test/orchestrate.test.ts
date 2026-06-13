/**
 * Tests for the Loop orchestration layer ({@link loop}, {@link fanout},
 * {@link verify}, {@link orchestrate}).
 *
 * A Loop is the harness driving a {@link Session} with no human in the seat: it
 * supplies turns, judges results, and re-prompts. We drive it with the same
 * scripted {@link FakeClient} the loop/session tests use, so every iteration and
 * branch is deterministic and network-free. Each Session a test constructs gets
 * its own FakeClient, since a fan-out's branches must be independent.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { Session } from "../src/session.ts";
import {
    loop,
    fanout,
    verify,
    orchestrate,
    spawnFrom,
    type LoopStep,
} from "../src/orchestrate.ts";
import type { TurnResult } from "../src/session.ts";
import { FakeClient, textTurn, callTurn } from "./helpers/fakeClient.ts";

/** A Session over a one-reply scripted client — the minimal Construct. */
function constructSaying(...replies: string[]): Session {
    return new Session({ client: new FakeClient(replies.map(textTurn)), system: "S" });
}

// ── loop: iterate one Construct ──────────────────────────────────────────────

test("loop stops immediately when done holds on the first turn", async () => {
    const session = constructSaying("DONE");
    const res = await loop(session, "go", {
        done: (s) => s.turn.text.includes("DONE"),
        next: () => "keep going",
    });

    assert.equal(res.stopReason, "done");
    assert.equal(res.iterations, 1);
    assert.equal(res.final.text, "DONE");
});

test("loop re-prompts with next() until done, in the same Session", async () => {
    // Three turns: two that aren't done, then one that is.
    const session = constructSaying("working", "still working", "DONE");
    const prompts: string[] = [];
    const res = await loop(session, "start", {
        done: (s) => s.turn.text === "DONE",
        next: (s: LoopStep) => {
            const p = `continue (after iter ${s.iteration})`;
            prompts.push(p);
            return p;
        },
    });

    assert.equal(res.stopReason, "done");
    assert.equal(res.iterations, 3);
    // next() was called after the 1st and 2nd turns, not after the 3rd (done).
    assert.deepEqual(prompts, ["continue (after iter 1)", "continue (after iter 2)"]);
    // All three turns ran through ONE Session: its history holds every exchange.
    // user+assistant per send × 3 sends = 6 messages.
    assert.equal(session.history().length, 6);
});

test("loop stops at maxIterations when done never holds", async () => {
    const session = constructSaying("a", "b", "c", "d", "e");
    const res = await loop(session, "go", {
        done: () => false,
        next: () => "again",
        maxIterations: 3,
    });

    assert.equal(res.stopReason, "maxIterations");
    assert.equal(res.iterations, 3);
    // next() is not consulted after the final allowed iteration, so only 3 sends
    // happened — the client still has turns "d"/"e" unused, which is fine.
});

test("loop sums usage across every iteration", async () => {
    // Each FakeClient turn reports {input:1, output:1}; three iterations → 3 each.
    const session = constructSaying("x", "y", "DONE");
    const res = await loop(session, "go", {
        done: (s) => s.turn.text === "DONE",
        next: () => "more",
    });
    assert.equal(res.usage.inputTokens, 3);
    assert.equal(res.usage.outputTokens, 3);
});

test("loop forwards every LoopEvent to onEvent with the iteration number", async () => {
    const session = constructSaying("nope", "DONE");
    const seen: { text: string; iteration: number }[] = [];
    await loop(session, "go", {
        done: (s) => s.turn.text === "DONE",
        next: () => "again",
        onEvent: (e, iteration) => {
            if (e.kind === "text") seen.push({ text: e.text, iteration });
        },
    });
    assert.deepEqual(seen, [
        { text: "nope", iteration: 1 },
        { text: "DONE", iteration: 2 },
    ]);
});

test("loop supports an async done predicate", async () => {
    const session = constructSaying("not yet", "READY");
    const res = await loop(session, "go", {
        done: async (s) => {
            await Promise.resolve();
            return s.turn.text === "READY";
        },
        next: () => "again",
    });
    assert.equal(res.stopReason, "done");
    assert.equal(res.iterations, 2);
});

test("loop rejects a maxIterations below 1", async () => {
    const session = constructSaying("x");
    await assert.rejects(
        loop(session, "go", { done: () => true, next: () => "x", maxIterations: 0 }),
        /maxIterations must be ≥ 1/,
    );
});

// ── fanout: many independent Constructs ──────────────────────────────────────

test("fanout runs one Construct per task and returns results in input order", async () => {
    // Each branch gets its own client/Session so they're truly independent.
    const replies = ["one", "two", "three"];
    let i = 0;
    const res = await fanout(["t0", "t1", "t2"], () => constructSaying(replies[i++]!));

    assert.equal(res.length, 3);
    assert.deepEqual(
        res.map((b) => b.result?.text),
        ["one", "two", "three"],
    );
    assert.deepEqual(
        res.map((b) => b.index),
        [0, 1, 2],
    );
});

test("fanout isolates a throwing branch as an error result, not a batch failure", async () => {
    // A client with an empty script throws on first send ("called more times than
    // scripted") — a clean way to make exactly one branch fail.
    const res = await fanout(["ok0", "boom1", "ok2"], (task) => {
        if (task === "boom1") return new Session({ client: new FakeClient([]), system: "S" });
        return constructSaying("fine");
    });

    assert.equal(res[0]!.result?.text, "fine");
    assert.equal(res[2]!.result?.text, "fine");
    // The middle branch failed but its siblings completed.
    assert.equal(res[1]!.result, null);
    assert.ok(res[1]!.error instanceof Error);
    assert.match(res[1]!.error!.message, /more times than scripted/);
});

test("fanout never exceeds its concurrency cap", async () => {
    let active = 0;
    let peak = 0;
    // A spawn whose Session.send we can't easily instrument; instead use a client
    // that records concurrency via a gate in generate. Simplest: a custom client.
    const makeGatedSession = () => {
        const client = new FakeClient([textTurn("done")]);
        const origStream = client.stream.bind(client);
        // Wrap stream to observe overlap. The loop's drive() consumes stream().
        client.stream = async function* (params) {
            active++;
            peak = Math.max(peak, active);
            try {
                await Promise.resolve(); // yield, so siblings can start if uncapped
                for await (const d of origStream(params)) yield d;
            } finally {
                active--;
            }
        };
        return new Session({ client, system: "S" });
    };

    await fanout(["a", "b", "c", "d", "e", "f"], () => makeGatedSession(), { concurrency: 2 });
    assert.ok(peak <= 2, `peak concurrency ${peak} exceeded cap of 2`);
});

test("fanout rejects a concurrency below 1", async () => {
    await assert.rejects(
        fanout(["t"], () => constructSaying("x"), { concurrency: 0 }),
        /concurrency must be ≥ 1/,
    );
});

test("spawnFrom mints a distinct Session per branch", async () => {
    // Shared config, but each call must yield a NEW Session (independent history).
    const cfg = { client: new FakeClient([textTurn("a"), textTurn("b")]), system: "S" };
    const spawn = spawnFrom(cfg);
    const s1 = spawn("t0", 0);
    const s2 = spawn("t1", 1);
    assert.notEqual(s1, s2, "spawnFrom returned the same Session twice");
});

// ── verify: an adversarial check ─────────────────────────────────────────────

test("verify maps a PASS reply to ok", async () => {
    const verifier = constructSaying("Looks correct and complete. PASS");
    const v = await verify(verifier, "the candidate work");
    assert.equal(v.ok, true);
    assert.match(v.rationale, /PASS/);
});

test("verify maps a FAIL reply to not ok", async () => {
    const verifier = constructSaying("Misses an edge case. FAIL");
    const v = await verify(verifier, "the candidate work");
    assert.equal(v.ok, false);
});

test("verify honours a custom decide predicate", async () => {
    const verifier = constructSaying("score: 9/10");
    const v = await verify(verifier, "work", {
        decide: (text) => /([7-9]|10)\/10/.test(text),
    });
    assert.equal(v.ok, true);
});

test("verify sends the candidate through the prompt builder", async () => {
    const client = new FakeClient([textTurn("PASS")]);
    const verifier = new Session({ client, system: "skeptic" });
    await verify(verifier, "CANDIDATE-MARKER", {
        prompt: (c) => `check this: ${c}`,
    });
    // The wire saw the wrapped candidate.
    const wire = client.calls[0]!.messages
        .flatMap((m) => m.content)
        .filter((p): p is { kind: "text"; text: string } => p.kind === "text")
        .map((p) => p.text)
        .join(" ");
    assert.match(wire, /check this: CANDIDATE-MARKER/);
});

// ── orchestrate: fan out, verify, merge ──────────────────────────────────────

test("orchestrate confirms only branches that pass verification", async () => {
    const work = ["good", "bad"];
    let w = 0;
    const workReplies = ["solution A", "solution B"];
    // Verifier: pass the first candidate, fail the second.
    let v = 0;
    const verdicts = ["PASS", "FAIL"];

    const res = await orchestrate(work, {
        spawn: () => constructSaying(workReplies[w++]!),
        spawnVerifier: () => constructSaying(verdicts[v++]!),
    });

    assert.equal(res.branches.length, 2);
    assert.equal(res.confirmed.length, 1);
    assert.equal(res.confirmed[0]!.result?.text, "solution A");
    assert.equal(res.confirmed[0]!.verdict?.ok, true);
    // The failed-verification branch is in branches but not confirmed.
    assert.equal(res.branches[1]!.verdict?.ok, false);
});

test("orchestrate carries a failed worker branch through with a null verdict", async () => {
    let verifierSpawned = 0;
    const res = await orchestrate(["boom", "ok"], {
        spawn: (task) =>
            task === "boom"
                ? new Session({ client: new FakeClient([]), system: "S" }) // throws
                : constructSaying("good work"),
        spawnVerifier: () => {
            verifierSpawned++;
            return constructSaying("PASS");
        },
    });

    // The boom branch: failed, never verified.
    const boom = res.branches.find((b) => b.task === "boom")!;
    assert.equal(boom.result, null);
    assert.equal(boom.verdict, null);
    // Only the successful branch got a verifier spawned for it.
    assert.equal(verifierSpawned, 1);
    assert.equal(res.confirmed.length, 1);
    assert.equal(res.confirmed[0]!.task, "ok");
});

test("orchestrate of a single task is just a verified single Construct", async () => {
    const res = await orchestrate(["the one task"], {
        spawn: () => constructSaying("the answer"),
        spawnVerifier: () => constructSaying("PASS"),
    });
    assert.equal(res.branches.length, 1);
    assert.equal(res.confirmed.length, 1);
    assert.equal(res.confirmed[0]!.result?.text, "the answer");
});

test("orchestrate tags events with their branch and phase", async () => {
    const phases: { phase: string; text: string }[] = [];
    await orchestrate(["t0"], {
        spawn: () => constructSaying("work-output"),
        spawnVerifier: () => constructSaying("PASS"),
        onEvent: (e, _branch, phase) => {
            if (e.kind === "text") phases.push({ phase, text: e.text });
        },
    });
    assert.deepEqual(phases, [
        { phase: "work", text: "work-output" },
        { phase: "verify", text: "PASS" },
    ]);
});

test("orchestrate works with a Construct that uses tools mid-branch", async () => {
    // A worker branch that calls a tool then answers — proves the branch drive
    // runs the full agentic loop, not just one model turn.
    const noop = {
        name: "noop",
        description: "does nothing",
        parameters: { type: "object" },
        async run() {
            return "ok";
        },
    };
    const res = await orchestrate(["do it"], {
        spawn: () =>
            new Session({
                client: new FakeClient([callTurn("c1", "noop", {}), textTurn("finished")]),
                system: "S",
                tools: [noop],
            }),
        spawnVerifier: () => constructSaying("PASS"),
    });
    assert.equal(res.confirmed.length, 1);
    assert.equal(res.confirmed[0]!.result?.text, "finished");
});
