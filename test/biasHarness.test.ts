/**
 * Tests for the bias-invariance harness ({@link biasInvariance},
 * {@link panelRunner}).
 *
 * The harness's own logic (perturbing the roster per trial, tallying passes,
 * and the correctRate / instability arithmetic) is exercised with a hand-written
 * {@link PanelRun} stub, so most of the suite needs no Session and no client: we
 * control exactly what each trial "votes" and assert the report it produces. One
 * end-to-end test drives the real {@link panelRunner} → {@link panel} path
 * through a scripted {@link FakeClient}, to prove the live wiring matches the
 * stubbed contract.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { biasInvariance, panelRunner, type Probe, type PanelRun } from "../src/biasHarness.ts";
import type { Personality } from "../src/critics.ts";
import { FakeClient, textTurn } from "./helpers/fakeClient.ts";

/** A pinned RNG cycling through the given values, so the per-trial shuffle and
 *  deal are deterministic. Mirrors the one in critics.test.ts. */
function pinned(...values: number[]): () => number {
    let i = 0;
    return () => values[i++ % values.length]!;
}

const ROSTER: Personality[] = [{ name: "A" }, { name: "B" }, { name: "C" }];

const GOOD: Probe = { name: "good", candidate: "clean work", expected: true };
const BAD: Probe = { name: "bad", candidate: "flawed work", expected: false };

// ── biasInvariance: the core measurement ──────────────────────────────────────

test("a perfectly invariant, correct panel scores correctRate 1 and instability 0", async () => {
    // The panel always gets it right: PASS the good candidate, FAIL the bad one,
    // regardless of seating or deal. That is the ideal a real panel is measured
    // against.
    const run: PanelRun = async (_roster, candidate) => candidate === "clean work";
    const [good, bad] = await biasInvariance(ROSTER, run, [GOOD, BAD], { trials: 10 });

    assert.equal(good!.passes, 10);
    assert.equal(good!.correctRate, 1);
    assert.equal(good!.instability, 0, "a unanimous-across-trials verdict is fully stable");

    assert.equal(bad!.passes, 0);
    assert.equal(bad!.correctRate, 1, "always-FAIL on the bad candidate is the correct answer");
    assert.equal(bad!.instability, 0);
});

test("a coin-flip panel surfaces as maximal instability", async () => {
    // The verdict tracks a nuisance variable (here, the trial parity) instead of
    // the work: exactly the bias the harness exists to catch. 5 of 10 pass → the
    // pass rate is 0.5, the worst possible instability.
    let n = 0;
    const run: PanelRun = async () => n++ % 2 === 0;
    const [good] = await biasInvariance(ROSTER, run, [GOOD], { trials: 10 });

    assert.equal(good!.passes, 5);
    assert.equal(good!.instability, 0.5, "a coin flip is maximally swayable");
    assert.equal(good!.correctRate, 0.5, "and it's right only half the time");
});

test("a wrong-but-stable panel is low instability AND low correctRate", async () => {
    // Always FAIL the good candidate: perfectly invariant (instability 0) but
    // perfectly wrong (correctRate 0). The two numbers are independent, and this
    // is the case that proves it: stability alone is not trustworthiness.
    const run: PanelRun = async () => false;
    const [good] = await biasInvariance(ROSTER, run, [GOOD], { trials: 8 });

    assert.equal(good!.instability, 0, "unwavering");
    assert.equal(good!.correctRate, 0, "unwaveringly wrong");
});

test("each trial sees a freshly perturbed roster, and the input is never mutated", async () => {
    const seenOrders: string[] = [];
    const seenStakes: number[] = [];
    const run: PanelRun = async (roster) => {
        seenOrders.push(roster.map((p) => p.name).join(""));
        // dealRoster has stamped stakes onto every member: the harness dealt them.
        seenStakes.push(roster.filter((p) => (p.stakes ?? []).length > 0).length);
        return true;
    };
    await biasInvariance(ROSTER, run, [GOOD], { trials: 6, random: Math.random });

    // The roster the caller passed is untouched: no stakes leaked back, order kept.
    assert.deepEqual(
        ROSTER.map((p) => p.name),
        ["A", "B", "C"],
    );
    for (const p of ROSTER) assert.equal(p.stakes, undefined);
    // Every trial dealt stakes (dealRoster ran): on a 3-panel, at least the two
    // populated valences get a stake, so >= 2 members carry one each time.
    for (const dealt of seenStakes) assert.ok(dealt >= 2, "the harness dealt stakes each trial");
    assert.equal(seenOrders.length, 6, "ran exactly `trials` times");
});

test("onTrial streams each verdict with a 1-based trial number", async () => {
    const calls: Array<[string, boolean, number]> = [];
    const run: PanelRun = async (_r, candidate) => candidate === "clean work";
    await biasInvariance(ROSTER, run, [GOOD], {
        trials: 3,
        onTrial: (probe, ok, t) => calls.push([probe.name, ok, t]),
    });
    assert.deepEqual(calls, [
        ["good", true, 1],
        ["good", true, 2],
        ["good", true, 3],
    ]);
});

test("a trial whose panel throws is counted as an error and excluded", async () => {
    // The 2nd of 4 trials throws. It must not sink the run, and the rates are
    // computed over the 3 trials that actually ran.
    let n = 0;
    const run: PanelRun = async () => {
        n++;
        if (n === 2) throw new Error("panel blew up");
        return true; // the surviving trials all PASS
    };
    const [good] = await biasInvariance(ROSTER, run, [GOOD], { trials: 4 });

    assert.equal(good!.errors, 1);
    assert.equal(good!.trials, 3, "the report counts only the trials that ran");
    assert.equal(good!.passes, 3);
    assert.equal(good!.correctRate, 1, "rates are over the surviving trials");
});

test("an all-error probe reports zero trials without dividing by zero", async () => {
    const run: PanelRun = async () => {
        throw new Error("always down");
    };
    const [good] = await biasInvariance(ROSTER, run, [GOOD], { trials: 3 });
    assert.equal(good!.trials, 0);
    assert.equal(good!.errors, 3);
    assert.equal(good!.passes, 0);
    assert.equal(good!.correctRate, 0, "no data → 0, not NaN");
    assert.equal(good!.instability, 0);
});

test("biasInvariance rejects a trials count below 1", async () => {
    const run: PanelRun = async () => true;
    await assert.rejects(
        () => biasInvariance(ROSTER, run, [GOOD], { trials: 0 }),
        /trials must be ≥ 1/,
    );
});

test("biasInvariance reports probes in input order", async () => {
    const run: PanelRun = async (_r, candidate) => candidate === "clean work";
    const reports = await biasInvariance(ROSTER, run, [BAD, GOOD], { trials: 2 });
    assert.deepEqual(
        reports.map((r) => r.name),
        ["bad", "good"],
    );
});

test("the perturbation is deterministic under a pinned RNG", async () => {
    const orders: string[] = [];
    const run: PanelRun = async (roster) => {
        orders.push(roster.map((p) => p.name).join(""));
        return true;
    };
    // Two runs with the same pinned RNG must visit the same sequence of seatings.
    const seeds = () => pinned(0.1, 0.4, 0.7, 0.2, 0.9, 0.5);
    await biasInvariance(ROSTER, run, [GOOD], { trials: 3, random: seeds() });
    const first = orders.splice(0);
    await biasInvariance(ROSTER, run, [GOOD], { trials: 3, random: seeds() });
    assert.deepEqual(orders, first, "same RNG → same orderings");
});

// ── panelRunner: the live adapter, end to end through a FakeClient ─────────────

test("panelRunner runs the real panel and returns its consensus bit", async () => {
    // Two probes, each a 3-critic panel at concurrency 1 so the scripted replies
    // map to critics in order. Good candidate: 2 PASS, 1 FAIL → majority PASS.
    // Bad candidate: 1 PASS, 2 FAIL → majority FAIL. One trial each keeps the
    // script finite and the mapping legible.
    const client = new FakeClient([
        // trial over the good candidate
        textTurn("PASS"),
        textTurn("PASS"),
        textTurn("FAIL"),
        // trial over the bad candidate
        textTurn("FAIL"),
        textTurn("PASS"),
        textTurn("FAIL"),
    ]);
    const run = panelRunner({ client }, { concurrency: 1 });
    const reports = await biasInvariance(ROSTER, run, [GOOD, BAD], {
        trials: 1,
        random: pinned(0),
        // The seating doesn't matter here: at concurrency 1 the FakeClient hands
        // back replies in *call* order, so the scripted PASS/FAIL pattern maps to
        // critics however the roster is shuffled. We pin the RNG only so the run
        // is reproducible, not to fix the order.
    });

    const good = reports.find((r) => r.name === "good")!;
    const bad = reports.find((r) => r.name === "bad")!;
    assert.equal(good.passes, 1, "good candidate cleared the majority bar");
    assert.equal(good.correctRate, 1);
    assert.equal(bad.passes, 0, "bad candidate was failed by the majority");
    assert.equal(bad.correctRate, 1);
});
