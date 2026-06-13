/**
 * Tests for the adversarial critic panel ({@link Personality}, {@link critic},
 * {@link panel}, {@link panelVerify}, and the consensus rules).
 *
 * The persona rendering and consensus rules are pure, so they're tested with no
 * Session at all. The panel itself is driven by the same scripted
 * {@link FakeClient} the orchestrate tests use. A FakeClient shifts one scripted
 * turn per `generate`, so to keep the candidate order ↔ reply order mapping
 * deterministic we run the panel at `concurrency: 1`: critics then judge in
 * roster order, and a single client scripted PASS/FAIL per persona suffices.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
    personaSystem,
    critic,
    panel,
    panelVerify,
    majorityRule,
    unanimousRule,
    type Personality,
    type CriticVerdict,
} from "../src/critics.ts";
import { FakeClient, textTurn } from "./helpers/fakeClient.ts";

// ── personaSystem: render a Personality into a system prompt ──────────────────

test("personaSystem renders only the fields that are present", () => {
    const thin: Personality = { name: "Dana" };
    const out = personaSystem(thin);
    assert.match(out, /^You are Dana\./);
    // No empty labelled sections for absent fields.
    assert.doesNotMatch(out, /Disposition:/);
    assert.doesNotMatch(out, /Your standards:/);
    assert.doesNotMatch(out, /Your expertise/);
    // The standing "judge in character / end with PASS or FAIL" instruction is
    // always present — it's what makes the persona a verifier.
    assert.match(out, /PASS or FAIL/);
});

test("personaSystem folds in every populated trait and the extra", () => {
    const full: Personality = {
        name: "Priya",
        role: "staff security engineer",
        disposition: "assumes every input is hostile",
        standards: "rejects anything that widens the attack surface unmitigated",
        expertise: "authn and session handling",
        extra: "Always cite a CWE number when you reject.",
    };
    const out = personaSystem(full);
    assert.match(out, /You are Priya, staff security engineer\./);
    assert.match(out, /Your expertise is authn and session handling\./);
    assert.match(out, /Disposition: assumes every input is hostile/);
    assert.match(out, /Your standards: rejects anything that widens/);
    // The escape-hatch text lands after the standing instruction.
    assert.match(out, /cite a CWE number/);
    assert.ok(out.indexOf("PASS or FAIL") < out.indexOf("cite a CWE"));
});

test("critic builds a Session whose system prompt is the rendered persona", () => {
    const p: Personality = { name: "Lee", role: "the on-call paged at 3am" };
    const session = critic(p, { client: new FakeClient([]) });
    // The Session doesn't expose its config, but it must have been constructed
    // without throwing and be ready to verify — exercised end-to-end below.
    assert.ok(session);
});

// ── consensus rules ───────────────────────────────────────────────────────────

/** Build a verdicts list from a terse spec: true=PASS, false=FAIL, null=abstain. */
function verdicts(...votes: (boolean | null)[]): CriticVerdict[] {
    return votes.map((v, i) => ({
        critic: { name: `c${i}` },
        verdict: v === null ? null : { ok: v, rationale: v ? "PASS" : "FAIL" },
    }));
}

test("majorityRule passes only with strictly more PASS than FAIL", () => {
    assert.equal(majorityRule(verdicts(true, true, false)), true);
    assert.equal(majorityRule(verdicts(true, false)), false, "a tie is not consensus");
    assert.equal(majorityRule(verdicts(false, false, true)), false);
    // Abstentions don't vote: 2 PASS, 1 FAIL, 3 abstain → still passes.
    assert.equal(majorityRule(verdicts(true, true, false, null, null, null)), true);
    assert.equal(majorityRule(verdicts(null, null)), false, "all-abstain fails");
});

test("unanimousRule passes only when someone voted and no one dissented", () => {
    assert.equal(unanimousRule(verdicts(true, true, true)), true);
    assert.equal(unanimousRule(verdicts(true, false, true)), false, "one FAIL sinks it");
    // Abstentions are ignored, but at least one real PASS is required.
    assert.equal(unanimousRule(verdicts(true, null, null)), true);
    assert.equal(unanimousRule(verdicts(null, null)), false, "all-abstain fails");
});

// ── panel: run a roster and adjudicate ────────────────────────────────────────

const ROSTER: Personality[] = [
    { name: "Hawk", role: "security hawk" },
    { name: "Pragmatist", role: "ship-it pragmatist" },
    { name: "Pedant", role: "line-by-line pedant" },
];

test("panel collapses critic verdicts by majority and keeps every voice", async () => {
    // Scripted in roster order (concurrency 1): PASS, PASS, FAIL → 2-1 majority.
    const client = new FakeClient([textTurn("PASS"), textTurn("PASS"), textTurn("FAIL")]);
    const result = await panel(ROSTER, { client }, "the candidate work", {
        concurrency: 1,
    });

    assert.equal(result.ok, true);
    assert.equal(result.verdicts.length, 3);
    // Every voice is preserved and attributed, in roster order.
    assert.deepEqual(
        result.verdicts.map((v) => [v.critic.name, v.verdict?.ok]),
        [
            ["Hawk", true],
            ["Pragmatist", true],
            ["Pedant", false],
        ],
    );
});

test("panel honours a custom consensus rule", async () => {
    const client = new FakeClient([textTurn("PASS"), textTurn("PASS"), textTurn("FAIL")]);
    // Same 2-1 split, but unanimous bar → the single FAIL sinks it.
    const result = await panel(ROSTER, { client }, "work", {
        concurrency: 1,
        consensus: unanimousRule,
    });
    assert.equal(result.ok, false);
});

test("panel records a thrown critic as an abstention, not a crash", async () => {
    // An empty-script client makes the critic's verify send throw (FakeClient
    // errors when generate is called more times than scripted). The panel must
    // record that as a null verdict, not let it reject.
    const result = await panel([{ name: "Broken" }], { client: new FakeClient([]) }, "w", {
        concurrency: 1,
    });
    assert.equal(result.verdicts[0]!.verdict, null);
    assert.equal(result.ok, false, "an all-abstention panel does not pass");
});

test("panel rejects a concurrency below 1", async () => {
    await assert.rejects(
        () => panel(ROSTER, { client: new FakeClient([]) }, "w", { concurrency: 0 }),
        /concurrency must be ≥ 1/,
    );
});

// ── panelVerify: a panel standing in for one verifier ─────────────────────────

test("panelVerify folds every critic's reasoning into one Verdict rationale", async () => {
    const client = new FakeClient([
        textTurn("Looks safe. PASS"),
        textTurn("Good enough. PASS"),
        textTurn("Missing a guard. FAIL"),
    ]);
    const v = await panelVerify(ROSTER, { client }, "work", { concurrency: 1 });
    assert.equal(v.ok, true); // 2-1 majority
    // The rationale attributes each opinion by persona name.
    assert.match(v.rationale, /Hawk: PASS — Looks safe\. PASS/);
    assert.match(v.rationale, /Pragmatist: PASS — Good enough\. PASS/);
    assert.match(v.rationale, /Pedant: FAIL — Missing a guard\. FAIL/);
});
