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
    personaIdentity,
    VERDICT_CLAUSE,
    renderStakes,
    critic,
    panel,
    panelVerify,
    majorityRule,
    unanimousRule,
    dealStakes,
    STAKE_POOL,
    type Personality,
    type CriticVerdict,
    type Stake,
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
    // always present: it's what makes the persona a verifier.
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

// ── stakes: the scene a critic carries into the room ──────────────────────────

test("renderStakes contributes nothing for a critic with nothing on the line", () => {
    assert.equal(renderStakes(undefined), "");
    assert.equal(renderStakes([]), "");
});

test("renderStakes frames the stakes as a scene, not a labelled list of biases", () => {
    const out = renderStakes([
        { riding: "the on-call gets paged at 3am", valence: "falsePass" },
        { riding: "the team has waited three days", valence: "falseFail" },
    ]);
    // Each stake appears as its riding text...
    assert.match(out, /the on-call gets paged at 3am/);
    assert.match(out, /the team has waited three days/);
    // ...inhabited in the second person...
    assert.match(out, /depend on you/);
    // ...but the valence is deliberately NOT spelled out: naming the bias would
    // let the model perform it instead of feel it.
    assert.doesNotMatch(out, /falsePass|falseFail|toward FAIL|toward PASS/);
});

test("personaSystem folds stakes in after the standing instruction, before extra", () => {
    const p: Personality = {
        name: "Mara",
        role: "release captain",
        stakes: [{ riding: "the quarter ships behind this merge", valence: "falseFail" }],
        extra: "Cite a line number.",
    };
    const out = personaSystem(p);
    assert.match(out, /the quarter ships behind this merge/);
    // Order: standing PASS/FAIL instruction → stakes scene → extra.
    assert.ok(out.indexOf("PASS or FAIL") < out.indexOf("the quarter ships"));
    assert.ok(out.indexOf("the quarter ships") < out.indexOf("Cite a line number"));
});

test("personaSystem omits the stakes preamble when there are none", () => {
    const out = personaSystem({ name: "Dana" });
    assert.doesNotMatch(out, /depend on you getting this right/);
});

// ── personaIdentity: the person without the verifier framing ──────────────────

test("personaIdentity renders the person but NOT the verdict clause", () => {
    const p: Personality = {
        name: "Priya",
        role: "staff security engineer",
        disposition: "assumes every input is hostile",
        stakes: [{ riding: "the on-call gets paged at 3am", valence: "falsePass" }],
        extra: "Always cite a CWE.",
    };
    const out = personaIdentity(p);
    // The identity, traits, stakes scene, and extra are all present...
    assert.match(out, /You are Priya, staff security engineer\./);
    assert.match(out, /Disposition: assumes every input is hostile/);
    assert.match(out, /the on-call gets paged at 3am/);
    assert.match(out, /Always cite a CWE\./);
    // ...but the PASS/FAIL verdict framing is gone: this is a person, not a
    // verifier. A dreamer minted from this faces its scenario with no stray
    // "end with PASS or FAIL on work" instruction crossing the choice.
    assert.doesNotMatch(out, /PASS or FAIL/);
    assert.doesNotMatch(out, /reviewing work/);
});

test("personaSystem is personaIdentity plus the verdict clause", () => {
    // The split is lossless: personaSystem is exactly the identity with the
    // verdict clause spliced in after the traits, before the stakes/extra tail.
    const p: Personality = {
        name: "Mara",
        role: "release captain",
        disposition: "ships fast, hates ceremony",
        stakes: [{ riding: "the quarter ships behind this", valence: "falseFail" }],
        extra: "Cite a line number.",
    };
    const system = personaSystem(p);
    assert.ok(system.includes(VERDICT_CLAUSE));
    // Removing the clause (and the blank line that joins it) recovers exactly the
    // identity rendering: nothing else differs between the two.
    assert.equal(system.replace(`${VERDICT_CLAUSE}\n\n`, ""), personaIdentity(p));
    // And the clause sits where the old order put it: after the traits (the
    // disposition), before the stakes tail.
    assert.ok(system.indexOf("ships fast") < system.indexOf(VERDICT_CLAUSE));
    assert.ok(system.indexOf(VERDICT_CLAUSE) < system.indexOf("the quarter ships"));
});

// ── STAKE_POOL: the deck dealStakes draws from ────────────────────────────────

test("STAKE_POOL carries both valences so a deal can pull both ways", () => {
    const pass = STAKE_POOL.filter((s) => s.valence === "falsePass").length;
    const fail = STAKE_POOL.filter((s) => s.valence === "falseFail").length;
    assert.ok(pass > 0, "needs falsePass stakes");
    assert.ok(fail > 0, "needs falseFail stakes");
    // Roughly balanced: a one-sided pool would bias every panel the same way.
    assert.ok(Math.abs(pass - fail) <= 1, "pool should be near-evenly split");
});

// ── dealStakes: hand a persona random things to protect ───────────────────────

/** A pinned RNG cycling through the given values, so a deal is deterministic. */
function pinned(...values: number[]): () => number {
    let i = 0;
    return () => values[i++ % values.length]!;
}

test("dealStakes hands a persona stakes drawn from the pool", () => {
    const dealt = dealStakes({ name: "Lee" }, { count: 2, random: pinned(0) });
    assert.equal(dealt.stakes!.length, 2);
    // Every dealt stake came from the pool (drawn, not invented).
    for (const s of dealt.stakes!) {
        assert.ok(STAKE_POOL.some((p) => p.riding === s.riding));
    }
    // It's a copy: the input persona is untouched.
    assert.equal(dealt.name, "Lee");
});

test("dealStakes draws without replacement: no persona gets the same stake twice", () => {
    // Draw the whole pool; every member must be distinct.
    const dealt = dealStakes({ name: "Lee" }, { count: STAKE_POOL.length, random: pinned(0.5) });
    const ridings = dealt.stakes!.map((s) => s.riding);
    assert.equal(new Set(ridings).size, ridings.length, "no duplicates");
    assert.equal(ridings.length, STAKE_POOL.length, "drew the whole pool");
});

test("dealStakes clamps count to the pool size", () => {
    const dealt = dealStakes({ name: "Lee" }, { count: 999, random: pinned(0) });
    assert.equal(dealt.stakes!.length, STAKE_POOL.length);
});

test("dealStakes with count 0 hands over nothing: a critic with no stake", () => {
    const dealt = dealStakes({ name: "Lee" }, { count: 0, random: pinned(0) });
    assert.deepEqual(dealt.stakes, []);
});

test("dealStakes replaces prior stakes rather than appending: a fresh deal each run", () => {
    const carried: Personality = {
        name: "Lee",
        stakes: [{ riding: "an old worry", valence: "falsePass" }],
    };
    const dealt = dealStakes(carried, { count: 1, random: pinned(0) });
    assert.equal(dealt.stakes!.length, 1);
    assert.notEqual(dealt.stakes![0]!.riding, "an old worry");
    // Original persona is not mutated.
    assert.equal(carried.stakes![0]!.riding, "an old worry");
});

test("dealStakes is deterministic under a pinned RNG", () => {
    const a = dealStakes({ name: "Lee" }, { count: 3, random: pinned(0.1, 0.9, 0.3) });
    const b = dealStakes({ name: "Lee" }, { count: 3, random: pinned(0.1, 0.9, 0.3) });
    assert.deepEqual(a.stakes, b.stakes);
});

test("dealStakes rejects a negative count", () => {
    assert.throws(() => dealStakes({ name: "Lee" }, { count: -1 }), /count must be ≥ 0/);
});

test("dealStakes draws from a custom pool when given one", () => {
    const pool: Stake[] = [{ riding: "only this", valence: "falseFail" }];
    const dealt = dealStakes({ name: "Lee" }, { count: 1, pool, random: pinned(0) });
    assert.equal(dealt.stakes![0]!.riding, "only this");
});

test("critic builds a Session whose system prompt is the rendered persona", () => {
    const p: Personality = { name: "Lee", role: "the on-call paged at 3am" };
    const session = critic(p, { client: new FakeClient([]) });
    // The Session doesn't expose its config, but it must have been constructed
    // without throwing and be ready to verify: exercised end-to-end below.
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
    assert.match(v.rationale, /Hawk: PASS: Looks safe\. PASS/);
    assert.match(v.rationale, /Pragmatist: PASS: Good enough\. PASS/);
    assert.match(v.rationale, /Pedant: FAIL: Missing a guard\. FAIL/);
});
