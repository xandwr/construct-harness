/**
 * Tests for the working mind ({@link WorkingMind} and {@link workingMindContext}).
 *
 * The working mind is the Construct's recent state, pushed onto every turn so it
 * doesn't wake up cold. These tests pin the mechanism in isolation: promotion and
 * reinforcement, recency+reinforcement decay, per-band eviction, the rendered
 * shape, and the provider's silence when empty. The Session-level integration
 * (state actually carrying forward across sends, and cooling out when dropped)
 * lives in session.test.ts.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { WorkingMind, workingMindContext } from "../src/workingMind.ts";

test("an empty mind renders null and the provider stays silent", () => {
    const mind = new WorkingMind();
    assert.equal(mind.render(), null);
    assert.equal(workingMindContext(mind).contribute({ messages: [], turn: 0 }), undefined);
});

test("a noted thought is held and rendered in its band", () => {
    const mind = new WorkingMind();
    mind.note("thought", "the retry policy is the suspect");
    const text = mind.render();
    assert.ok(text);
    assert.match(text!, /train of thought/i);
    assert.match(text!, /the retry policy is the suspect/);
});

test("the provider pushes the rendered mind as a system contribution", () => {
    const mind = new WorkingMind();
    mind.note("memory", "user is allergic to peanuts");
    const contribution = workingMindContext(mind).contribute({ messages: [], turn: 0 });
    assert.ok(contribution && contribution.system);
    assert.match(contribution!.system!, /user is allergic to peanuts/);
});

test("blank notes are ignored", () => {
    const mind = new WorkingMind();
    mind.note("thought", "   \n  ");
    assert.equal(mind.render(), null);
    assert.equal(mind.snapshot().length, 0);
});

test("re-noting the same thought reinforces (one item), not duplicates", () => {
    const mind = new WorkingMind();
    mind.note("thought", "Same Thought");
    mind.note("thought", "same   thought"); // differs only in case/whitespace
    const snap = mind.snapshot();
    assert.equal(snap.length, 1, "normalized-equal thoughts collapse to one item");
    assert.equal(snap[0]!.warmth, 1);
});

test("reinforcement keeps the freshest phrasing of the same key", () => {
    const mind = new WorkingMind();
    mind.note("memory", "old wording", "m7");
    mind.note("memory", "new wording", "m7"); // same store id, edited content
    const snap = mind.snapshot();
    assert.equal(snap.length, 1);
    assert.equal(snap[0]!.text, "new wording");
});

test("an item noted this turn is exempt from this turn's decay", () => {
    const mind = new WorkingMind({ decay: 0.5 });
    mind.note("thought", "fresh");
    mind.tick(); // ages the turn; the just-noted item is exempt
    assert.equal(mind.snapshot()[0]!.warmth, 1, "noted-this-turn item didn't cool yet");
});

test("an idle item cools by the decay factor each turn", () => {
    const mind = new WorkingMind({ decay: 0.5, floor: 0.01 });
    mind.note("thought", "idle");
    mind.tick(); // turn 0 -> 1: exempt (noted this turn)
    mind.tick(); // turn 1 -> 2: cools once
    assert.equal(mind.snapshot()[0]!.warmth, 0.5);
    mind.tick(); // cools again
    assert.equal(mind.snapshot()[0]!.warmth, 0.25);
});

test("an item drops out once it cools below the floor", () => {
    const mind = new WorkingMind({ decay: 0.5, floor: 0.3 });
    mind.note("thought", "fading");
    mind.tick(); // exempt
    mind.tick(); // 1.0 -> 0.5 (>= floor, stays)
    assert.equal(mind.snapshot().length, 1);
    mind.tick(); // 0.5 -> 0.25 (< 0.3, dropped)
    assert.equal(mind.snapshot().length, 0);
    assert.equal(mind.render(), null);
});

test("a recurring thought stays warm while a dropped one fades (the whole thesis)", () => {
    const mind = new WorkingMind({ decay: 0.5, floor: 0.2 });
    mind.note("thought", "kept alive");
    mind.note("thought", "let go");
    for (let i = 0; i < 4; i++) {
        mind.note("thought", "kept alive"); // reinforced every turn
        mind.tick();
    }
    const snap = mind.snapshot();
    const texts = snap.map((i) => i.text);
    assert.ok(texts.includes("kept alive"), "the reinforced thought is still held");
    assert.ok(!texts.includes("let go"), "the abandoned thought cooled out");
});

test("a band over its cap evicts the coldest, keeping the warmest", () => {
    const mind = new WorkingMind({ decay: 0.9, floor: 0.01, capPerBand: 2 });
    // Three thoughts, each noted on its own turn so warmth differs by age.
    mind.note("thought", "oldest");
    mind.tick();
    mind.note("thought", "middle");
    mind.tick();
    mind.note("thought", "newest");
    mind.tick();
    const texts = mind.snapshot().map((i) => i.text);
    assert.equal(texts.length, 2, "capped at 2 per band");
    assert.ok(texts.includes("newest") && texts.includes("middle"), "warmest two kept");
    assert.ok(!texts.includes("oldest"), "coldest evicted");
});

test("bands are independently capped and render separately", () => {
    const mind = new WorkingMind({ capPerBand: 3 });
    mind.note("thought", "t1");
    mind.note("memory", "m1");
    mind.note("memory", "m2");
    const text = mind.render()!;
    assert.match(text, /train of thought[\s\S]*t1/i);
    assert.match(text, /surfaced[\s\S]*m1/i);
    assert.match(text, /m2/);
    // A thought-only mind omits the memory section, and vice versa.
    const t = new WorkingMind();
    t.note("thought", "only a thought");
    assert.doesNotMatch(t.render()!, /surfaced/i);
});

test("snapshot is warmest-first", () => {
    const mind = new WorkingMind({ decay: 0.5 });
    mind.note("thought", "cooler");
    mind.tick(); // cooler is now subject to decay next tick
    mind.tick(); // cooler cools to 0.5
    mind.note("thought", "warmer"); // fresh at 1.0
    const snap = mind.snapshot();
    assert.equal(snap[0]!.text, "warmer");
    assert.equal(snap[1]!.text, "cooler");
});

test("render does not leak the warmth number (mechanism stays hidden)", () => {
    const mind = new WorkingMind();
    mind.note("thought", "held thing");
    const text = mind.render()!;
    assert.doesNotMatch(text, /warmth|0\.\d|\bdecay\b/i);
});
