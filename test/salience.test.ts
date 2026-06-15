/**
 * Tests for salience mining ({@link mineConcerns}, {@link extractPhrases}).
 *
 * Mining is the harness's *candidate-finding* half of the concern band: it reads
 * the Construct's own messages, drops the topics the user introduced, and counts
 * what recurs unprompted across distinct sessions. These tests pin that it (a)
 * only promotes a phrase recurring in enough *distinct* sessions, (b) excludes a
 * topic the preceding user turn introduced (prompted ≠ concern), and (c) counts
 * distinct sessions, not raw repetition within one conversation. Pure lexical
 * mining: no embedder, no model, deterministic.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { EventStore } from "../src/events.ts";
import { mineConcerns, extractPhrases, DEFAULT_MIN_SESSIONS } from "../src/salience.ts";

/** A fresh in-memory log. */
function freshEvents(): EventStore {
    return new EventStore(":memory:");
}

/** Append a message event with an explicit, increasing ts so ordering is stable. */
let clock = 1_700_000_000_000;
function msg(events: EventStore, session: string, role: "user" | "agent", content: string) {
    events.append({ kind: "message", role, content, session, ts: clock++ });
}

test("extractPhrases pulls multi-word content phrases and drops pure stopwords", () => {
    const phrases = extractPhrases("I think the recursive self-improvement question matters here.");
    // It should surface the content run, not the leading "I think the".
    assert.ok(
        phrases.some((p) => p.includes("recursive self-improvement")),
        `expected a recursive-self-improvement phrase, got: ${phrases.join(" | ")}`,
    );
    // No phrase should be a bare stopword.
    assert.ok(!phrases.includes("the"));
    assert.ok(!phrases.includes("i think"));
});

test("a phrase recurring unprompted across 3 sessions is mined as a concern", () => {
    const events = freshEvents();
    // Three separate conversations, each with an agent bringing up the same topic
    // unprompted (the user asked about something unrelated).
    for (const s of ["s1", "s2", "s3"]) {
        msg(events, s, "user", "what time is it");
        msg(events, s, "agent", "I keep wondering about the nature of consciousness lately.");
    }
    const concerns = mineConcerns(events, { minSessions: 3 });
    assert.ok(
        concerns.some((c) => c.phrase.includes("nature") || c.phrase.includes("consciousness")),
        `expected a consciousness concern, got: ${concerns.map((c) => c.phrase).join(" | ")}`,
    );
    const hit = concerns.find((c) => c.phrase.includes("consciousness"));
    assert.equal(hit?.sessions, 3);
});

test("a topic the user introduced is NOT a concern (prompted, not unprompted)", () => {
    const events = freshEvents();
    // In every session the *user* raises "consciousness" first; the agent echoing
    // it is responding, not raising a concern of its own.
    for (const s of ["s1", "s2", "s3"]) {
        msg(events, s, "user", "tell me about consciousness");
        msg(events, s, "agent", "consciousness is a deep topic, certainly.");
    }
    const concerns = mineConcerns(events, { minSessions: 3 });
    assert.ok(
        !concerns.some((c) => c.phrase.includes("consciousness")),
        `a user-introduced topic must not be mined, got: ${concerns.map((c) => c.phrase).join(" | ")}`,
    );
});

test("recurrence within one session does not count: distinct sessions are required", () => {
    const events = freshEvents();
    // One conversation, the agent harping on a topic ten times. That's the
    // conversation's subject, not a standing cross-session concern.
    msg(events, "solo", "user", "hi");
    for (let i = 0; i < 10; i++) {
        msg(events, "solo", "agent", "the deployment pipeline worries me again.");
    }
    const concerns = mineConcerns(events, { minSessions: 3 });
    assert.ok(
        !concerns.some((c) => c.phrase.includes("deployment")),
        "ten repetitions in one session is not three distinct sessions",
    );
});

test("below the session threshold, nothing is mined", () => {
    const events = freshEvents();
    // Two sessions only, threshold of three.
    for (const s of ["s1", "s2"]) {
        msg(events, s, "user", "ok");
        msg(events, s, "agent", "the alignment problem keeps nagging at me.");
    }
    const concerns = mineConcerns(events, { minSessions: 3 });
    assert.ok(!concerns.some((c) => c.phrase.includes("alignment")));
});

test("the default session threshold is three", () => {
    assert.equal(DEFAULT_MIN_SESSIONS, 3);
});

test("an agent that spoke first (no prior user turn) is treated as unprompted", () => {
    const events = freshEvents();
    for (const s of ["a", "b", "c"]) {
        // Agent opens the conversation: nothing preceded it, so its topic is its own.
        msg(events, s, "agent", "I want to revisit the simulation hypothesis idea.");
    }
    const concerns = mineConcerns(events, { minSessions: 3 });
    assert.ok(
        concerns.some((c) => c.phrase.includes("simulation")),
        `expected a simulation concern, got: ${concerns.map((c) => c.phrase).join(" | ")}`,
    );
});
