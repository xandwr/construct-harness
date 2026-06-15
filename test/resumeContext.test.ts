/**
 * Tests for the resume catch-up provider ({@link resumeContext}).
 *
 * The catch-up fires once — turn 0 of a conversation resumed after a real gap —
 * and assembles a "while you were away" block from the dream and goal-change
 * events that accumulated during the absence. These tests pin the gate (turn,
 * session-start, gap), the assembly (dreams listed, goals summarized), and the
 * silences (no gap, no events, no session start). Pure log assembly: no model.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { EventStore } from "../src/events.ts";
import { resumeContext, RESUME_THRESHOLD_MS } from "../src/resumeContext.ts";
import { DREAM_EVENT_KIND } from "../src/dreaming.ts";
import { GOAL_EVENT_KIND } from "../src/goals.ts";
import { RoleType } from "../src/types.ts";
import type { Message } from "../src/types.ts";

const T0 = 1_700_000_000_000;

function freshEvents(): EventStore {
    return new EventStore(":memory:");
}

/** A conversation whose *prior* activity is `ts`, followed by the current turn
 *  stamped ~now — mirroring the real shape a provider sees (the current user turn
 *  is already the tail; the gap is measured against the message before it). So the
 *  provider reads a gap of (now - ts). */
function convoEndingAt(ts: number): Message[] {
    return [
        {
            sender: { role: RoleType.Agent },
            timestamp: ts,
            content: [{ kind: "text", text: "earlier" }],
        },
        {
            sender: { role: RoleType.User },
            timestamp: Date.now(),
            content: [{ kind: "text", text: "hi" }],
        },
    ];
}

/** Append a dream event at `ts` with a persona + scenario. */
function dream(events: EventStore, ts: number, name: string, scenario: string, choice: string) {
    events.append({
        kind: DREAM_EVENT_KIND,
        role: "agent",
        content: choice,
        meta: { persona: { name }, scenario, sourceMemoryIds: [] },
        ts,
    });
}

/** Append a goal-change event at `ts`. */
function goalChange(
    events: EventStore,
    ts: number,
    change: "created" | "status",
    status: "active" | "done" | "abandoned",
) {
    events.append({
        kind: GOAL_EVENT_KIND,
        role: "agent",
        content: `Goal ${change}`,
        meta: { change, goalId: 1, status },
        ts,
    });
}

test("after a long gap, dreams and goal changes are summarized in the catch-up", () => {
    const events = freshEvents();
    // Anchor the gap window against the real clock (the provider reads Date.now):
    // last activity 31 minutes ago, with dreams/goal-changes falling inside the
    // window [lastActive, now].
    const lastActive = Date.now() - (RESUME_THRESHOLD_MS + 60_000);
    dream(
        events,
        lastActive + 1000,
        "Vera Ostrakh",
        "You must decide whether to reveal a flawed map.",
        "She revealed it.",
    );
    dream(
        events,
        lastActive + 2000,
        "Kenji Watanabe",
        "You choose rigor over speed under deadline.",
        "He chose rigor.",
    );
    goalChange(events, lastActive + 3000, "status", "done");

    const provider = resumeContext(events);
    const contribution = provider.contribute({
        messages: convoEndingAt(lastActive),
        turn: 0,
        sessionStart: T0,
    });
    assert.ok(contribution?.system, "a catch-up block should be produced after a long gap");
    const text = contribution!.system!;
    assert.match(text, /while you were away/i);
    assert.match(text, /Vera Ostrakh/);
    assert.match(text, /Kenji Watanabe/);
    assert.match(text, /goal.*completed/i);
    // It must not be a model retelling: the choice prose ("She revealed it") is
    // not what the catch-up surfaces — only persona + theme.
    assert.doesNotMatch(text, /She revealed it/);
});

test("no catch-up when the gap is below the threshold", () => {
    const events = freshEvents();
    dream(events, Date.now() - 1000, "Someone", "a scenario", "a choice");
    const provider = resumeContext(events);
    const contribution = provider.contribute({
        messages: convoEndingAt(Date.now() - 60_000), // one minute: well under 30
        turn: 0,
        sessionStart: T0,
    });
    assert.equal(contribution, undefined);
});

test("no catch-up on a non-opening turn", () => {
    const events = freshEvents();
    dream(events, T0, "Someone", "a scenario", "a choice");
    const provider = resumeContext(events);
    const contribution = provider.contribute({
        messages: convoEndingAt(Date.now() - (RESUME_THRESHOLD_MS + 60_000)),
        turn: 1, // not the first turn
        sessionStart: T0,
    });
    assert.equal(contribution, undefined);
});

test("no catch-up without a session start (a throwaway session never catches up)", () => {
    const events = freshEvents();
    dream(events, T0, "Someone", "a scenario", "a choice");
    const provider = resumeContext(events);
    const contribution = provider.contribute({
        messages: convoEndingAt(Date.now() - (RESUME_THRESHOLD_MS + 60_000)),
        turn: 0,
        // sessionStart omitted
    });
    assert.equal(contribution, undefined);
});

test("a long gap with nothing logged stays silent (nothing to catch up on)", () => {
    const events = freshEvents();
    // No dreams, no goal changes during the gap.
    const provider = resumeContext(events);
    const contribution = provider.contribute({
        messages: convoEndingAt(Date.now() - (RESUME_THRESHOLD_MS + 60_000)),
        turn: 0,
        sessionStart: T0,
    });
    assert.equal(contribution, undefined);
});

test("dreams from BEFORE the gap are not re-surfaced (only the absence window)", () => {
    const events = freshEvents();
    const lastActive = Date.now() - (RESUME_THRESHOLD_MS + 60_000);
    // A dream from well before the last activity: it predates the gap.
    dream(events, lastActive - 1_000_000, "OldDreamer", "an old scenario", "an old choice");
    const provider = resumeContext(events);
    const contribution = provider.contribute({
        messages: convoEndingAt(lastActive),
        turn: 0,
        sessionStart: T0,
    });
    // The only dream predates the window, and nothing else happened, so silence.
    assert.equal(contribution, undefined);
});

test("a fresh conversation with no prior activity never catches up", () => {
    const events = freshEvents();
    dream(events, T0, "Someone", "a scenario", "a choice");
    const provider = resumeContext(events);
    const contribution = provider.contribute({
        messages: [], // no messages: nothing to resume after
        turn: 0,
        sessionStart: T0,
    });
    assert.equal(contribution, undefined);
});
