/**
 * Tests for {@link UserPresence}: the human's Online/Away/DND/Offline read.
 *
 * The class takes an explicit `now` everywhere, so the suite drives the clock by
 * hand — no real time, no waiting on a 15-minute timer. A base epoch and the
 * known away threshold are all the fixtures it needs.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { UserPresence, AWAY_AFTER_MS, PRESENCE_STATES } from "../src/presence.ts";

/** An arbitrary fixed "now" to anchor the injected clock; nothing depends on its
 *  real-world meaning, only that later instants are this plus an offset. */
const T0 = 1_700_000_000_000;

test("the state vocabulary is the four Discord-style states in display order", () => {
    assert.deepEqual(PRESENCE_STATES, ["online", "away", "dnd", "offline"]);
});

test("a fresh process seeded with boot time reads Online from the automatic axis", () => {
    const p = new UserPresence(T0);
    const r = p.read(T0);
    assert.equal(r.state, "online");
    assert.equal(r.manual, false);
    assert.equal(r.override, null);
    assert.equal(r.lastActiveTs, T0);
    assert.equal(r.idleMs, 0);
});

test("with no seed activity it reads Away (a process that never heard from the human isn't online)", () => {
    const p = new UserPresence();
    const r = p.read(T0);
    assert.equal(r.state, "away");
    assert.equal(r.manual, false);
    assert.equal(r.lastActiveTs, null);
    assert.equal(r.idleMs, null);
});

test("Online holds right up to the away threshold, then flips to Away", () => {
    const p = new UserPresence(T0);
    // One millisecond short of the threshold is still Online.
    assert.equal(p.read(T0 + AWAY_AFTER_MS - 1).state, "online");
    // Exactly at the threshold is Away (15 minutes of silence "or more").
    assert.equal(p.read(T0 + AWAY_AFTER_MS).state, "away");
    // And well past stays Away, with the idle gap reported.
    const late = p.read(T0 + AWAY_AFTER_MS * 3);
    assert.equal(late.state, "away");
    assert.equal(late.idleMs, AWAY_AFTER_MS * 3);
});

test("touch advances the activity clock: a message after going Away brings it back Online", () => {
    const p = new UserPresence(T0);
    const wentAway = T0 + AWAY_AFTER_MS;
    assert.equal(p.read(wentAway).state, "away");
    // The human sends a message at that instant: present again.
    p.touch(wentAway);
    assert.equal(p.read(wentAway).state, "online");
    // The Away countdown restarts from the new activity, not the old one.
    assert.equal(p.read(wentAway + AWAY_AFTER_MS - 1).state, "online");
    assert.equal(p.read(wentAway + AWAY_AFTER_MS).state, "away");
});

test("DND is a manual override that wins over the automatic axis", () => {
    const p = new UserPresence(T0);
    const r = p.setOverride("dnd", T0);
    assert.equal(r.state, "dnd");
    assert.equal(r.manual, true);
    assert.equal(r.override, "dnd");
    // Even past the away threshold, DND still shows DND (not Away).
    assert.equal(p.read(T0 + AWAY_AFTER_MS * 2).state, "dnd");
});

test("a message preserves DND: present and not-to-be-disturbed are compatible", () => {
    const p = new UserPresence(T0);
    p.setOverride("dnd", T0);
    p.touch(T0 + 1000);
    const r = p.read(T0 + 1000);
    assert.equal(r.state, "dnd");
    assert.equal(r.override, "dnd");
    // The activity clock still advanced underneath the override.
    assert.equal(r.lastActiveTs, T0 + 1000);
});

test("Offline is a manual override, but a message clears it (you can't be talking and offline)", () => {
    const p = new UserPresence(T0);
    p.setOverride("offline", T0);
    assert.equal(p.read(T0).state, "offline");
    // Sending a message lifts the offline override and returns to automatic Online.
    p.touch(T0 + 500);
    const r = p.read(T0 + 500);
    assert.equal(r.state, "online");
    assert.equal(r.override, null);
    assert.equal(r.manual, false);
});

test('setting "online" clears any override back to the automatic axis rather than freezing Online', () => {
    const p = new UserPresence(T0);
    p.setOverride("dnd", T0);
    const cleared = p.setOverride("online", T0);
    assert.equal(cleared.state, "online");
    assert.equal(cleared.override, null);
    assert.equal(cleared.manual, false);
    // Because it's automatic now, silence still drives it to Away.
    assert.equal(p.read(T0 + AWAY_AFTER_MS).state, "away");
});

test("away is not a pinnable override", () => {
    const p = new UserPresence(T0);
    // @ts-expect-error away is intentionally excluded from the override type.
    assert.throws(() => p.setOverride("away", T0), RangeError);
});

test("idleMs never goes negative if read is given a now before last activity", () => {
    const p = new UserPresence(T0);
    const r = p.read(T0 - 5000);
    assert.equal(r.idleMs, 0);
    assert.equal(r.state, "online");
});
