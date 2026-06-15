/**
 * Tests for the downtime daemon ({@link DowntimeDaemon}).
 *
 * The daemon bridges presence to downtime work: dream when the human is genuinely
 * away, stop when they're back, cap the spend, never crash on a bad dream. These
 * tests drive `tick()` by hand (no interval, no real waiting) against a scripted
 * client and an injected presence clock, pinning each clause of the contract:
 * the idle grace, the away gate, the per-session cap, the human-returned reset,
 * and the concern-mining side effect.
 *
 * Each dream runs two model turns here: a persona JSON, then the persona's choice
 * (the corpus is empty, so scenario sampling uses the fallback with no model
 * turn). The FakeClient is scripted accordingly.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { MemoryStore } from "../src/memory.ts";
import { EventStore } from "../src/events.ts";
import { UserPresence } from "../src/presence.ts";
import { DowntimeDaemon } from "../src/downtimeDaemon.ts";
import { DREAM_EVENT_KIND } from "../src/dreaming.ts";
import { FakeClient, textTurn } from "./helpers/fakeClient.ts";

/** One dream's worth of scripted turns: a persona, then a choice. */
function dreamTurns() {
    return [
        textTurn('```json\n{"name":"A Dreamer","role":"a wanderer"}\n```'),
        textTurn("I would choose to wait and see."),
    ];
}

/** A FakeClient scripted for `n` dreams. */
function clientForDreams(n: number): FakeClient {
    const turns = [];
    for (let i = 0; i < n; i++) turns.push(...dreamTurns());
    return new FakeClient(turns);
}

/** Presence that reads away-with-long-idle: boot far in the past, never touched. */
function awayPresence(): UserPresence {
    // No boot seed ⇒ lastActiveTs is null ⇒ read() is Away, but idleMs is null,
    // which the daemon treats as "no measurable idle" and skips. So seed a boot
    // time and rely on a later read; the daemon reads Date.now(), so seed boot
    // well in the past.
    return new UserPresence(Date.now() - 60 * 60 * 1000); // an hour ago
}

function deps(client: FakeClient, presence: UserPresence) {
    const store = new MemoryStore(":memory:");
    const events = new EventStore(":memory:");
    return { store, events, presence };
}

test("an away human past the idle grace gets one dream per tick", async () => {
    const presence = awayPresence();
    const client = clientForDreams(1);
    const { store, events } = deps(client, presence);
    const daemon = new DowntimeDaemon({
        presence,
        events,
        store,
        client,
        minIdleMs: 60_000, // an hour idle is well past this
    });

    await daemon.tick();
    const dreams = events.recent({ kind: DREAM_EVENT_KIND });
    assert.equal(dreams.length, 1, "one tick while away ⇒ one dream");
});

test("an online human triggers no dreaming", async () => {
    // Boot at 'now' so presence reads Online (idle ~0).
    const presence = new UserPresence(Date.now());
    const client = clientForDreams(1);
    const { store, events } = deps(client, presence);
    const daemon = new DowntimeDaemon({ presence, events, store, client, minIdleMs: 60_000 });

    await daemon.tick();
    assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 0);
});

test("away but not yet past the idle grace does not dream", async () => {
    // Boot 20 minutes ago: presence is Away (past the 15m presence threshold), but
    // the daemon's minIdleMs is 30m, so it should hold off.
    const presence = new UserPresence(Date.now() - 20 * 60 * 1000);
    const client = clientForDreams(1);
    const { store, events } = deps(client, presence);
    const daemon = new DowntimeDaemon({
        presence,
        events,
        store,
        client,
        minIdleMs: 30 * 60 * 1000,
    });
    await daemon.tick();
    assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 0);
});

test("dreams are capped per downtime session", async () => {
    const presence = awayPresence();
    const client = clientForDreams(5); // script more than the cap, to be safe
    const { store, events } = deps(client, presence);
    const daemon = new DowntimeDaemon({
        presence,
        events,
        store,
        client,
        minIdleMs: 60_000,
        maxPerSession: 2,
    });
    // Tick more times than the cap; only `maxPerSession` dreams should run.
    for (let i = 0; i < 4; i++) await daemon.tick();
    assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 2);
});

test("the cap resets after the human returns and leaves again (a new downtime session)", async () => {
    // First absence: away the whole time, cap of 1.
    const away = new UserPresence(Date.now() - 60 * 60 * 1000);
    const client = clientForDreams(3);
    const { store, events } = deps(client, away);
    const daemon = new DowntimeDaemon({
        presence: away,
        events,
        store,
        client,
        minIdleMs: 60_000,
        maxPerSession: 1,
    });
    await daemon.tick();
    await daemon.tick();
    assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 1, "cap holds within a session");

    // Human returns: a touch makes presence Online, so the next tick resets the
    // per-session counters.
    away.touch(Date.now());
    await daemon.tick(); // sees online, resets, no dream
    assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 1);

    // They leave again: backdate last-activity so presence is Away with long idle.
    away.touch(Date.now() - 60 * 60 * 1000);
    await daemon.tick();
    assert.equal(
        events.recent({ kind: DREAM_EVENT_KIND }).length,
        2,
        "a fresh absence re-arms the cap and dreams again",
    );
});

test("a bad dream is swallowed, not fatal (degrade, don't crash)", async () => {
    const presence = awayPresence();
    // A persona reply with no JSON object ⇒ PersonaError inside dreamOnce.
    const client = new FakeClient([textTurn("I refuse to invent anyone.")]);
    const { store, events } = deps(client, presence);
    const daemon = new DowntimeDaemon({ presence, events, store, client, minIdleMs: 60_000 });
    await assert.doesNotReject(() => daemon.tick());
    assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 0, "no dream was recorded");
});

test("mining runs after a dream and exposes concern candidates", async () => {
    const presence = awayPresence();
    const client = clientForDreams(1);
    const { store, events } = deps(client, presence);
    // Pre-seed the log with a recurring unprompted topic across three sessions so
    // mining has something to find.
    let ts = Date.now() - 10_000;
    for (const s of ["s1", "s2", "s3"]) {
        events.append({ kind: "message", role: "user", content: "hello", session: s, ts: ts++ });
        events.append({
            kind: "message",
            role: "agent",
            content: "I keep thinking about the recursion problem.",
            session: s,
            ts: ts++,
        });
    }
    let mined: string[] | undefined;
    const daemon = new DowntimeDaemon({
        presence,
        events,
        store,
        client,
        minIdleMs: 60_000,
        mining: { minSessions: 3 },
        onConcerns: (c) => {
            mined = c.map((x) => x.text);
        },
    });
    await daemon.tick();
    assert.ok(mined, "onConcerns should have fired after the dream");
    assert.ok(
        daemon.concerns().some((c) => c.includes("recursion")),
        `expected a recursion concern, got: ${daemon.concerns().join(" | ")}`,
    );
});

test("start()/stop() are idempotent and stop disarms further work", async () => {
    const presence = awayPresence();
    const client = clientForDreams(2);
    const { store, events } = deps(client, presence);
    const daemon = new DowntimeDaemon({ presence, events, store, client, minIdleMs: 60_000 });
    daemon.start();
    daemon.start(); // second start is a no-op (doesn't throw or double-arm)
    daemon.stop();
    daemon.stop(); // idempotent
    // Manual tick still works after stop (it just disarmed the interval).
    await daemon.tick();
    assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 1);
});
