/**
 * Tests for the dream loop ({@link sampleScenario}, {@link dreamOnce},
 * {@link dreamLoop}): the layer where dreaming stops being a persona factory and
 * becomes the feature.
 *
 * A dream drives several model turns in a fixed order (persona generation, then
 * scenario abstraction when one isn't supplied, then the persona's choice), so
 * the scripted {@link FakeClient} queue is ordered to match. Provenance and the
 * appended `dream` event span the MemoryStore and the EventStore, so the stores
 * that need to share rows share one temp file (a `:memory:` db is per-connection).
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
    sampleScenario,
    dreamOnce,
    dreamLoop,
    DREAM_EVENT_KIND,
    SCENARIO_SYSTEM,
    PersonaError,
} from "../src/dreaming.ts";
import type { Dream } from "../src/dreaming.ts";
import { Memory, MemoryStore } from "../src/memory.ts";
import { EventStore } from "../src/events.ts";
import { personaIdentity } from "../src/critics.ts";
import { FakeClient, textTurn } from "./helpers/fakeClient.ts";
import type { ScriptedTurn } from "../src/testing.ts";

/** The text of the system turn carried by a recorded FakeClient call. The system
 *  prompt rides as a {@link Message} with `sender.role === "system"`. */
function systemTextOf(call: {
    messages: { sender: { role: string }; content: { kind: string; text?: string }[] }[];
}): string {
    const system = call.messages.find((m) => m.sender.role === "system");
    assert.ok(system, "a system turn should be present");
    return system!.content.map((p) => (p.kind === "text" ? p.text : "")).join("");
}

/** A persona reply turn, as the model would emit it (fenced JSON). */
function personaTurn(name: string): ScriptedTurn {
    return textTurn(`\`\`\`json\n{"name":"${name}"}\n\`\`\``);
}

/** Open a memory store and an event log over one shared temp file, run `fn`, and
 *  clean both up. Shared because provenance and dream events span both. */
async function withStores(
    fn: (store: MemoryStore, events: EventStore) => Promise<void>,
): Promise<void> {
    const dir = mkdtempSync(join(tmpdir(), "dreamloop-"));
    const path = join(dir, "shared.sqlite");
    const store = new MemoryStore(path);
    const events = new EventStore(path);
    try {
        await fn(store, events);
    } finally {
        events.close();
        store.close();
        rmSync(dir, { recursive: true, force: true });
    }
}

function seedCorpus(store: MemoryStore): void {
    store.save(new Memory({ content: "user ships infra on Fridays", importance: 0.9 }));
    store.save(new Memory({ content: "user mentors a nervous junior", importance: 0.8 }));
    store.save(new Memory({ content: "user was burned by a silent data corruption" }));
}

// ── sampleScenario ────────────────────────────────────────────────────────────

test("sampleScenario abstracts a scenario from the corpus and records its sources", async () => {
    await withStores(async (store) => {
        seedCorpus(store);
        const client = new FakeClient([textTurn("You must decide whether to ship. Choose.")]);

        const scenario = await sampleScenario(store, {
            client,
            sampleSize: 2,
            random: () => 0, // deterministic sample
        });

        assert.equal(scenario.prompt, "You must decide whether to ship. Choose.");
        assert.equal(scenario.sourceMemoryIds.length, 2, "two memories should ground the scenario");

        // The abstraction turn ran under SCENARIO_SYSTEM with no passive context.
        const system = client.calls[0]!.messages.find((m) => m.sender.role === "system");
        const systemText = system!.content.map((p) => (p.kind === "text" ? p.text : "")).join("");
        assert.equal(systemText, SCENARIO_SYSTEM);
    });
});

test("sampleScenario falls back to a generic scenario on an empty corpus", async () => {
    await withStores(async (store) => {
        // Empty corpus: no model turn is needed, so an empty script is fine.
        const client = new FakeClient([]);
        const scenario = await sampleScenario(store, { client });

        assert.equal(scenario.sourceMemoryIds.length, 0);
        assert.match(scenario.prompt, /Choose/);
        assert.equal(client.calls.length, 0, "no model turn should run for an empty corpus");
    });
});

test("sampleScenario falls back when the model returns an empty scenario", async () => {
    await withStores(async (store) => {
        seedCorpus(store);
        const client = new FakeClient([textTurn("   ")]); // whitespace-only reply
        const scenario = await sampleScenario(store, { client, random: () => 0 });
        // Source ids still recorded; prompt degrades to the fallback rather than "".
        assert.ok(scenario.sourceMemoryIds.length > 0);
        assert.match(scenario.prompt, /Choose/);
    });
});

// ── dreamOnce ─────────────────────────────────────────────────────────────────

test("dreamOnce conjures a persona, faces a sampled scenario, and logs the choice", async () => {
    await withStores(async (store, events) => {
        seedCorpus(store);
        // Order matters: persona, then scenario, then the persona's choice.
        const client = new FakeClient([
            personaTurn("Mara"),
            textTurn("You must decide whether to ship under pressure. Choose."),
            textTurn("I hold it to verify. Reasoning... FAIL"),
        ]);

        const dream = await dreamOnce({ client, store, events, random: () => 0 });

        assert.equal(dream.persona.name, "Mara");
        assert.match(dream.scenario.prompt, /decide whether to ship/);
        assert.match(dream.choice, /I hold it to verify/);

        // The choice was appended as exactly one dream event.
        const logged = events.recent({ kind: DREAM_EVENT_KIND });
        assert.equal(logged.length, 1);
        const ev = logged[0]!;
        assert.equal(ev.id, dream.event.id);
        assert.equal(ev.kind, DREAM_EVENT_KIND);
        assert.equal(ev.role, "agent");
        assert.match(ev.content, /I hold it to verify/);

        // meta carries the structured record: who dreamed, what they faced, and
        // which memories grounded it.
        const meta = ev.meta as {
            persona: { name: string };
            scenario: string;
            sourceMemoryIds: number[];
        };
        assert.equal(meta.persona.name, "Mara");
        assert.match(meta.scenario, /decide whether to ship/);
        assert.deepEqual(meta.sourceMemoryIds, dream.scenario.sourceMemoryIds);
    });
});

test("dreamOnce reuses a supplied scenario and skips corpus sampling", async () => {
    await withStores(async (store, events) => {
        seedCorpus(store);
        // No scenario turn in the script: a supplied scenario means only persona
        // generation and the choice turn run.
        const client = new FakeClient([personaTurn("Dana"), textTurn("I approve it. PASS")]);

        const dream = await dreamOnce({
            client,
            store,
            events,
            scenario: { prompt: "Pre-built dilemma. Choose.", sourceMemoryIds: [42] },
        });

        assert.equal(client.calls.length, 2, "only persona + choice turns should run");
        assert.equal(dream.scenario.prompt, "Pre-built dilemma. Choose.");
        assert.deepEqual(dream.scenario.sourceMemoryIds, [42]);
        assert.match(dream.choice, /I approve it/);

        // The dreamer faces the scenario AS the persona but NOT as a verifier: its
        // system prompt is the persona's identity, with no PASS/FAIL verdict clause
        // to cross the "choose, and say why" scenario. (calls[1] is the choice
        // turn; calls[0] was persona generation.)
        const choiceSystem = systemTextOf(client.calls[1]!);
        assert.equal(choiceSystem, personaIdentity(dream.persona));
        assert.doesNotMatch(choiceSystem, /PASS or FAIL/);
    });
});

test("dreamOnce deals stakes to the dreamer when asked", async () => {
    await withStores(async (store, events) => {
        const client = new FakeClient([personaTurn("Sam"), textTurn("My call: PASS")]);
        const dream = await dreamOnce({
            client,
            store,
            events,
            scenario: { prompt: "A dilemma. Choose.", sourceMemoryIds: [] },
            deal: { count: 1, random: () => 0 },
        });
        assert.ok(dream.persona.stakes, "the dreamer should arrive stake-dealt");
        assert.equal(dream.persona.stakes!.length, 1);
        // The dealt stake rides into the event's persona record too.
        const meta = dream.event.meta as { persona: { stakes?: unknown[] } };
        assert.equal(meta.persona.stakes!.length, 1);
    });
});

test("dreamOnce surfaces a PersonaError when the dreamer won't parse", async () => {
    await withStores(async (store, events) => {
        const client = new FakeClient([textTurn("I'd rather not invent anyone.")]);
        await assert.rejects(
            dreamOnce({
                client,
                store,
                events,
                scenario: { prompt: "x", sourceMemoryIds: [] },
            }),
            PersonaError,
        );
        // Nothing was logged: a dream that never happened leaves no event.
        assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 0);
    });
});

// ── dreamLoop ─────────────────────────────────────────────────────────────────

test("dreamLoop runs N dreams and collects them in order", async () => {
    await withStores(async (store, events) => {
        // Two dreams, each: persona + choice (scenario is supplied, so no scenario
        // turn). Queue is consumed front to back across both dreams.
        const client = new FakeClient([
            personaTurn("One"),
            textTurn("first choice PASS"),
            personaTurn("Two"),
            textTurn("second choice FAIL"),
        ]);

        const seen: number[] = [];
        const result = await dreamLoop({
            client,
            store,
            events,
            scenario: { prompt: "Shared dilemma. Choose.", sourceMemoryIds: [] },
            count: 2,
            onDream: ({ index }) => seen.push(index),
        });

        assert.equal(result.dreams.length, 2);
        assert.equal(result.failures.length, 0);
        assert.deepEqual(
            result.dreams.map((d: Dream) => d.persona.name),
            ["One", "Two"],
        );
        assert.deepEqual(seen, [0, 1]);
        // Both choices reached the log.
        assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 2);
    });
});

test("dreamLoop tolerates a bad dream: records the failure and rolls on", async () => {
    await withStores(async (store, events) => {
        // Dream 1's persona reply is junk (a PersonaError); dream 2 is well-formed.
        const client = new FakeClient([
            textTurn("no persona here"), // dream 1: fails to parse
            personaTurn("Good"), // dream 2: persona
            textTurn("a real choice PASS"), // dream 2: choice
        ]);

        const result = await dreamLoop({
            client,
            store,
            events,
            scenario: { prompt: "Dilemma. Choose.", sourceMemoryIds: [] },
            count: 2,
        });

        assert.equal(result.dreams.length, 1, "the good dream survived");
        assert.equal(result.dreams[0]!.persona.name, "Good");
        assert.equal(result.failures.length, 1, "the bad dream is recorded, not thrown");
        assert.equal(result.failures[0]!.index, 0);
        assert.ok(result.failures[0]!.error instanceof PersonaError);
        // Only the good dream reached the log.
        assert.equal(events.recent({ kind: DREAM_EVENT_KIND }).length, 1);
    });
});

test("dreamLoop rejects a count below 1", async () => {
    await withStores(async (store, events) => {
        const client = new FakeClient([]);
        await assert.rejects(
            dreamLoop({
                client,
                store,
                events,
                scenario: { prompt: "x", sourceMemoryIds: [] },
                count: 0,
            }),
            RangeError,
        );
    });
});
