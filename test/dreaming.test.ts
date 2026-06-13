/**
 * Tests for the persona factory ({@link extractJsonObject},
 * {@link parsePersonality}, {@link parsePersonaReply}, {@link generatePersona}).
 *
 * The parsing layer is pure, so it's tested with no Session at all and the bulk
 * of the hardening lives here: a model returns junk, and every junk shape must
 * become a recoverable {@link PersonaError} rather than an unhandled throw or a
 * malformed persona. {@link generatePersona} is then driven by the scripted
 * {@link FakeClient}, asserting it parses the reply, deals stakes when asked,
 * and varies its prompt by the seed.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
    extractJsonObject,
    parsePersonality,
    parsePersonaReply,
    generatePersona,
    PersonaError,
    PERSONA_SYSTEM,
} from "../src/dreaming.ts";
import { personaSystem, STAKE_POOL } from "../src/critics.ts";
import { FakeClient, textTurn } from "./helpers/fakeClient.ts";

// ── extractJsonObject: pull the first balanced object from prose ──────────────

test("extractJsonObject returns a bare object unchanged", () => {
    assert.equal(extractJsonObject('{"name":"Mara"}'), '{"name":"Mara"}');
});

test("extractJsonObject strips surrounding prose and a fence", () => {
    const text = 'Sure! Here is the person:\n```json\n{"name":"Mara"}\n```\nHope that helps.';
    assert.equal(extractJsonObject(text), '{"name":"Mara"}');
});

test("extractJsonObject handles nested objects", () => {
    const text = 'noise {"a":{"b":{"c":1}},"d":2} trailing';
    assert.equal(extractJsonObject(text), '{"a":{"b":{"c":1}},"d":2}');
});

test("extractJsonObject ignores braces inside string values", () => {
    // The `}` inside the disposition string must NOT close the object early.
    const text = '{"name":"X","disposition":"loves } and { symbols"}';
    assert.equal(extractJsonObject(text), text);
});

test("extractJsonObject handles escaped quotes inside strings", () => {
    // The escaped quote must not end the string, so the real closing brace wins.
    const text = '{"name":"she said \\"hi\\"","role":"poet"}';
    assert.equal(extractJsonObject(text), text);
});

test("extractJsonObject handles an escaped backslash before a quote", () => {
    // "path":"C:\\" — the backslash is escaped, so the following quote DOES close
    // the string. A naive escape tracker would swallow the closing quote here.
    const text = '{"path":"C:\\\\","name":"X"}';
    assert.equal(extractJsonObject(text), text);
});

test("extractJsonObject returns the first object when several are present", () => {
    const text = '{"name":"A"} and then {"name":"B"}';
    assert.equal(extractJsonObject(text), '{"name":"A"}');
});

test("extractJsonObject returns null when there is no object", () => {
    assert.equal(extractJsonObject("no json here at all"), null);
    assert.equal(extractJsonObject(""), null);
});

test("extractJsonObject returns null for an unclosed object (truncated reply)", () => {
    assert.equal(extractJsonObject('{"name":"Mara", "role":"engin'), null);
});

test("extractJsonObject returns the literal substring even if it is invalid JSON", () => {
    // Balanced braces but not valid JSON: extraction is structural, validation
    // happens at the parse step (parsePersonaReply), which is where it should.
    assert.equal(extractJsonObject("{nope}"), "{nope}");
});

// ── parsePersonality: validate a parsed value into a Personality ──────────────

test("parsePersonality accepts a full persona and trims its fields", () => {
    const persona = parsePersonality({
        name: "  Mara  ",
        role: "staff security engineer",
        disposition: "assumes hostile input",
        standards: "rejects unmitigated attack surface",
        expertise: "authn",
    });
    assert.equal(persona.name, "Mara");
    assert.equal(persona.role, "staff security engineer");
    assert.equal(persona.expertise, "authn");
});

test("parsePersonality accepts a name-only persona", () => {
    const persona = parsePersonality({ name: "Dana" });
    assert.deepEqual(persona, { name: "Dana" });
});

test("parsePersonality drops unknown fields rather than rejecting", () => {
    const persona = parsePersonality({ name: "Dana", vibe: "spooky", age: 40 });
    assert.deepEqual(persona, { name: "Dana" });
});

test("parsePersonality omits optional fields that are empty or whitespace", () => {
    const persona = parsePersonality({ name: "Dana", role: "   ", disposition: "" });
    assert.deepEqual(persona, { name: "Dana" });
});

test("parsePersonality ignores null/undefined optional fields", () => {
    const persona = parsePersonality({ name: "Dana", role: null, expertise: undefined });
    assert.deepEqual(persona, { name: "Dana" });
});

test("parsePersonality never accepts model-supplied stakes", () => {
    // Stakes are dealt, not self-assigned. A model trying to hand itself a stake
    // is silently dropped (it's an unknown field to the validator).
    const persona = parsePersonality({
        name: "Dana",
        stakes: [{ riding: "my reputation", valence: "falsePass" }],
    });
    assert.deepEqual(persona, { name: "Dana" });
    assert.equal(persona.stakes, undefined);
});

test("parsePersonality rejects a non-object", () => {
    assert.throws(() => parsePersonality(42), PersonaError);
    assert.throws(() => parsePersonality("Dana"), PersonaError);
    assert.throws(() => parsePersonality(null), PersonaError);
    assert.throws(() => parsePersonality([{ name: "Dana" }]), PersonaError);
});

test("parsePersonality rejects a missing or non-string name", () => {
    assert.throws(() => parsePersonality({}), PersonaError);
    assert.throws(() => parsePersonality({ name: 42 }), PersonaError);
    assert.throws(() => parsePersonality({ role: "engineer" }), PersonaError);
});

test("parsePersonality rejects an empty/whitespace name", () => {
    assert.throws(() => parsePersonality({ name: "" }), PersonaError);
    assert.throws(() => parsePersonality({ name: "   " }), PersonaError);
});

test("parsePersonality rejects a non-string optional field", () => {
    assert.throws(() => parsePersonality({ name: "Dana", role: 42 }), PersonaError);
    assert.throws(() => parsePersonality({ name: "Dana", disposition: ["a"] }), PersonaError);
});

test("a parsed persona renders cleanly through personaSystem", () => {
    // The whole point of dropping empty fields: no dangling labels downstream.
    const persona = parsePersonality({ name: "Dana", role: "  ", disposition: "blunt" });
    const system = personaSystem(persona);
    assert.match(system, /You are Dana\./);
    assert.match(system, /Disposition: blunt/);
    assert.doesNotMatch(system, /,\s*\./); // no "You are Dana, ." from an empty role
});

// ── parsePersonaReply: extract + parse + validate end to end ──────────────────

test("parsePersonaReply handles a fenced reply with prose", () => {
    const reply = 'Here you go:\n```json\n{"name":"Mara","role":"engineer"}\n```';
    const persona = parsePersonaReply(reply);
    assert.equal(persona.name, "Mara");
    assert.equal(persona.role, "engineer");
});

test("parsePersonaReply throws when there is no JSON object", () => {
    assert.throws(() => parsePersonaReply("I could not think of anyone."), {
        name: "PersonaError",
        message: /no JSON object/,
    });
});

test("parsePersonaReply throws on malformed JSON", () => {
    assert.throws(() => parsePersonaReply("{name: Mara, no quotes}"), {
        name: "PersonaError",
        message: /did not parse/,
    });
});

test("parsePersonaReply throws when valid JSON is not a valid persona", () => {
    assert.throws(() => parsePersonaReply('{"role":"engineer"}'), {
        name: "PersonaError",
        message: /name is required/,
    });
});

// ── generatePersona: drive one model turn and parse it ────────────────────────

test("generatePersona parses the model's reply into a Personality", async () => {
    const client = new FakeClient([
        textTurn('```json\n{"name":"Mara","disposition":"skeptical"}\n```'),
    ]);
    const persona = await generatePersona({ client });
    assert.equal(persona.name, "Mara");
    assert.equal(persona.disposition, "skeptical");
    assert.equal(persona.stakes, undefined); // no deal requested
});

test("generatePersona drives the persona system prompt and no context", async () => {
    const client = new FakeClient([textTurn('{"name":"Mara"}')]);
    await generatePersona({ client });

    // One generate call, carrying PERSONA_SYSTEM as the system turn.
    assert.equal(client.calls.length, 1);
    const messages = client.calls[0]!.messages;
    const system = messages.find((m) => m.sender.role === "system");
    assert.ok(system, "a system turn should be present");
    const systemText = system!.content.map((p) => (p.kind === "text" ? p.text : "")).join("");
    // Exact equality is the invariant: context: [] suppresses the temporal
    // provider, so nothing (no "current date" preamble) is appended to the
    // persona instruction. Any added context would lengthen this string.
    assert.equal(systemText, PERSONA_SYSTEM);
});

test("generatePersona folds the seed into the user prompt", async () => {
    const client = new FakeClient([textTurn('{"name":"Mara"}')]);
    await generatePersona({ client, seed: "theme: betrayal" });

    const messages = client.calls[0]!.messages;
    const user = messages.find((m) => m.sender.role === "user");
    const userText = user!.content.map((p) => (p.kind === "text" ? p.text : "")).join("");
    assert.match(userText, /theme: betrayal/);
});

test("generatePersona deals stakes when asked, deterministically", async () => {
    const client = new FakeClient([textTurn('{"name":"Mara"}')]);
    // A pinned random picks the first pool entry; count 1 by default.
    const persona = await generatePersona({
        client,
        deal: { count: 1, random: () => 0 },
    });
    assert.ok(persona.stakes, "stakes should be dealt");
    assert.equal(persona.stakes!.length, 1);
    // With random ()=>0 the partial Fisher–Yates leaves deck[0] in place.
    assert.equal(persona.stakes![0]!.riding, STAKE_POOL[0]!.riding);
});

test("generatePersona surfaces a PersonaError for an unparseable reply", async () => {
    const client = new FakeClient([textTurn("I'd rather not.")]);
    await assert.rejects(generatePersona({ client }), PersonaError);
});

test("generatePersona lets a transport error from the client propagate", async () => {
    // An empty script makes FakeClient throw on the (single) stream call: that's
    // a transport-shaped failure, distinct from a PersonaError, and must NOT be
    // swallowed as a parse problem.
    const client = new FakeClient([]);
    await assert.rejects(generatePersona({ client }), (err: Error) => {
        assert.ok(
            !(err instanceof PersonaError),
            "transport failure must not masquerade as a parse error",
        );
        return true;
    });
});
