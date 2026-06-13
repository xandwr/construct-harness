/**
 * Tests for passive context ({@link applyContext} and the temporal provider).
 *
 * Passive context is folded onto the outgoing messages *each turn* by the loop;
 * here we test the fold in isolation (the loop integration lives in
 * loop.test.ts), plus the temporal provider's shape and graceful degradation.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { applyContext, temporalContext } from "../src/context.ts";
import type { ContextProvider } from "../src/context.ts";
import { RoleType } from "../src/types.ts";
import type { Message } from "../src/types.ts";

const user = (text: string): Message => ({
    sender: { role: RoleType.User },
    timestamp: 0,
    content: [{ kind: "text", text }],
});

/** A provider that contributes a fixed system string. */
function sysProvider(name: string, text: string | undefined): ContextProvider {
    return { name, contribute: () => (text === undefined ? undefined : { system: text }) };
}

/** Pull the concatenated system text out of a message list. */
function systemText(messages: Message[]): string {
    return messages
        .filter((m) => m.sender.role === RoleType.System)
        .flatMap((m) => m.content)
        .filter((p): p is Extract<typeof p, { kind: "text" }> => p.kind === "text")
        .map((p) => p.text)
        .join("\n\n");
}

test("applyContext with no providers returns a copy, unchanged", () => {
    const messages = [user("hi")];
    const out = applyContext(messages, [], 0);
    assert.deepEqual(out, messages);
    assert.notEqual(out, messages, "must be a fresh array, not the caller's");
});

test("a system contribution is appended as a system turn", () => {
    const out = applyContext([user("hi")], [sysProvider("p", "ambient fact")], 0);
    assert.equal(out.length, 2);
    assert.equal(out[1]!.sender.role, RoleType.System);
    assert.equal(systemText(out), "ambient fact");
});

test("multiple system contributions are joined in provider order", () => {
    const out = applyContext(
        [user("hi")],
        [sysProvider("a", "first"), sysProvider("b", "second")],
        0,
    );
    assert.equal(systemText(out), "first\n\nsecond");
});

test("a provider returning undefined contributes nothing", () => {
    const out = applyContext([user("hi")], [sysProvider("silent", undefined)], 0);
    assert.equal(out.length, 1, "no system turn appended");
});

test("empty / whitespace-only system text is dropped, not injected blank", () => {
    const out = applyContext([user("hi")], [sysProvider("blank", "   ")], 0);
    assert.equal(out.length, 1);
});

test("message contributions are appended after the conversation", () => {
    const injector: ContextProvider = {
        name: "inject",
        contribute: () => ({ messages: [user("standing reminder")] }),
    };
    const out = applyContext([user("hi")], [injector], 0);
    assert.equal(out.length, 2);
    assert.equal(out[1]!.content[0]!.kind, "text");
    assert.deepEqual(systemText(out), "");
});

test("applyContext never mutates the caller's array", () => {
    const messages = [user("hi")];
    const before = messages.length;
    applyContext(messages, [sysProvider("p", "x")], 0);
    assert.equal(messages.length, before);
});

test("providers see the turn index and the conversation", () => {
    const seen: { turn: number; count: number }[] = [];
    const spy: ContextProvider = {
        name: "spy",
        contribute: (scope) => {
            seen.push({ turn: scope.turn, count: scope.messages.length });
            return undefined;
        },
    };
    applyContext([user("a"), user("b")], [spy], 3);
    assert.deepEqual(seen, [{ turn: 3, count: 2 }]);
});

// ── Temporal provider ─────────────────────────────────────────────────────────

test("temporalContext states the current date and time as a system fact", () => {
    const provider = temporalContext({ timeZone: "UTC" });
    assert.equal(provider.name, "temporal");
    const contribution = provider.contribute({ messages: [], turn: 0 });
    assert.ok(contribution?.system);
    assert.match(contribution.system, /current date and time/i);
    assert.match(contribution.system, /UTC/);
});

test("temporalContext recomputes the time on each call", async () => {
    const provider = temporalContext({ timeZone: "UTC", locale: "en-US" });
    const first = provider.contribute({ messages: [], turn: 0 })!.system!;
    // Wait past a one-second boundary so the formatted "long" time string moves.
    await new Promise((r) => setTimeout(r, 1100));
    const second = provider.contribute({ messages: [], turn: 1 })!.system!;
    assert.notEqual(first, second, "the rendered time should advance between turns");
});

test("an invalid timezone degrades to the host default instead of throwing", () => {
    // Must not throw at construction…
    const provider = temporalContext({ timeZone: "Not/AZone" });
    // …and must still produce a usable temporal fact.
    const contribution = provider.contribute({ messages: [], turn: 0 });
    assert.match(contribution!.system!, /current date and time/i);
});
