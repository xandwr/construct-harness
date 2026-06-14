/**
 * Tests for passive context ({@link applyContext} and the temporal provider).
 *
 * Passive context is folded onto the outgoing messages *each turn* by the loop;
 * here we test the fold in isolation (the loop integration lives in
 * loop.test.ts), plus the temporal provider's shape and graceful degradation.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { applyContext, temporalContext, humanizeDuration } from "../src/context.ts";
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

test("applyContext with no providers returns a copy, unchanged", async () => {
    const messages = [user("hi")];
    const out = await applyContext(messages, [], 0);
    assert.deepEqual(out, messages);
    assert.notEqual(out, messages, "must be a fresh array, not the caller's");
});

test("a system contribution is appended as a system turn", async () => {
    const out = await applyContext([user("hi")], [sysProvider("p", "ambient fact")], 0);
    assert.equal(out.length, 2);
    assert.equal(out[1]!.sender.role, RoleType.System);
    assert.equal(systemText(out), "ambient fact");
});

test("multiple system contributions are joined in provider order", async () => {
    const out = await applyContext(
        [user("hi")],
        [sysProvider("a", "first"), sysProvider("b", "second")],
        0,
    );
    assert.equal(systemText(out), "first\n\nsecond");
});

test("an async provider's contribution is awaited and folded in", async () => {
    const slow: ContextProvider = {
        name: "slow",
        contribute: async () => {
            await new Promise((r) => setTimeout(r, 5));
            return { system: "from a store" };
        },
    };
    const out = await applyContext([user("hi")], [slow], 0);
    assert.equal(systemText(out), "from a store");
});

test("async contributions fold in provider order regardless of which resolves first", async () => {
    // The first provider is slow, the second fast: completion order is reversed,
    // but the join must follow provider order so the cached prefix stays stable.
    const slow: ContextProvider = {
        name: "slow",
        contribute: async () => {
            await new Promise((r) => setTimeout(r, 10));
            return { system: "first" };
        },
    };
    const fast: ContextProvider = { name: "fast", contribute: () => ({ system: "second" }) };
    const out = await applyContext([user("hi")], [slow, fast], 0);
    assert.equal(systemText(out), "first\n\nsecond");
});

test("a throwing provider is dropped, not allowed to fail the turn", async () => {
    const boom: ContextProvider = {
        name: "boom",
        contribute: () => {
            throw new Error("provider blew up");
        },
    };
    const ok: ContextProvider = { name: "ok", contribute: () => ({ system: "survivor" }) };
    const out = await applyContext([user("hi")], [boom, ok], 0);
    assert.equal(systemText(out), "survivor");
});

test("a provider returning undefined contributes nothing", async () => {
    const out = await applyContext([user("hi")], [sysProvider("silent", undefined)], 0);
    assert.equal(out.length, 1, "no system turn appended");
});

test("empty / whitespace-only system text is dropped, not injected blank", async () => {
    const out = await applyContext([user("hi")], [sysProvider("blank", "   ")], 0);
    assert.equal(out.length, 1);
});

test("message contributions are appended after the conversation", async () => {
    const injector: ContextProvider = {
        name: "inject",
        contribute: () => ({ messages: [user("standing reminder")] }),
    };
    const out = await applyContext([user("hi")], [injector], 0);
    assert.equal(out.length, 2);
    assert.equal(out[1]!.content[0]!.kind, "text");
    assert.deepEqual(systemText(out), "");
});

test("applyContext never mutates the caller's array", async () => {
    const messages = [user("hi")];
    const before = messages.length;
    await applyContext(messages, [sysProvider("p", "x")], 0);
    assert.equal(messages.length, before);
});

test("providers see the turn index, the conversation, and sessionStart", async () => {
    const seen: { turn: number; count: number; start?: number }[] = [];
    const spy: ContextProvider = {
        name: "spy",
        contribute: (scope) => {
            seen.push({
                turn: scope.turn,
                count: scope.messages.length,
                start: scope.sessionStart,
            });
            return undefined;
        },
    };
    await applyContext([user("a"), user("b")], [spy], 3, 12345);
    assert.deepEqual(seen, [{ turn: 3, count: 2, start: 12345 }]);
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

// ── Temporal: elapsed + session duration ────────────────────────────────────

test("temporalContext reports time since the previous message", () => {
    const provider = temporalContext({ timeZone: "UTC" });
    // A prior message stamped two hours ago: the provider should phrase the gap.
    const twoHoursAgo = Date.now() - 2 * 60 * 60 * 1000;
    const msg: Message = {
        sender: { role: RoleType.User },
        timestamp: twoHoursAgo,
        content: [{ kind: "text", text: "earlier" }],
    };
    const out = provider.contribute({ messages: [msg], turn: 1 })!.system!;
    assert.match(out, /previous message was .* ago/i);
    assert.match(out, /hours? ago/i);
});

test("temporalContext reports session duration when sessionStart is given", () => {
    const provider = temporalContext({ timeZone: "UTC" });
    const threeDaysAgo = Date.now() - 3 * 24 * 60 * 60 * 1000;
    const out = provider.contribute({
        messages: [],
        turn: 5,
        sessionStart: threeDaysAgo,
    })!.system!;
    assert.match(out, /running for .* days/i);
});

test("temporalContext omits elapsed lines on the opening turn", () => {
    const provider = temporalContext({ timeZone: "UTC" });
    // No prior message, no sessionStart: just the absolute time, nothing relative.
    const out = provider.contribute({ messages: [], turn: 0 })!.system!;
    assert.match(out, /current date and time/i);
    assert.ok(!/ago/.test(out), "no 'previous message' line without a prior message");
    assert.ok(!/running for/.test(out), "no session-duration line without sessionStart");
});

test("temporalContext elapsed:false yields just the absolute time", () => {
    const provider = temporalContext({ timeZone: "UTC", elapsed: false });
    const old = Date.now() - 5 * 60 * 60 * 1000;
    const out = provider.contribute({ messages: [], turn: 2, sessionStart: old })!.system!;
    assert.match(out, /current date and time/i);
    assert.ok(!/running for/.test(out), "session duration suppressed when elapsed is off");
});

test("humanizeDuration phrases spans at a human scale", () => {
    assert.equal(humanizeDuration(0), "just now");
    assert.equal(humanizeDuration(10_000), "just now");
    assert.equal(humanizeDuration(60_000), "a minute");
    assert.equal(humanizeDuration(90_000), "2 minutes"); // 1.5 min rounds to 2
    assert.equal(humanizeDuration(5 * 60_000), "5 minutes");
    assert.equal(humanizeDuration(60 * 60_000), "an hour");
    assert.equal(humanizeDuration(3 * 60 * 60_000), "3 hours");
    assert.equal(humanizeDuration(24 * 60 * 60_000), "a day");
    assert.equal(humanizeDuration(3 * 24 * 60 * 60_000), "3 days");
    assert.equal(humanizeDuration(10 * 24 * 60 * 60_000), "a week");
    assert.equal(humanizeDuration(45 * 24 * 60 * 60_000), "2 months");
    assert.equal(humanizeDuration(400 * 24 * 60 * 60_000), "a year");
    // Negative span (clock skew) reads as just now, never a negative phrase.
    assert.equal(humanizeDuration(-5000), "just now");
});
