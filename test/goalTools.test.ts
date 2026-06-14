/**
 * Tests for the goal tool bridge ({@link goalTools}, {@link goalContext}).
 *
 * Exercises the tools two ways — directly (calling each `ToolDef.run`, the way
 * the loop would) and through {@link runLoop} driven by the scripted
 * {@link FakeClient} — plus the passive provider that injects active goals into
 * the system prompt each turn.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { GoalStore } from "../src/goals.ts";
import { goalTools, goalContext, DEFAULT_GOAL_LIMIT } from "../src/goalTools.ts";
import { applyContext } from "../src/context.ts";
import { runLoop } from "../src/bridge/loop.ts";
import { RoleType } from "../src/types.ts";
import type { Message, ToolDef, ToolResultPart } from "../src/types.ts";
import { FakeClient, callTurn, textTurn } from "./helpers/fakeClient.ts";

function freshStore(): GoalStore {
    return new GoalStore(":memory:");
}

function tool(tools: ToolDef[], name: string): ToolDef {
    const t = tools.find((x) => x.name === name);
    assert.ok(t, `expected a tool named ${name}`);
    return t;
}

const user = (text: string): Message => ({
    sender: { role: RoleType.User },
    timestamp: 0,
    content: [{ kind: "text", text }],
});

function systemText(messages: Message[]): string {
    return messages
        .filter((m) => m.sender.role === RoleType.System)
        .flatMap((m) => m.content)
        .filter((p): p is Extract<typeof p, { kind: "text" }> => p.kind === "text")
        .map((p) => p.text)
        .join("\n\n");
}

// ---------------------------------------------------------------------------
// Tool shape
// ---------------------------------------------------------------------------

test("goalTools exposes set, update, and list", () => {
    const store = freshStore();
    assert.deepEqual(
        goalTools(store).map((t) => t.name),
        ["goal_set", "goal_update", "goal_list"],
    );
    store.close();
});

// ---------------------------------------------------------------------------
// goal_set
// ---------------------------------------------------------------------------

test("goal_set creates an active goal scoped to the session", () => {
    const store = freshStore();
    const set = tool(goalTools(store, "s_me"), "goal_set");
    return set.run({ content: "land the feature" }).then((res) => {
        const r = res as { set: boolean; goal: { content: string; status: string } };
        assert.equal(r.set, true);
        assert.equal(r.goal.status, "active");
        // Scoped: a different session sees nothing.
        assert.equal(store.list({ session: "s_other" }).length, 0);
        assert.equal(store.list({ session: "s_me" }).length, 1);
        store.close();
    });
});

test("goal_set surfaces a validation error as a readable result", async () => {
    const store = freshStore();
    const set = tool(goalTools(store), "goal_set");
    const res = (await set.run({ content: "   " })) as { set: boolean; error?: string };
    assert.equal(res.set, false);
    assert.match(res.error ?? "", /empty/);
    store.close();
});

// ---------------------------------------------------------------------------
// goal_update
// ---------------------------------------------------------------------------

test("goal_update marks a goal done", async () => {
    const store = freshStore();
    const tools = goalTools(store, "s");
    const created = (await tool(tools, "goal_set").run({ content: "do it" })) as {
        goal: { id: number };
    };
    const res = (await tool(tools, "goal_update").run({
        id: created.goal.id,
        status: "done",
    })) as { updated: boolean; goal: { status: string } };
    assert.equal(res.updated, true);
    assert.equal(res.goal.status, "done");
    store.close();
});

test("goal_update can revise content and status together", async () => {
    const store = freshStore();
    const tools = goalTools(store, "s");
    const created = (await tool(tools, "goal_set").run({ content: "rough goal" })) as {
        goal: { id: number };
    };
    const res = (await tool(tools, "goal_update").run({
        id: created.goal.id,
        content: "sharpened goal",
        status: "done",
    })) as { updated: boolean; goal: { content: string; status: string } };
    assert.equal(res.goal.content, "sharpened goal");
    assert.equal(res.goal.status, "done");
    store.close();
});

test("goal_update reports a missing id and an empty patch", async () => {
    const store = freshStore();
    const update = tool(goalTools(store), "goal_update");
    const missing = (await update.run({ id: 99999, status: "done" })) as {
        updated: boolean;
        error?: string;
    };
    assert.equal(missing.updated, false);
    assert.match(missing.error ?? "", /no goal/);

    const empty = (await update.run({ id: 1 })) as { updated: boolean; error?: string };
    assert.equal(empty.updated, false);
    assert.match(empty.error ?? "", /status and\/or content/);
    store.close();
});

test("goal_update rejects a bogus status with a readable error", async () => {
    const store = freshStore();
    const tools = goalTools(store, "s");
    const created = (await tool(tools, "goal_set").run({ content: "x" })) as {
        goal: { id: number };
    };
    const res = (await tool(tools, "goal_update").run({
        id: created.goal.id,
        status: "paused",
    })) as { updated: boolean; error?: string };
    assert.equal(res.updated, false);
    assert.match(res.error ?? "", /active, done, abandoned/);
    store.close();
});

// ---------------------------------------------------------------------------
// goal_list
// ---------------------------------------------------------------------------

test("goal_list returns this session's goals, filterable by status", async () => {
    const store = freshStore();
    const tools = goalTools(store, "s");
    const a = (await tool(tools, "goal_set").run({ content: "alpha" })) as { goal: { id: number } };
    await tool(tools, "goal_set").run({ content: "beta" });
    await tool(tools, "goal_update").run({ id: a.goal.id, status: "done" });

    const all = (await tool(tools, "goal_list").run({})) as { count: number };
    assert.equal(all.count, 2);
    const active = (await tool(tools, "goal_list").run({ status: "active" })) as {
        count: number;
        goals: { content: string }[];
    };
    assert.equal(active.count, 1);
    assert.equal(active.goals[0].content, "beta");
    store.close();
});

// ---------------------------------------------------------------------------
// goalContext provider
// ---------------------------------------------------------------------------

test("goalContext injects active goals and stays silent when there are none", async () => {
    const store = freshStore();
    const provider = goalContext(store, "s");

    // No goals yet: contributes nothing, so no empty system turn.
    const empty = await applyContext([user("hi")], [provider], 0);
    assert.equal(systemText(empty), "");

    store.create({ content: "finish the migration", session: "s" });
    const withGoal = await applyContext([user("hi")], [provider], 0);
    assert.match(systemText(withGoal), /active goals/i);
    assert.match(systemText(withGoal), /finish the migration/);
    store.close();
});

test("goalContext shows only active goals, scoped to its session", async () => {
    const store = freshStore();
    store.create({ content: "mine active", session: "s" });
    const done = store.create({ content: "mine done", session: "s" });
    store.setStatus(done.id, "done");
    store.create({ content: "another session", session: "other" });

    const out = systemText(await applyContext([user("hi")], [goalContext(store, "s")], 0));
    assert.match(out, /mine active/);
    assert.ok(!/mine done/.test(out), "completed goals aren't injected");
    assert.ok(!/another session/.test(out), "other sessions' goals aren't injected");
    store.close();
});

test("goalContext injects shared (global) goals under their own heading", async () => {
    const store = freshStore();
    // A global goal (no session) is shared standing intent every conversation sees.
    store.create({ content: "uphold the house style" });
    store.create({ content: "ship the migration", session: "s" });

    const out = systemText(await applyContext([user("hi")], [goalContext(store, "s")], 0));
    // Both sections present, kept distinct.
    assert.match(out, /shared goals/i);
    assert.match(out, /uphold the house style/);
    assert.match(out, /this conversation's active goals/i);
    assert.match(out, /ship the migration/);
    // The shared section comes first (standing intent before the turn's work).
    assert.ok(out.indexOf("uphold the house style") < out.indexOf("ship the migration"));
    store.close();
});

test("goalContext shows shared goals to every session, not just one", async () => {
    const store = freshStore();
    store.create({ content: "global standing goal" });

    // A different session — with no goals of its own — still sees the shared one.
    const out = systemText(await applyContext([user("hi")], [goalContext(store, "other")], 0));
    assert.match(out, /shared goals/i);
    assert.match(out, /global standing goal/);
    assert.ok(
        !/this conversation's active goals/i.test(out),
        "a session with no goals of its own shows only the shared section",
    );
    store.close();
});

test("goalContext does not leak one session's goals into another via the global scope", async () => {
    const store = freshStore();
    store.create({ content: "session-A goal", session: "A" });
    // Session B reads: it must see neither A's goal (wrong session) nor treat it as
    // global (it has a session, so it's not session-less).
    const out = systemText(await applyContext([user("hi")], [goalContext(store, "B")], 0));
    assert.ok(!/session-A goal/.test(out), "session A's goal must not surface for session B");
    assert.equal(out, "", "with no shared and no B goals, nothing is injected");
    store.close();
});

test("goalContext caps the injected list at DEFAULT_GOAL_LIMIT", async () => {
    const store = freshStore();
    for (let i = 0; i < DEFAULT_GOAL_LIMIT + 5; i++) {
        store.create({ content: `goal ${i}`, session: "s" });
    }
    const out = systemText(await applyContext([user("hi")], [goalContext(store, "s")], 0));
    const lines = out.split("\n").filter((l) => l.startsWith("- "));
    assert.equal(lines.length, DEFAULT_GOAL_LIMIT);
    store.close();
});

// ---------------------------------------------------------------------------
// Dispatchable through the loop
// ---------------------------------------------------------------------------

test("goal_set is dispatchable through runLoop and persists", async () => {
    const store = freshStore();
    const tools = goalTools(store, "s");
    const client = new FakeClient([
        callTurn("c1", "goal_set", { content: "win the demo" }),
        textTurn("goal set"),
    ]);

    const result = await runLoop(client, { messages: [user("track this for me")], tools });

    const toolResults = result.messages
        .flatMap((m) => m.content)
        .filter((p): p is ToolResultPart => p.kind === "tool_result");
    assert.equal(toolResults.length, 1);
    assert.equal((toolResults[0].result as { set: boolean }).set, true);
    assert.equal(store.list({ session: "s" }).length, 1);
    store.close();
});
