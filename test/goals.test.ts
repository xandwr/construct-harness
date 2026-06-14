/**
 * Tests for the goal store ({@link GoalStore}).
 *
 * Like the memory and event suites, every store is an in-memory SQLite database
 * so the suite never touches disk, and `now` is injected wherever ordering or
 * timestamps matter. A couple of tests need a real file (shared migration); those
 * use a temp directory.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { GoalStore, GoalError, isGoalStatus } from "../src/goals.ts";
import { MemoryStore, SCHEMA_VERSION, MAX_CONTENT_LENGTH } from "../src/memory.ts";

function withTempDir(fn: (dir: string) => void): void {
    const dir = mkdtempSync(join(tmpdir(), "goalstore-"));
    try {
        fn(dir);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}

function freshStore(): GoalStore {
    return new GoalStore(":memory:");
}

// ---------------------------------------------------------------------------
// Migration & shared schema
// ---------------------------------------------------------------------------

test("a fresh GoalStore is migrated to SCHEMA_VERSION", () => {
    const store = freshStore();
    assert.equal(store.version, SCHEMA_VERSION);
    store.close();
});

test("GoalStore and MemoryStore share one schema on the same file", () => {
    withTempDir((dir) => {
        const path = join(dir, "shared.sqlite");
        // Opening either store brings the whole schema current under one
        // user_version; opening both must not re-run or clash.
        const goals = new GoalStore(path);
        const memory = new MemoryStore(path);
        assert.equal(goals.version, SCHEMA_VERSION);
        assert.equal(memory.version, SCHEMA_VERSION);
        goals.close();
        memory.close();
    });
});

// ---------------------------------------------------------------------------
// Create & read
// ---------------------------------------------------------------------------

test("create assigns an id, defaults to active, and stamps timestamps", () => {
    const store = freshStore();
    const g = store.create({ content: "ship the release", now: 1000 });
    assert.ok(g.id > 0);
    assert.equal(g.content, "ship the release");
    assert.equal(g.status, "active");
    assert.equal(g.created, 1000);
    assert.equal(g.updated, 1000);
    store.close();
});

test("create trims content and rejects empty or oversize", () => {
    const store = freshStore();
    assert.equal(store.create({ content: "  spaced  " }).content, "spaced");
    assert.throws(() => store.create({ content: "   " }), GoalError);
    assert.throws(() => store.create({ content: "x".repeat(MAX_CONTENT_LENGTH + 1) }), GoalError);
    store.close();
});

test("get returns a stored goal and undefined for a missing id", () => {
    const store = freshStore();
    const g = store.create({ content: "a goal" });
    assert.equal(store.get(g.id)?.content, "a goal");
    assert.equal(store.get(99999), undefined);
    store.close();
});

test("list returns goals oldest first (to-do reading order)", () => {
    const store = freshStore();
    store.create({ content: "first", now: 1 });
    store.create({ content: "second", now: 2 });
    store.create({ content: "third", now: 3 });
    assert.deepEqual(
        store.list().map((g) => g.content),
        ["first", "second", "third"],
    );
    store.close();
});

// ---------------------------------------------------------------------------
// Status filter & session scope
// ---------------------------------------------------------------------------

test("list filters by status", () => {
    const store = freshStore();
    const a = store.create({ content: "active one" });
    const b = store.create({ content: "to finish" });
    store.setStatus(b.id, "done");
    assert.deepEqual(
        store.list({ status: "active" }).map((g) => g.content),
        ["active one"],
    );
    assert.deepEqual(
        store.list({ status: "done" }).map((g) => g.content),
        ["to finish"],
    );
    void a;
    store.close();
});

test("list filters by session, and a no-session read sees everything", () => {
    const store = freshStore();
    store.create({ content: "mine", session: "s_me" });
    store.create({ content: "theirs", session: "s_other" });
    store.create({ content: "global" });
    assert.deepEqual(
        store.list({ session: "s_me" }).map((g) => g.content),
        ["mine"],
    );
    assert.equal(store.list().length, 3);
    store.close();
});

test("scope='global' reads only goals with no session (session IS NULL)", () => {
    const store = freshStore();
    store.create({ content: "shared-a" });
    store.create({ content: "scoped", session: "s_x" });
    store.create({ content: "shared-b" });
    // The distinction a bare session filter can't draw: a no-session read sees
    // everything, but scope='global' sees only the session-less rows.
    assert.deepEqual(
        store
            .list({ scope: "global" })
            .map((g) => g.content)
            .sort(),
        ["shared-a", "shared-b"],
    );
    // A bare list (no scope, no session) still sees all three.
    assert.equal(store.list().length, 3);
    store.close();
});

test("scope='global' composes with status", () => {
    const store = freshStore();
    const a = store.create({ content: "shared-active" });
    const b = store.create({ content: "shared-done" });
    store.setStatus(b.id, "done");
    store.create({ content: "scoped-active", session: "s_x" });
    assert.deepEqual(
        store.list({ scope: "global", status: "active" }).map((g) => g.content),
        ["shared-active"],
    );
    void a;
    store.close();
});

test("scope='session' reads only that session, never the global rows", () => {
    const store = freshStore();
    store.create({ content: "global" });
    store.create({ content: "mine", session: "s_me" });
    store.create({ content: "theirs", session: "s_other" });
    assert.deepEqual(
        store.list({ scope: "session", session: "s_me" }).map((g) => g.content),
        ["mine"],
    );
    store.close();
});

test("count honors scope='global'", () => {
    const store = freshStore();
    store.create({ content: "shared-a" });
    store.create({ content: "shared-b" });
    store.create({ content: "scoped", session: "s_x" });
    assert.equal(store.count(), 3);
    assert.equal(store.count({ scope: "global" }), 2);
    assert.equal(store.count({ scope: "session", session: "s_x" }), 1);
    store.close();
});

// ---------------------------------------------------------------------------
// Update: status & content
// ---------------------------------------------------------------------------

test("setStatus moves a goal and bumps updated; unknown id returns undefined", () => {
    const store = freshStore();
    const g = store.create({ content: "do it", now: 1 });
    const done = store.setStatus(g.id, "done", 2);
    assert.equal(done?.status, "done");
    assert.equal(done?.updated, 2);
    assert.equal(done?.created, 1, "created is immutable");
    assert.equal(store.setStatus(99999, "done"), undefined);
    store.close();
});

test("the schema CHECK rejects a bogus status even past the type guard", () => {
    const store = freshStore();
    const g = store.create({ content: "x" });
    // isGoalStatus is the front line; force a bad value past it to prove the DB
    // constraint is the backstop.
    assert.throws(() => store.setStatus(g.id, "bogus" as never), GoalError);
    store.close();
});

test("edit revises text, bumps updated, and validates", () => {
    const store = freshStore();
    const g = store.create({ content: "rough", now: 1 });
    const edited = store.edit(g.id, "sharpened", 5);
    assert.equal(edited?.content, "sharpened");
    assert.equal(edited?.updated, 5);
    assert.throws(() => store.edit(g.id, "   "), GoalError);
    assert.equal(store.edit(99999, "whatever"), undefined);
    store.close();
});

// ---------------------------------------------------------------------------
// Delete & count
// ---------------------------------------------------------------------------

test("delete removes a goal and reports whether a row went", () => {
    const store = freshStore();
    const g = store.create({ content: "oops" });
    assert.equal(store.delete(g.id), true);
    assert.equal(store.get(g.id), undefined);
    assert.equal(store.delete(g.id), false);
    store.close();
});

test("count honors status and session filters", () => {
    const store = freshStore();
    store.create({ content: "a", session: "s1" });
    const b = store.create({ content: "b", session: "s1" });
    store.create({ content: "c", session: "s2" });
    store.setStatus(b.id, "done");
    assert.equal(store.count(), 3);
    assert.equal(store.count({ session: "s1" }), 2);
    assert.equal(store.count({ status: "active" }), 2);
    assert.equal(store.count({ status: "active", session: "s1" }), 1);
    store.close();
});

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

test("isGoalStatus recognizes only the three lifecycle states", () => {
    assert.ok(isGoalStatus("active"));
    assert.ok(isGoalStatus("done"));
    assert.ok(isGoalStatus("abandoned"));
    assert.ok(!isGoalStatus("paused"));
    assert.ok(!isGoalStatus(3));
    assert.ok(!isGoalStatus(undefined));
});

test("a closed store rejects further use", () => {
    const store = freshStore();
    store.close();
    assert.throws(() => store.create({ content: "x" }), GoalError);
});
