/**
 * Tests for the slash-command registry ({@link src/commands.ts}).
 *
 * The registry is inert data plus a few pure helpers (signature rendering,
 * lookup, prefix filtering) that both the REPL's `/help` and the web client's `/`
 * menu read. These lock in the catalogue's shape and the helper behaviors every
 * surface depends on — bracket convention, alias resolution, prefix matching —
 * so a surface can render the menu without re-deriving any of it.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
    BUILTIN_COMMANDS,
    commandSignature,
    findCommand,
    matchCommands,
    type SlashCommand,
} from "../src/commands.ts";

test("the built-in catalogue is non-empty and well-formed", () => {
    assert.ok(BUILTIN_COMMANDS.length > 0, "expected built-in commands");
    for (const c of BUILTIN_COMMANDS) {
        assert.equal(typeof c.name, "string");
        assert.notEqual(c.name, "", "a command name must be non-empty");
        assert.ok(!c.name.startsWith("/"), "names carry no leading slash");
        assert.equal(typeof c.description, "string");
        assert.ok(Array.isArray(c.params), "params is an array");
    }
});

test("command names are unique", () => {
    const names = BUILTIN_COMMANDS.map((c) => c.name);
    assert.equal(new Set(names).size, names.length, "duplicate command name");
});

test("the catalogue includes the REPL's session-level commands", () => {
    const names = new Set(BUILTIN_COMMANDS.map((c) => c.name));
    for (const expected of ["reset", "history", "help", "exit"]) {
        assert.ok(names.has(expected), `missing /${expected}`);
    }
});

test("commandSignature renders a nullary command as just its name", () => {
    const cmd: SlashCommand = { name: "reset", description: "x", params: [] };
    assert.equal(commandSignature(cmd), "/reset");
});

test("commandSignature brackets required vs optional params", () => {
    const cmd: SlashCommand = {
        name: "model",
        description: "switch model",
        params: [
            { name: "name", description: "the model id", required: true },
            { name: "effort", description: "thinking effort", required: false },
        ],
    };
    assert.equal(commandSignature(cmd), "/model <name> [effort]");
});

test("findCommand resolves a primary name, case-insensitively", () => {
    assert.equal(findCommand("reset")?.name, "reset");
    assert.equal(findCommand("RESET")?.name, "reset");
    assert.equal(findCommand("  reset  ")?.name, "reset");
});

test("findCommand resolves an alias to its primary", () => {
    // /quit is an alias of /exit in the built-in set.
    assert.equal(findCommand("quit")?.name, "exit");
});

test("findCommand returns undefined for an unknown word or a bare slash", () => {
    assert.equal(findCommand("frobnicate"), undefined);
    assert.equal(findCommand(""), undefined);
});

test("matchCommands with an empty prefix returns the whole catalogue", () => {
    assert.deepEqual(
        matchCommands("").map((c) => c.name),
        BUILTIN_COMMANDS.map((c) => c.name),
    );
});

test("matchCommands narrows by name prefix", () => {
    const names = matchCommands("re").map((c) => c.name);
    assert.deepEqual(names, ["reset"]);
});

test("matchCommands matches an alias prefix too", () => {
    // `qu` matches no name but is a prefix of the `quit` alias of /exit.
    const names = matchCommands("qu").map((c) => c.name);
    assert.deepEqual(names, ["exit"]);
});

test("matchCommands preserves catalogue order", () => {
    const roster: SlashCommand[] = [
        { name: "alpha", description: "", params: [] },
        { name: "alps", description: "", params: [] },
        { name: "beta", description: "", params: [] },
    ];
    assert.deepEqual(
        matchCommands("al", roster).map((c) => c.name),
        ["alpha", "alps"],
    );
});
