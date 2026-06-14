/**
 * Client-side helpers for the chat composer's slash-command menu.
 *
 * The catalogue itself comes from the server ({@link getCommands}, mirroring the
 * harness `BUILTIN_COMMANDS`); this module is the pure presentation logic the
 * composer needs around it: deciding when a draft is "opening a command", which
 * commands match what's been typed, and how to render one command's call
 * signature. It mirrors `src/commands.ts`'s `commandSignature` / `matchCommands`
 * so the menu reads the same as the REPL's `/help`, but stays a separate copy
 * because the client can't import server-side modules.
 */

import type { WireCommand } from "./api";

/** The leading `/word` of a draft when it's opening a slash command, or null when
 *  it isn't one. A command draft is a single token starting with `/` and no
 *  whitespace yet — so `/re` opens the menu but `/reset now` (a space typed) has
 *  moved past selection and the menu closes. Returns the typed prefix without the
 *  slash (`re`), which {@link filterCommands} matches on; `/` alone yields `''`,
 *  the show-everything case. */
export function commandPrefix(draft: string): string | null {
    if (!draft.startsWith("/")) return null;
    const rest = draft.slice(1);
    // Any whitespace means the human has committed to a command and is now typing
    // its argument (or prose) — stop offering the menu.
    if (/\s/.test(rest)) return null;
    return rest;
}

/** Filter the catalogue by a typed prefix (the text after `/`), matching a
 *  command's name or any alias case-insensitively. An empty prefix returns the
 *  whole list (the bare `/` case). Order is preserved so the menu stays stable as
 *  it narrows. */
export function filterCommands(prefix: string, commands: WireCommand[]): WireCommand[] {
    const p = prefix.trim().toLowerCase();
    if (p === "") return commands;
    return commands.filter(
        (c) => c.name.startsWith(p) || (c.aliases ?? []).some((a) => a.toLowerCase().startsWith(p)),
    );
}

/** Render a command's call signature for a menu row, e.g. `/reset` or
 *  `/model <name>`. Required params show as `<name>`, optional ones as `[name]`,
 *  matching the harness `commandSignature` so the UI and the REPL agree. */
export function commandSignature(cmd: WireCommand): string {
    const parts = [`/${cmd.name}`];
    for (const p of cmd.params) parts.push(p.required ? `<${p.name}>` : `[${p.name}]`);
    return parts.join(" ");
}
