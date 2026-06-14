/**
 * The slash-command registry: the single source of truth for which slash
 * commands a client surface offers, and what each one accepts.
 *
 * A slash command is a session-level action a transcript can't carry on its
 * own — clearing history, asking how long a conversation is, resuming. Both the
 * REPL ({@link runRepl}) and the web client present these, so the definitions
 * live here, provider-neutral and UI-agnostic, the way {@link AppDef} is the one
 * place the client lists its applets. Each surface renders the same list its own
 * way: the REPL prints them under `/help`, and the client pops a menu when the
 * composer opens with a `/`.
 *
 * This module only *describes* the commands; it does not run them. Execution is
 * the surface's job (the REPL switches on the name in {@link runRepl}; the client
 * dispatches its own), because what `/reset` means depends on what the surface
 * holds. Keeping the catalogue separate from the dispatch is what lets a new
 * surface advertise the exact same menu without copying a switch statement.
 */

/** One parameter a slash command accepts, for the menu's per-row hint. Commands
 *  today are mostly nullary; this is the shape a parameterized one (a future
 *  `/model <name>`) would fill in, and what the client renders beside the name so
 *  the human sees the call signature before typing it. */
export interface CommandParam {
    /** The parameter's placeholder name, shown in the signature (e.g. `name`). */
    name: string;
    /** A short, plain description of what the parameter selects. */
    description: string;
    /** Whether the command requires this argument. An optional one renders in
     *  brackets (`[name]`); a required one bare (`<name>`). */
    required: boolean;
}

/** A slash command as advertised to a client surface: enough to list it in a
 *  menu (name + summary + parameters) and to match what the human types. The
 *  definition is inert data; the surface owns running it. */
export interface SlashCommand {
    /** The command keyword, without the leading slash (e.g. `reset`). Unique
     *  within {@link BUILTIN_COMMANDS}; this is what the human types and what a
     *  surface switches on to dispatch. */
    name: string;
    /** A one-line, plain description of what invoking it does. Shown beside the
     *  name in a menu row. */
    description: string;
    /** The parameters it accepts, in signature order. Empty for a nullary
     *  command (the common case today). */
    params: CommandParam[];
    /** Other keywords that invoke the same command, for the human's muscle memory
     *  (`/quit` for `/exit`). Not shown as their own menu rows; listed on the
     *  primary so a surface can still dispatch them. */
    aliases?: string[];
}

/**
 * The built-in slash commands every client surface advertises.
 *
 * These mirror the session-level actions the REPL has always owned ({@link
 * runRepl}'s command handler) — clearing history, reporting its length, asking
 * for help, leaving — now named once so the web client can list the same set in
 * its `/` menu rather than reinventing it. Surfaces may ignore a command that
 * doesn't apply to them (a web composer has no process to `/exit`), but the menu
 * stays a faithful catalogue of what the harness understands.
 */
export const BUILTIN_COMMANDS: SlashCommand[] = [
    {
        name: "reset",
        description: "clear this conversation's history and start fresh",
        params: [],
    },
    {
        name: "history",
        description: "report how many messages are in the conversation",
        params: [],
    },
    {
        name: "help",
        description: "list the available slash commands",
        params: [],
    },
    {
        name: "exit",
        description: "leave the session (REPL only)",
        params: [],
        aliases: ["quit"],
    },
];

/**
 * Render a command's call signature for a one-line menu/help row, e.g.
 * `/model <name>` or `/reset`. Required params show as `<name>`, optional ones as
 * `[name]`, so the human reads the shape before typing it. This is the one place
 * the bracket convention lives, so the REPL's `/help` and the client's menu agree.
 */
export function commandSignature(cmd: SlashCommand): string {
    const parts = [`/${cmd.name}`];
    for (const p of cmd.params) parts.push(p.required ? `<${p.name}>` : `[${p.name}]`);
    return parts.join(" ");
}

/**
 * Resolve a typed command word (without the leading slash) to its definition,
 * matching the primary name or any alias, case-insensitively. Returns undefined
 * for an unknown word so a surface can report it rather than guess. The empty
 * string (a bare `/`) is treated as unknown: it's a menu trigger, not a command.
 */
export function findCommand(
    word: string,
    commands: SlashCommand[] = BUILTIN_COMMANDS,
): SlashCommand | undefined {
    const w = word.trim().toLowerCase();
    if (w === "") return undefined;
    return commands.find(
        (c) => c.name === w || (c.aliases ?? []).some((a) => a.toLowerCase() === w),
    );
}

/**
 * Filter the command list by a typed prefix (the text after `/`, before any
 * space), for the client's live menu. An empty prefix returns every command (the
 * bare `/` case: show the whole menu). Otherwise match the name or any alias by
 * prefix, case-insensitively, so typing `/re` narrows to `/reset`. Order is
 * preserved from {@link BUILTIN_COMMANDS} so the menu stays stable as it filters.
 */
export function matchCommands(
    prefix: string,
    commands: SlashCommand[] = BUILTIN_COMMANDS,
): SlashCommand[] {
    const p = prefix.trim().toLowerCase();
    if (p === "") return [...commands];
    return commands.filter(
        (c) => c.name.startsWith(p) || (c.aliases ?? []).some((a) => a.toLowerCase().startsWith(p)),
    );
}
