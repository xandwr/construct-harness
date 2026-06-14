/**
 * The live runtime configuration: the knobs the settings page turns that the
 * harness actually reads on the next turn.
 *
 * Where {@link StatusConfig} froze the boot configuration for a read-only status
 * page, this is its mutable successor — one object the server holds and the
 * settings routes write through, so a model switch or a tool toggle in the UI
 * changes what a real conversation does. It owns three live concerns:
 *
 *  - **model** — delegated to the {@link ModelClient}'s own setter (the client
 *    reads its model id per request, so a change here lands on every
 *    conversation's next turn at once; there's one process-wide client).
 *  - **provider options** — the {@link ProviderOptions} object every Session was
 *    built with is held here and mutated *in place*, so live conversations pick
 *    up a server-tool or effort change because the Session reads
 *    `cfg.providerOptions` each turn off this same reference.
 *  - **local tool enablement** — which harness-owned tools (the shell, the KB
 *    note tools) a *newly built* Session wires in. A Session captures its tool
 *    list at construction, so toggling a local tool affects conversations
 *    started after the toggle, not ones already live — the honest limitation the
 *    status payload reports via `appliesToNewSessions`.
 *
 * Reads are cheap and side-effect free (the settings page can poll), writes are
 * validated (an unknown model or effort is rejected, not silently applied).
 */

import type { ModelClient, ProviderOptions } from "./bridge/types.ts";
import type { AnthropicOptions, ServerToolName } from "./bridge/anthropic.ts";
import { HarnessError } from "./bridge/errors.ts";

/** The effort levels the catalogue offers, in ascending depth/spend. Mirrors
 *  {@link AnthropicOptions.effort}; `undefined` (the provider default, "high")
 *  is offered as the explicit "default" choice in the UI rather than a level. */
export const EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"] as const;
export type EffortLevel = (typeof EFFORT_LEVELS)[number];

/** The provider-hosted server tools a caller may toggle, with a human label and
 *  a one-line note for the settings list. The order is the display order. */
export const SERVER_TOOL_CATALOG: { id: ServerToolName; label: string; note: string }[] = [
    { id: "web_search", label: "web search", note: "answer about the live world" },
    { id: "web_fetch", label: "web fetch", note: "read a named URL" },
    {
        id: "code_execution",
        label: "code execution",
        note: "run code in a provider sandbox",
    },
];

/** One harness-owned (local) tool the runtime can toggle: a stable key, the
 *  tool names it covers in the agent-facing toolset, and a human label. A single
 *  key can map to several tool names (a factory that returns more than one). */
export interface LocalToolGroup {
    /** Stable toggle key (the value the UI switch carries). */
    key: string;
    /** Human label for the settings list. */
    label: string;
    /** One-line note describing what enabling it gives the Construct. */
    note: string;
    /** The agent-facing tool names this group contributes when enabled. */
    toolNames: string[];
}

/** Whether a model id and an effort string are accepted, for write validation. */
const isEffort = (v: unknown): v is EffortLevel =>
    typeof v === "string" && (EFFORT_LEVELS as readonly string[]).includes(v);

/**
 * Holds and mutates the live runtime knobs. Constructed once in `buildDeps`
 * with the client it drives, the shared provider-options object every Session
 * references, and the catalogue of local tool groups the wiring supports.
 */
export class RuntimeConfig {
    /** The process-wide model client; model changes go through its setter. */
    private readonly client: ModelClient;
    /** The single ProviderOptions object every Session was built with. Mutated in
     *  place so live conversations see server-tool/effort changes. */
    private readonly providerOptions: ProviderOptions & AnthropicOptions;
    /** Every local tool group the server knows how to wire, for the UI list and
     *  to bound which keys are valid to toggle. */
    readonly localGroups: LocalToolGroup[];
    /** The set of local tool group keys currently enabled. New Sessions wire in
     *  exactly the tools whose group is in this set. */
    private readonly enabledLocal: Set<string>;

    constructor(
        client: ModelClient,
        providerOptions: ProviderOptions & AnthropicOptions,
        localGroups: LocalToolGroup[],
        /** Which groups start enabled (the boot wiring). */
        enabledLocalKeys: string[],
    ) {
        this.client = client;
        this.providerOptions = providerOptions;
        this.localGroups = localGroups;
        this.enabledLocal = new Set(enabledLocalKeys);
    }

    /** The model id the next turn will use. */
    get model(): string {
        return this.client.model;
    }

    /**
     * Switch the model. Delegates to the client's validated {@link ModelClient.setModel}
     * (an unknown id throws an `invalid_request` {@link HarnessError} there, leaving
     * the live model untouched). A client that doesn't support switching is
     * rejected the same way rather than silently no-op'ing. Takes effect on every
     * conversation's next turn.
     */
    setModel(id: string): void {
        if (!this.client.setModel) {
            throw new HarnessError("this provider does not support switching models", {
                kind: "invalid_request",
                retryable: false,
            });
        }
        this.client.setModel(id);
    }

    /** The server tools currently enabled (a copy, newest-first display order
     *  preserved by the catalogue ordering at write time). */
    get serverTools(): ServerToolName[] {
        return [...(this.providerOptions.serverTools ?? [])];
    }

    /**
     * Replace the enabled server-tool set, in place on the shared options object
     * so live conversations pick it up next turn. Validates every name against
     * the catalogue and de-duplicates, keeping catalogue (display) order so the
     * stored set reads the same as the UI. An unknown name is rejected rather
     * than silently dropped.
     */
    setServerTools(names: string[]): void {
        const known = new Set(SERVER_TOOL_CATALOG.map((t) => t.id));
        for (const n of names) {
            if (!known.has(n as ServerToolName)) {
                throw new HarnessError(`unknown server tool "${n}"`, {
                    kind: "invalid_request",
                    retryable: false,
                });
            }
        }
        const want = new Set(names);
        this.providerOptions.serverTools = SERVER_TOOL_CATALOG.filter((t) => want.has(t.id)).map(
            (t) => t.id,
        );
    }

    /** The effort level, or undefined for the provider default ("high"). */
    get effort(): EffortLevel | undefined {
        return this.providerOptions.effort;
    }

    /**
     * Set the reasoning-effort level, or clear it (null/undefined) to fall back
     * to the provider default. Mutated in place so live conversations see it.
     * An unrecognized level is rejected.
     */
    setEffort(level: string | null | undefined): void {
        if (level === null || level === undefined || level === "") {
            delete this.providerOptions.effort;
            return;
        }
        if (!isEffort(level)) {
            throw new HarnessError(`unknown effort level "${level}"`, {
                kind: "invalid_request",
                retryable: false,
            });
        }
        this.providerOptions.effort = level;
    }

    /** Whether a local tool group is currently enabled for new Sessions. */
    isLocalEnabled(key: string): boolean {
        return this.enabledLocal.has(key);
    }

    /** The enabled local group keys, in catalogue order. */
    get enabledLocalKeys(): string[] {
        return this.localGroups.filter((g) => this.enabledLocal.has(g.key)).map((g) => g.key);
    }

    /**
     * Enable or disable a local tool group for *newly built* Sessions. Validated
     * against {@link localGroups} so an unknown key is rejected. Returns nothing;
     * the change is visible to the next {@link SessionPool.resolve} that builds a
     * Session, not to conversations already live (they captured their tools at
     * construction).
     */
    setLocalEnabled(key: string, enabled: boolean): void {
        if (!this.localGroups.some((g) => g.key === key)) {
            throw new HarnessError(`unknown tool "${key}"`, {
                kind: "invalid_request",
                retryable: false,
            });
        }
        if (enabled) this.enabledLocal.add(key);
        else this.enabledLocal.delete(key);
    }
}
