/**
 * The model catalogue: which providers exist and the model variants each one
 * serves, by current model id.
 *
 * This is the single source of truth the settings page's two dropdowns read
 * from — pick a provider, then pick one of its variants — and the same list the
 * runtime validates a model switch against, so the UI can only ever offer (and
 * the server only ever accept) a model the bridge actually knows how to drive.
 *
 * It lives in the bridge, not the client, because *which models exist* is
 * provider knowledge: the Anthropic id strings and their context windows come
 * from the Claude API, and a second provider would add its own block here rather
 * than teach the frontend a new vocabulary. The server exposes this verbatim
 * over /api/status so the page never hardcodes a model id.
 *
 * Ids are the exact current strings (no date suffixes); keep them in lockstep
 * with the SDK and the Claude API model list. `default: true` marks the variant
 * a fresh process boots on when MODEL is unset — exactly one per provider.
 */

/** A provider's stable id (also the value the provider dropdown carries). */
export type ProviderId = "anthropic";

/** One selectable model: its wire id, a human label for the dropdown, and the
 *  coarse facts worth showing next to it (context window, max output). */
export interface ModelVariant {
    /** The exact model id sent to the provider (e.g. "claude-opus-4-8"). */
    id: string;
    /** A short human label for the dropdown ("Opus 4.8"). */
    label: string;
    /** Context window in tokens, for a one-line capability hint. */
    contextWindow: number;
    /** Max output tokens, same. */
    maxOutput: number;
    /** True for the one variant a provider defaults to when none is configured. */
    default?: boolean;
}

/** A provider and the model variants it serves. */
export interface ProviderCatalogEntry {
    /** Stable provider id (matches {@link ProviderId}). */
    id: ProviderId;
    /** Human label for the provider dropdown. */
    label: string;
    /** The provider's model variants, newest/most-capable first. */
    models: ModelVariant[];
}

/**
 * Every provider the harness can drive and the models each serves. Today only
 * Anthropic — the structure is the point: adding a second provider is appending
 * an entry, and the settings dropdowns, the model validation, and the status
 * payload all pick it up without further change.
 *
 * The Anthropic list mirrors the current Claude API model catalogue. Fable 5 is
 * the most capable; Opus 4.8 is the default the bridge boots on (see
 * DEFAULT_MODEL in anthropic.ts). Sonnet and Haiku round out the speed/cost tiers.
 */
export const PROVIDERS: ProviderCatalogEntry[] = [
    {
        id: "anthropic",
        label: "Anthropic",
        models: [
            {
                id: "claude-fable-5",
                label: "Fable 5",
                contextWindow: 1_000_000,
                maxOutput: 128_000,
            },
            {
                id: "claude-opus-4-8",
                label: "Opus 4.8",
                contextWindow: 1_000_000,
                maxOutput: 128_000,
                default: true,
            },
            {
                id: "claude-opus-4-7",
                label: "Opus 4.7",
                contextWindow: 1_000_000,
                maxOutput: 128_000,
            },
            {
                id: "claude-opus-4-6",
                label: "Opus 4.6",
                contextWindow: 1_000_000,
                maxOutput: 128_000,
            },
            {
                id: "claude-sonnet-4-6",
                label: "Sonnet 4.6",
                contextWindow: 1_000_000,
                maxOutput: 64_000,
            },
            {
                id: "claude-haiku-4-5",
                label: "Haiku 4.5",
                contextWindow: 200_000,
                maxOutput: 64_000,
            },
        ],
    },
];

/** The provider that owns a given model id, or undefined when no provider
 *  serves it. Used to resolve which provider a configured/selected model
 *  belongs to (the provider dropdown's value follows the model). */
export function providerForModel(modelId: string): ProviderCatalogEntry | undefined {
    return PROVIDERS.find((p) => p.models.some((m) => m.id === modelId));
}

/** Whether any provider in the catalogue serves this model id. The runtime
 *  rejects a switch to an unknown id rather than handing the provider a string
 *  that would 404 on the next turn. */
export function isKnownModel(modelId: string): boolean {
    return providerForModel(modelId) !== undefined;
}
