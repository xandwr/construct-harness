/**
 * The provider-agnostic bridge contract.
 *
 * This module is the boundary between the harness's static core (`../types.ts`)
 * and any concrete model API. It imports *only* from the core: never from a
 * provider SDK: so that the core stays the single source of truth for what a
 * message is, and providers remain interchangeable implementations beneath it.
 *
 * A provider supplies one thing: a `ModelClient`. Anthropic is the first.
 */

import type { Message, ToolDef } from "../types.ts";

/**
 * What a provider can do beyond the common request/response path.
 *
 * The harness branches on these flags rather than on a concrete client class,
 * so adding a second provider never means an `instanceof AnthropicClient`
 * check leaking into harness code. A `false` flag means "this provider ignores
 * the corresponding {@link GenerateParams.providerOptions} for that feature."
 */
export interface ProviderCapabilities {
    /** Provider can produce/return a reasoning trace (e.g. Anthropic thinking). */
    readonly thinking: boolean;
    /** Provider supports an effort/depth knob distinct from token limits. */
    readonly effort: boolean;
    /** Provider can cache a stable prompt prefix across requests. */
    readonly promptCaching: boolean;
    /** Provider hosts tools server-side (web search, code exec, etc.). */
    readonly serverTools: boolean;
    /** Provider can stream incremental output (see {@link ModelClient.stream}). */
    readonly streaming: boolean;
}

/**
 * Provider-specific request knobs that the core has no vocabulary for.
 *
 * This is deliberately opaque at the bridge level: each provider narrows it to
 * its own typed options (see Anthropic's `AnthropicOptions`). Keeping it out of
 * {@link GenerateParams}'s typed fields is what stops one provider's vocabulary
 * from accreting into the core. The harness only sets it when it has already
 * checked the matching {@link ProviderCapabilities} flag.
 */
export type ProviderOptions = Record<string, unknown>;

/** A provider-neutral generation request, expressed entirely in core types. */
export interface GenerateParams {
    /** Conversation so far, oldest first. System guidance is carried as a
     *  `Sender` with `role: "system"`; mappers lift it where the provider wants
     *  it (Anthropic puts it top-level, not in the message array). */
    messages: Message[];
    /** Tools the model may call. The harness owns `run`; the mapper only ships
     *  `name`/`description`/`parameters` to the provider. */
    tools?: ToolDef[];
    /** Upper bound on generated tokens. Mappers supply a provider-appropriate
     *  default when omitted. */
    maxTokens?: number;
    /** Opaque, per-provider extras gated by {@link ProviderCapabilities}. */
    providerOptions?: ProviderOptions;
}

/** Why generation stopped, normalized across providers. `"other"` carries the
 *  raw provider reason in {@link GenerateResult.raw} for inspection. */
export type StopReason = "end_turn" | "tool_use" | "max_tokens" | "refusal" | "other";

/** Token accounting, normalized. Providers that don't report a field omit it. */
export interface Usage {
    inputTokens?: number;
    outputTokens?: number;
    /** Tokens served from a cached prefix, when the provider reports caching. */
    cacheReadTokens?: number;
}

/** A completed generation, mapped back into core types. */
export interface GenerateResult {
    /** The assistant turn, as a core `Message`. Its `content` may include
     *  `tool_call` parts when {@link stopReason} is `"tool_use"`. */
    message: Message;
    stopReason: StopReason;
    usage: Usage;
    /** The provider's model id that actually served the request. */
    model: string;
    /** Escape hatch: the untouched provider response, for debugging or for
     *  features not yet surfaced through the neutral result. */
    raw: unknown;
}

/**
 * Streaming deltas, in core vocabulary.
 *
 * A stream is a sequence of these terminated by exactly one `done`. Text and
 * tool-call args arrive incrementally; the consumer reassembles them, or reads
 * the final assembled `message` off the `done` delta.
 */
export type CoreDelta =
    | { kind: "text"; text: string }
    | { kind: "thinking"; text: string }
    | { kind: "tool_call_start"; id: string; name: string }
    | { kind: "tool_call_args"; id: string; partialJson: string }
    | { kind: "done"; result: GenerateResult };

/**
 * The one interface the harness depends on. Every provider implements it.
 *
 * `generate` is the single-shot path; `stream` is the primary path for real
 * runs (long outputs, tool loops) and yields {@link CoreDelta}s ending in a
 * `done` that carries the same {@link GenerateResult} `generate` would return.
 */
export interface ModelClient {
    /** Stable name of the provider, for logging/telemetry (e.g. "anthropic"). */
    readonly provider: string;
    /** The model id this client is configured to call. Read-only through the
     *  interface; a client that supports live switching exposes {@link setModel}. */
    readonly model: string;
    /** What this client supports beyond the common path. */
    readonly capabilities: ProviderCapabilities;

    /** Switch the model every subsequent request uses, if the client supports it.
     *  Optional: a client without it is pinned to its constructed model. When
     *  present, the change is read per request, so it lands on the next turn of
     *  every conversation this one client drives. Implementations validate the id
     *  and throw on an unknown one rather than letting the provider 404 later. */
    setModel?(id: string): void;

    /** Run one turn and return the complete result. */
    generate(params: GenerateParams): Promise<GenerateResult>;

    /** Run one turn, yielding incremental deltas. The final delta is `done`. */
    stream(params: GenerateParams): AsyncIterable<CoreDelta>;
}
