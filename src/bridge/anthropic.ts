/**
 * Anthropic: the first {@link ModelClient} implementation.
 *
 * This is the only module in the bridge that imports a provider SDK. It owns
 * every Anthropic-specific fact: that `system` is a top-level field rather than
 * a message, that thinking is configured via `thinking: {type: "adaptive"}`,
 * that effort lives under `output_config`, and how content blocks map to core
 * `ContentPart`s. The mappers are pure and exported so they can be unit-tested
 * without a network call.
 */

import Anthropic from "@anthropic-ai/sdk";
import { RoleType } from "../types.ts";
import type { ContentPart, Message, ToolDef } from "../types.ts";
import type {
    CoreDelta,
    GenerateParams,
    GenerateResult,
    ModelClient,
    ProviderCapabilities,
    StopReason,
    Usage,
} from "./types.ts";
import { HarnessError, isRetryableKind } from "./errors.ts";
import type { ErrorKind } from "./errors.ts";
import { withRetry } from "./retry.ts";
import type { RetryOptions } from "./retry.ts";
import { isKnownModel } from "./models.ts";

/** Default model + token ceilings, per the Claude API guidance. */
const DEFAULT_MODEL = "claude-opus-4-8";
const DEFAULT_MAX_TOKENS = 16_000; // non-streaming: stays under SDK HTTP timeouts
const DEFAULT_STREAM_MAX_TOKENS = 64_000; // streaming: room to think + act

/**
 * Anthropic-specific request knobs, the typed narrowing of the bridge's opaque
 * `ProviderOptions`. The harness sets these only after checking
 * {@link ANTHROPIC_CAPABILITIES}.
 */
/**
 * The provider-hosted tools the model can run server-side, with no `run` of ours
 * to dispatch: Anthropic executes them and folds the result into the same
 * assistant turn (the stop reason stays `end_turn`, so the loop never tries to
 * run them). The friendly names here map to the dated tool blocks the SDK ships
 * (see {@link SERVER_TOOL_BLOCKS}); the harness sets {@link AnthropicOptions.serverTools}
 * only after checking {@link ANTHROPIC_CAPABILITIES.serverTools}.
 */
export type ServerToolName = "web_search" | "web_fetch" | "code_execution";

export interface AnthropicOptions {
    /** Reasoning depth / spend. Maps to `output_config.effort`. */
    effort?: "low" | "medium" | "high" | "xhigh" | "max";
    /** Enable adaptive thinking. Off by default on Opus 4.8. */
    thinking?: boolean;
    /** Surface a readable thinking summary (default omits it). */
    thinkingDisplay?: boolean;
    /** Cache the system prompt prefix across requests. */
    cacheSystem?: boolean;
    /**
     * Provider-hosted tools to enable for the request (web search, web fetch,
     * code execution). These run server-side: Anthropic executes them and returns
     * the result inline, so unlike a custom {@link ToolDef} there is no `run` and
     * the agentic loop never dispatches them. Omit or pass `[]` for none. Gated by
     * {@link ANTHROPIC_CAPABILITIES.serverTools}.
     */
    serverTools?: ServerToolName[];
}

export const ANTHROPIC_CAPABILITIES: ProviderCapabilities = {
    thinking: true,
    effort: true,
    promptCaching: true,
    // Anthropic hosts tools server-side (web search, web fetch, code execution);
    // we emit the dated tool blocks for them when AnthropicOptions.serverTools
    // asks (see toServerTools / buildRequest). The model runs them in-turn, so
    // the loop never dispatches a `run` for them.
    serverTools: true,
    streaming: true,
};

/**
 * Map each friendly {@link ServerToolName} to the dated SDK tool block to emit.
 * Pinned to the newest version this SDK exposes; bump these in lockstep with an
 * SDK upgrade. A `name` the model sees plus a versioned `type` is all the API
 * needs to host the tool.
 */
const SERVER_TOOL_BLOCKS: Record<ServerToolName, Anthropic.ToolUnion> = {
    web_search: { name: "web_search", type: "web_search_20260209" },
    web_fetch: { name: "web_fetch", type: "web_fetch_20260309" },
    code_execution: { name: "code_execution", type: "code_execution_20260120" },
};

/** Build the server-tool blocks for the requested tools, de-duplicated and in a
 *  stable order, or undefined when none were asked for. Unknown names are skipped
 *  defensively (the option is typed, but providerOptions is opaque upstream). */
export function toServerTools(
    names: ServerToolName[] | undefined,
): Anthropic.ToolUnion[] | undefined {
    if (!names?.length) return undefined;
    const seen = new Set<ServerToolName>();
    const out: Anthropic.ToolUnion[] = [];
    for (const name of names) {
        const block = SERVER_TOOL_BLOCKS[name];
        if (block && !seen.has(name)) {
            seen.add(name);
            out.push(block);
        }
    }
    return out.length ? out : undefined;
}

// ── Core → Anthropic ────────────────────────────────────────────────────────

/** Split core messages into Anthropic's `{system, messages}` shape.
 *
 *  Anthropic keeps system guidance top-level rather than in the turn array, so
 *  we lift every `role: "system"` text part out and concatenate it. The rest
 *  map turn-for-turn. This function is pure: no SDK, no I/O. */
export function toAnthropicMessages(messages: Message[]): {
    system: string | undefined;
    messages: Anthropic.MessageParam[];
} {
    const systemParts: string[] = [];
    const out: Anthropic.MessageParam[] = [];

    for (const msg of messages) {
        if (msg.sender.role === RoleType.System) {
            for (const part of msg.content) {
                if (part.kind === "text") systemParts.push(part.text);
            }
            continue;
        }

        const role: "user" | "assistant" =
            msg.sender.role === RoleType.Agent ? "assistant" : "user";
        out.push({ role, content: toAnthropicContent(msg.content) });
    }

    return {
        system: systemParts.length ? systemParts.join("\n\n") : undefined,
        messages: out,
    };
}

/** Map core content parts to Anthropic content blocks. */
function toAnthropicContent(parts: ContentPart[]): Anthropic.ContentBlockParam[] {
    return parts.map((part): Anthropic.ContentBlockParam => {
        switch (part.kind) {
            case "text":
                return { type: "text", text: part.text };
            case "tool_call":
                return {
                    type: "tool_use",
                    id: part.id,
                    name: part.name,
                    input: (part.args ?? {}) as Record<string, unknown>,
                };
            case "tool_result":
                return {
                    type: "tool_result",
                    tool_use_id: part.callId,
                    content: stringifyResult(part.result),
                    is_error: part.isError,
                };
        }
    });
}

/** Tool results are arbitrary JSON in the core; Anthropic wants text/blocks.
 *
 *  A tool may return a value that doesn't serialize: a circular object, or one
 *  with a throwing `toJSON`. We never let that crash the mapping: the model gets
 *  a readable placeholder instead, and the loop keeps going. */
export function stringifyResult(result: unknown): string {
    if (typeof result === "string") return result;
    try {
        const json = JSON.stringify(result);
        // JSON.stringify(undefined) and stringify of a lone function return
        // undefined; fall back to a printable form in that case.
        return json ?? String(result);
    } catch (err) {
        const reason = err instanceof Error ? err.message : String(err);
        return `[unserializable tool result: ${reason}]`;
    }
}

/** Map core tool definitions to Anthropic tools. The core's `run` is the
 *  harness's concern and never crosses the wire. */
export function toAnthropicTools(tools: ToolDef[] | undefined): Anthropic.Tool[] | undefined {
    if (!tools?.length) return undefined;
    return tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: t.parameters as Anthropic.Tool.InputSchema,
    }));
}

// ── Anthropic → Core ────────────────────────────────────────────────────────

/** Map a completed Anthropic message back into a core {@link Message}. Pure. */
export function fromAnthropicMessage(msg: Anthropic.Message): Message {
    const content: ContentPart[] = [];
    for (const block of msg.content) {
        if (block.type === "text") {
            content.push({ kind: "text", text: block.text });
        } else if (block.type === "tool_use") {
            content.push({
                kind: "tool_call",
                id: block.id,
                name: block.name,
                args: block.input,
            });
        }
        // thinking blocks are intentionally dropped from the core message;
        // the streaming path surfaces them as `thinking` deltas instead.
    }
    return {
        sender: { role: RoleType.Agent, name: msg.model },
        timestamp: Date.now(),
        content,
    };
}

/** Normalize Anthropic's stop reason to the neutral set. */
export function toStopReason(reason: Anthropic.Message["stop_reason"]): StopReason {
    switch (reason) {
        case "end_turn":
        case "stop_sequence":
            return "end_turn";
        case "tool_use":
            return "tool_use";
        case "max_tokens":
            return "max_tokens";
        case "refusal":
            return "refusal";
        default:
            return "other";
    }
}

function toUsage(usage: Anthropic.Usage): Usage {
    return {
        inputTokens: usage.input_tokens,
        outputTokens: usage.output_tokens,
        cacheReadTokens: usage.cache_read_input_tokens ?? undefined,
    };
}

function toResult(msg: Anthropic.Message): GenerateResult {
    return {
        message: fromAnthropicMessage(msg),
        stopReason: toStopReason(msg.stop_reason),
        usage: toUsage(msg.usage),
        model: msg.model,
        raw: msg,
    };
}

// ── Error classification ──────────────────────────────────────────────────────

/** Anthropic's "I'm overloaded" status, which the standard fetch error classes
 *  don't have a dedicated subclass for. */
const HTTP_OVERLOADED = 529;

/** Read a `retry-after` hint from response headers, in ms. The API may send
 *  `retry-after-ms` (milliseconds) or the standard `retry-after` (seconds);
 *  prefer the millisecond form. Returns undefined when neither is present or
 *  parseable. */
function retryAfterMsFromHeaders(headers: unknown): number | undefined {
    if (!headers || typeof (headers as Headers).get !== "function") return undefined;
    const h = headers as Headers;
    const ms = h.get("retry-after-ms");
    if (ms) {
        const n = Number.parseFloat(ms);
        if (Number.isFinite(n) && n >= 0) return n;
    }
    const secs = h.get("retry-after");
    if (secs) {
        const n = Number.parseFloat(secs);
        if (Number.isFinite(n) && n >= 0) return n * 1000;
    }
    return undefined;
}

/** Map an HTTP status to a neutral {@link ErrorKind}. */
function kindForStatus(status: number): ErrorKind {
    if (status === 429) return "rate_limit";
    if (status === HTTP_OVERLOADED) return "overloaded";
    if (status === 401 || status === 403) return "auth";
    if (status === 400 || status === 404 || status === 409 || status === 422) {
        return "invalid_request";
    }
    if (status >= 500) return "server";
    return "unknown";
}

/**
 * Classify any thrown value into a {@link HarnessError}.
 *
 * Maps the Anthropic SDK's error classes and HTTP statuses onto the neutral
 * taxonomy, lifts a `retry-after` hint off the headers when present, and falls
 * back to `unknown` (non-retryable) for anything unrecognized. Pure and exported
 * so it can be unit-tested against synthetic SDK errors without a network call.
 * Already-classified {@link HarnessError}s pass through unchanged.
 */
export function classifyAnthropicError(err: unknown): HarnessError {
    if (err instanceof HarnessError) return err;

    // Caller aborted (AbortController): not a transport failure, never retried.
    if (err instanceof Anthropic.APIUserAbortError) {
        return new HarnessError("request canceled", {
            kind: "canceled",
            retryable: false,
            cause: err,
        });
    }
    // Timeout is a connection error subclass; check it before the parent.
    if (err instanceof Anthropic.APIConnectionTimeoutError) {
        return new HarnessError("request timed out", {
            kind: "timeout",
            retryable: true,
            cause: err,
        });
    }
    if (err instanceof Anthropic.APIConnectionError) {
        return new HarnessError("network error", { kind: "network", retryable: true, cause: err });
    }
    // Any HTTP-status APIError: classify by status, then by the body `type`.
    if (err instanceof Anthropic.APIError && typeof err.status === "number") {
        let kind = kindForStatus(err.status);
        // The body type can refine an ambiguous status (e.g. a 500-range
        // overloaded_error the status alone would call "server").
        if (err.type === "overloaded_error") kind = "overloaded";
        if (err.type === "rate_limit_error") kind = "rate_limit";
        return new HarnessError(err.message || `HTTP ${err.status}`, {
            kind,
            retryable: isRetryableKind(kind),
            retryAfterMs: retryAfterMsFromHeaders(err.headers),
            status: err.status,
            providerCode: err.type ?? undefined,
            cause: err,
        });
    }

    const message = err instanceof Error ? err.message : String(err);
    return new HarnessError(message || "unknown error", {
        kind: "unknown",
        retryable: false,
        cause: err,
    });
}

// ── Client ──────────────────────────────────────────────────────────────────

export interface AnthropicClientConfig {
    /** Defaults to the `ANTHROPIC_API_KEY` env var via the SDK. */
    apiKey?: string;
    /** Defaults to {@link DEFAULT_MODEL}. */
    model?: string;
    /**
     * Harness-level retry policy for `generate` and stream-start. Pass `false`
     * to disable (the SDK still does its own internal retries); omit for the
     * defaults in {@link withRetry}. We set the SDK's own `maxRetries` to 0 so
     * retries aren't applied twice: this layer is the single, observable retry
     * point, keying off the neutral {@link HarnessError} taxonomy.
     */
    retry?: RetryOptions | false;
}

export class AnthropicClient implements ModelClient {
    readonly provider = "anthropic";
    readonly capabilities = ANTHROPIC_CAPABILITIES;

    /** The model id every request is built with. Mutable through {@link model}'s
     *  setter so the settings page can switch models live: {@link buildRequest}
     *  reads it per request, so a change takes effect on the very next turn for
     *  every conversation this one client drives. */
    private currentModel: string;
    private readonly sdk: Anthropic;
    private readonly retry: RetryOptions | false;

    constructor(config: AnthropicClientConfig = {}) {
        // maxRetries: 0: this client owns retries (see `retry`), so the SDK's
        // own retry loop would otherwise double up and hide the classified error.
        this.sdk = new Anthropic({
            ...(config.apiKey ? { apiKey: config.apiKey } : {}),
            maxRetries: 0,
        });
        this.currentModel = config.model ?? DEFAULT_MODEL;
        this.retry = config.retry ?? {};
    }

    /** The model id this client currently calls. Satisfies {@link ModelClient.model};
     *  {@link setModel} makes it live. */
    get model(): string {
        return this.currentModel;
    }

    /**
     * Switch the model every subsequent request uses (the {@link ModelClient.setModel}
     * capability). Validated against the bridge's model catalogue
     * ({@link isKnownModel}) so a typo can't be handed to the provider to 404 on
     * the next turn — an unknown id throws an `invalid_request` {@link HarnessError}
     * the caller surfaces, and the live model is left unchanged. Takes effect on
     * the next request: there is no per-conversation pinning, so it switches the
     * whole process at once.
     */
    setModel(id: string): void {
        const next = id.trim();
        if (!isKnownModel(next)) {
            throw new HarnessError(`unknown model "${id}"`, {
                kind: "invalid_request",
                retryable: false,
            });
        }
        this.currentModel = next;
    }

    /** Run `fn` under the configured retry policy, mapping every failure to a
     *  {@link HarnessError} first so the policy sees the neutral `retryable`
     *  verdict and the caller never receives a raw SDK error. */
    private async run<T>(fn: () => Promise<T>): Promise<T> {
        const attempt = () => fn().catch((err) => Promise.reject(classifyAnthropicError(err)));
        if (this.retry === false) return attempt();
        return withRetry(attempt, this.retry);
    }

    /** Build the shared request body from neutral params + Anthropic options. */
    private buildRequest(
        params: GenerateParams,
        defaultMaxTokens: number,
    ): Anthropic.MessageCreateParams {
        const opts = (params.providerOptions ?? {}) as AnthropicOptions;
        const { system, messages } = toAnthropicMessages(params.messages);

        const req: Anthropic.MessageCreateParams = {
            model: this.model,
            max_tokens: params.maxTokens ?? defaultMaxTokens,
            messages,
        };

        if (system !== undefined) {
            req.system = opts.cacheSystem
                ? [{ type: "text", text: system, cache_control: { type: "ephemeral" } }]
                : system;
        }

        // Custom tools (ours, dispatched by the loop) and provider-hosted server
        // tools (run by Anthropic in-turn) share the one `tools` array.
        const custom = toAnthropicTools(params.tools);
        const server = toServerTools(opts.serverTools);
        if (custom || server) req.tools = [...(custom ?? []), ...(server ?? [])];

        if (opts.thinking) {
            req.thinking = {
                type: "adaptive",
                ...(opts.thinkingDisplay ? { display: "summarized" } : {}),
            };
        }
        if (opts.effort) {
            req.output_config = { effort: opts.effort };
        }

        return req;
    }

    async generate(params: GenerateParams): Promise<GenerateResult> {
        const req = this.buildRequest(params, DEFAULT_MAX_TOKENS);
        const msg = await this.run(() => this.sdk.messages.create({ ...req, stream: false }));
        return toResult(msg);
    }

    async *stream(params: GenerateParams): AsyncIterable<CoreDelta> {
        const req = this.buildRequest(params, DEFAULT_STREAM_MAX_TOKENS);

        // `messages.stream()` returns synchronously; the connection (and any
        // rate-limit/overload/network failure) surfaces on first iteration. So
        // retrying around the call itself would catch nothing. Instead, retry
        // the act of opening the stream *and pulling its first event* as one
        // unit: that's the window where a retry is safe (no delta emitted yet).
        // Once the first event is in hand, a later mid-stream failure can't be
        // retried without duplicating emitted text: it's classified and
        // rethrown below for the loop/REPL to handle.
        const toolIdByIndex = new Map<number, string>();
        // Open the stream and pull its first event as one retried unit, holding
        // onto the single iterator so the rest of the loop continues from it
        // (not a fresh one).
        const { stream, iterator, first } = await this.run(async () => {
            const s = this.sdk.messages.stream(req);
            const it = s[Symbol.asyncIterator]();
            const r = await it.next();
            return { stream: s, iterator: it, first: r };
        });

        try {
            // Replay the already-pulled first event, then drain the rest.
            for (let r = first; !r.done; r = await iterator.next()) {
                const event = r.value;
                if (event.type === "content_block_start") {
                    const block = event.content_block;
                    if (block.type === "tool_use") {
                        toolIdByIndex.set(event.index, block.id);
                        yield { kind: "tool_call_start", id: block.id, name: block.name };
                    }
                } else if (event.type === "content_block_delta") {
                    const delta = event.delta;
                    if (delta.type === "text_delta") {
                        yield { kind: "text", text: delta.text };
                    } else if (delta.type === "thinking_delta") {
                        yield { kind: "thinking", text: delta.thinking };
                    } else if (delta.type === "input_json_delta") {
                        yield {
                            kind: "tool_call_args",
                            id: toolIdByIndex.get(event.index) ?? "",
                            partialJson: delta.partial_json,
                        };
                    }
                }
            }

            const final = await stream.finalMessage();
            yield { kind: "done", result: toResult(final) };
        } catch (err) {
            // A failure once the stream is underway can't be retried here without
            // re-emitting text the consumer already saw, but it must still reach
            // the caller as a classified HarnessError rather than a raw SDK
            // error. The loop/REPL above decides whether to restart the turn.
            throw classifyAnthropicError(err);
        }
    }
}
