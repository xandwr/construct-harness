/**
 * Anthropic — the first {@link ModelClient} implementation.
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

/** Default model + token ceilings, per the Claude API guidance. */
const DEFAULT_MODEL = "claude-opus-4-8";
const DEFAULT_MAX_TOKENS = 16_000; // non-streaming: stays under SDK HTTP timeouts
const DEFAULT_STREAM_MAX_TOKENS = 64_000; // streaming: room to think + act

/**
 * Anthropic-specific request knobs, the typed narrowing of the bridge's opaque
 * `ProviderOptions`. The harness sets these only after checking
 * {@link ANTHROPIC_CAPABILITIES}.
 */
export interface AnthropicOptions {
    /** Reasoning depth / spend. Maps to `output_config.effort`. */
    effort?: "low" | "medium" | "high" | "xhigh" | "max";
    /** Enable adaptive thinking. Off by default on Opus 4.8. */
    thinking?: boolean;
    /** Surface a readable thinking summary (default omits it). */
    thinkingDisplay?: boolean;
    /** Cache the system prompt prefix across requests. */
    cacheSystem?: boolean;
}

export const ANTHROPIC_CAPABILITIES: ProviderCapabilities = {
    thinking: true,
    effort: true,
    promptCaching: true,
    serverTools: true,
    streaming: true,
};

// ── Core → Anthropic ────────────────────────────────────────────────────────

/** Split core messages into Anthropic's `{system, messages}` shape.
 *
 *  Anthropic keeps system guidance top-level rather than in the turn array, so
 *  we lift every `role: "system"` text part out and concatenate it. The rest
 *  map turn-for-turn. This function is pure — no SDK, no I/O. */
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

/** Tool results are arbitrary JSON in the core; Anthropic wants text/blocks. */
function stringifyResult(result: unknown): string {
    if (typeof result === "string") return result;
    return JSON.stringify(result);
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

// ── Client ──────────────────────────────────────────────────────────────────

export interface AnthropicClientConfig {
    /** Defaults to the `ANTHROPIC_API_KEY` env var via the SDK. */
    apiKey?: string;
    /** Defaults to {@link DEFAULT_MODEL}. */
    model?: string;
}

export class AnthropicClient implements ModelClient {
    readonly provider = "anthropic";
    readonly model: string;
    readonly capabilities = ANTHROPIC_CAPABILITIES;

    private readonly sdk: Anthropic;

    constructor(config: AnthropicClientConfig = {}) {
        this.sdk = new Anthropic(config.apiKey ? { apiKey: config.apiKey } : {});
        this.model = config.model ?? DEFAULT_MODEL;
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

        const tools = toAnthropicTools(params.tools);
        if (tools) req.tools = tools;

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
        const msg = await this.sdk.messages.create({ ...req, stream: false });
        return toResult(msg);
    }

    async *stream(params: GenerateParams): AsyncIterable<CoreDelta> {
        const req = this.buildRequest(params, DEFAULT_STREAM_MAX_TOKENS);
        const stream = this.sdk.messages.stream(req);

        // Anthropic's input_json_delta events identify their block by `index`,
        // not by tool-use id. Track index→id from the block-start events so we
        // can stamp each arg fragment with the call it belongs to.
        const toolIdByIndex = new Map<number, string>();

        for await (const event of stream) {
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
    }
}
