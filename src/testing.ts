/**
 * Testing utilities, shipped as part of the public surface (subpath
 * `construct-harness/testing`).
 *
 * {@link FakeClient} is a scripted {@link ModelClient}: you hand it a queue of
 * turns and each `generate`/`stream` shifts the next one off. It speaks the
 * bridge contract like any real provider, so you can drive a {@link Session},
 * the loop, or an orchestration end to end with zero network calls and zero
 * spend. We use it across our own tests; it is exported here so consumers can
 * write the same kind of deterministic dry-run against their own Constructs
 * without wiring a live key.
 *
 * A "turn" is just the parts the loop cares about: the content the assistant
 * produced and the stop reason. The helper wraps them into a full
 * {@link GenerateResult} with throwaway usage/model/raw fields.
 */

import { RoleType } from "./types.ts";
import type { ContentPart, Message } from "./types.ts";
import type {
    CoreDelta,
    GenerateParams,
    GenerateResult,
    ModelClient,
    ProviderCapabilities,
    StopReason,
} from "./bridge/types.ts";

export interface ScriptedTurn {
    content: ContentPart[];
    stopReason?: StopReason; // defaults inferred from content
    /**
     * Optional reasoning trace this turn streams ahead of its content, as one or
     * more `thinking` deltas. Lets a test drive the thinking path (a UI that
     * shows the trace, a server that forwards it) deterministically. Thinking is
     * never a persisted content part, so it appears only on the stream, never in
     * the resulting message — exactly like a real provider's. Ignored by the
     * non-streaming `generate`, which has no delta channel.
     */
    thinking?: string;
}

const NO_CAPS: ProviderCapabilities = {
    thinking: false,
    effort: false,
    promptCaching: false,
    serverTools: false,
    streaming: true,
};

/** Convenience: a turn that calls one tool. */
export function callTurn(id: string, name: string, args: unknown): ScriptedTurn {
    return { content: [{ kind: "tool_call", id, name, args }], stopReason: "tool_use" };
}

/** Convenience: a turn that just emits text and stops. */
export function textTurn(text: string): ScriptedTurn {
    return { content: [{ kind: "text", text }], stopReason: "end_turn" };
}

export class FakeClient implements ModelClient {
    readonly provider = "fake";
    readonly model = "fake-model";
    readonly capabilities = NO_CAPS;

    /** Records every params object passed to `generate`, in order. */
    readonly calls: GenerateParams[] = [];

    private readonly script: ScriptedTurn[];

    constructor(script: ScriptedTurn[]) {
        this.script = [...script];
    }

    async generate(params: GenerateParams): Promise<GenerateResult> {
        this.calls.push(params);
        const turn = this.script.shift();
        if (!turn) {
            throw new Error("FakeClient: generate called more times than scripted");
        }
        const hasToolCall = turn.content.some((p) => p.kind === "tool_call");
        const stopReason = turn.stopReason ?? (hasToolCall ? "tool_use" : "end_turn");
        const message: Message = {
            sender: { role: RoleType.Agent, name: this.model },
            timestamp: 0,
            content: turn.content,
        };
        return {
            message,
            stopReason,
            usage: { inputTokens: 1, outputTokens: 1 },
            model: this.model,
            raw: null,
        };
    }

    /**
     * Stream the next scripted turn as deltas: text parts arrive as `text`
     * deltas, tool calls as a `tool_call_start` + a single `tool_call_args` with
     * the args as JSON, and the turn ends with the same `done` result `generate`
     * would return. Records params in `calls` too, so streaming tests can assert
     * on what the loop sent.
     */
    async *stream(params: GenerateParams): AsyncIterable<CoreDelta> {
        this.calls.push(params);
        const turn = this.script.shift();
        if (!turn) {
            throw new Error("FakeClient: stream called more times than scripted");
        }
        const hasToolCall = turn.content.some((p) => p.kind === "tool_call");
        const stopReason = turn.stopReason ?? (hasToolCall ? "tool_use" : "end_turn");

        // A real provider streams its reasoning trace first, then the answer.
        if (turn.thinking) yield { kind: "thinking", text: turn.thinking };

        for (const part of turn.content) {
            if (part.kind === "text") {
                yield { kind: "text", text: part.text };
            } else if (part.kind === "tool_call") {
                yield { kind: "tool_call_start", id: part.id, name: part.name };
                yield {
                    kind: "tool_call_args",
                    id: part.id,
                    partialJson: JSON.stringify(part.args ?? {}),
                };
            }
        }

        const message: Message = {
            sender: { role: RoleType.Agent, name: this.model },
            timestamp: 0,
            content: turn.content,
        };
        yield {
            kind: "done",
            result: {
                message,
                stopReason,
                usage: { inputTokens: 1, outputTokens: 1 },
                model: this.model,
                raw: null,
            },
        };
    }
}
