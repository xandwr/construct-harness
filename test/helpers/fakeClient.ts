/**
 * A scripted {@link ModelClient} for driving the loop in tests without a network
 * call. You hand it a queue of turns; each `generate` shifts the next one off.
 *
 * A "turn" is just the parts the loop cares about: the content the assistant
 * produced and the stop reason. The helper wraps them into a full
 * {@link GenerateResult} with throwaway usage/model/raw fields.
 */

import { RoleType } from "../../src/types.ts";
import type { ContentPart, Message } from "../../src/types.ts";
import type {
    CoreDelta,
    GenerateParams,
    GenerateResult,
    ModelClient,
    ProviderCapabilities,
    StopReason,
} from "../../src/bridge/types.ts";

export interface ScriptedTurn {
    content: ContentPart[];
    stopReason?: StopReason; // defaults inferred from content
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
