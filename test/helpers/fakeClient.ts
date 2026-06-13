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
    streaming: false,
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

    // The loop under test uses only `generate`; `stream` is required by the
    // interface but unused here.
    async *stream(): AsyncIterable<CoreDelta> {
        throw new Error("FakeClient.stream is not implemented");
    }
}
