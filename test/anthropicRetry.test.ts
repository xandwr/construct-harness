/**
 * Integration test: the {@link AnthropicClient} retry path.
 *
 * We don't hit the network: we replace the client's underlying SDK
 * `messages.create` with a stub that fails a couple of times before succeeding,
 * and assert the client retries (with injected, instant sleep) and ultimately
 * returns a mapped {@link GenerateResult}. This proves the retry policy, the
 * classifier, and the client are wired together, not just individually correct.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import Anthropic from "@anthropic-ai/sdk";

import { AnthropicClient } from "../src/bridge/anthropic.ts";
import { HarnessError } from "../src/bridge/errors.ts";
import { RoleType } from "../src/types.ts";
import type { Message } from "../src/types.ts";

const user = (text: string): Message => ({
    sender: { role: RoleType.User },
    timestamp: 0,
    content: [{ kind: "text", text }],
});

/** A minimal successful Anthropic message the mapper can consume. */
function fakeMessage() {
    return {
        model: "claude-test",
        content: [{ type: "text", text: "hi" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
    };
}

/** Swap in a stubbed `messages.create` on a client's private SDK. */
function stubCreate(client: AnthropicClient, impl: () => Promise<unknown>) {
    // The sdk is private; reach it for the test only.
    const sdk = (client as unknown as { sdk: Anthropic }).sdk;
    (sdk.messages as unknown as { create: () => Promise<unknown> }).create = impl;
}

test("generate retries a 529 overload then succeeds", async () => {
    const client = new AnthropicClient({
        apiKey: "test",
        retry: { sleep: async () => {}, random: () => 0 },
    });

    let calls = 0;
    stubCreate(client, async () => {
        calls++;
        if (calls < 3) {
            throw Anthropic.APIError.generate(
                529,
                { type: "error", error: { type: "overloaded_error", message: "overloaded" } },
                undefined,
                new Headers(),
            );
        }
        return fakeMessage();
    });

    const res = await client.generate({ messages: [user("hi")] });
    assert.equal(calls, 3, "should have retried twice before succeeding");
    assert.equal(res.stopReason, "end_turn");
    assert.equal(res.usage.outputTokens, 1);
});

test("generate does not retry a 401 and surfaces a HarnessError", async () => {
    const client = new AnthropicClient({
        apiKey: "test",
        retry: { sleep: async () => {} },
    });

    let calls = 0;
    stubCreate(client, async () => {
        calls++;
        throw Anthropic.APIError.generate(401, undefined, undefined, new Headers());
    });

    await assert.rejects(
        client.generate({ messages: [user("hi")] }),
        (e) => e instanceof HarnessError && e.kind === "auth" && e.retryable === false,
    );
    assert.equal(calls, 1, "auth failure must not be retried");
});

test("retry: false disables the harness retry layer", async () => {
    const client = new AnthropicClient({ apiKey: "test", retry: false });
    let calls = 0;
    stubCreate(client, async () => {
        calls++;
        throw Anthropic.APIError.generate(500, undefined, undefined, new Headers());
    });
    await assert.rejects(
        client.generate({ messages: [user("hi")] }),
        (e) => e instanceof HarnessError && e.kind === "server",
    );
    assert.equal(calls, 1, "no retries when retry is disabled");
});

// ── Cancellation: the abort signal reaches the SDK ───────────────────────────

/** Swap in a stubbed `messages.stream` on a client's private SDK, returning a
 *  minimal async-iterable stream that ends with the assembled message. */
function stubStream(client: AnthropicClient, impl: (body: unknown, options: unknown) => unknown) {
    const sdk = (client as unknown as { sdk: Anthropic }).sdk;
    (sdk.messages as unknown as { stream: typeof impl }).stream = impl;
}

test("generate forwards the caller's abort signal to the SDK as a request option", async () => {
    const client = new AnthropicClient({ apiKey: "test", retry: false });
    const controller = new AbortController();
    let seenOptions: unknown;
    stubCreate(client, async function (this: unknown, ...args: unknown[]) {
        // create(body, options): the signal must ride on the second arg.
        seenOptions = args[1];
        return fakeMessage();
    } as unknown as () => Promise<unknown>);

    await client.generate({ messages: [user("hi")], signal: controller.signal });
    assert.equal(
        (seenOptions as { signal?: AbortSignal }).signal,
        controller.signal,
        "the abort signal must reach messages.create as a request option",
    );
});

test("stream forwards the caller's abort signal to the SDK as a request option", async () => {
    const client = new AnthropicClient({ apiKey: "test", retry: false });
    const controller = new AbortController();
    let seenOptions: unknown;
    stubStream(client, (_body, options) => {
        seenOptions = options;
        // A minimal MessageStream stand-in: async-iterable plus finalMessage().
        return {
            [Symbol.asyncIterator]() {
                let done = false;
                return {
                    async next() {
                        if (done) return { done: true, value: undefined };
                        done = true;
                        return {
                            done: false,
                            value: { type: "message_stop" },
                        };
                    },
                };
            },
            async finalMessage() {
                return fakeMessage();
            },
        };
    });

    const stream = client.stream({ messages: [user("hi")], signal: controller.signal });
    for await (const _ of stream) {
        // drain
    }
    assert.equal(
        (seenOptions as { signal?: AbortSignal }).signal,
        controller.signal,
        "the abort signal must reach messages.stream as a request option",
    );
});
