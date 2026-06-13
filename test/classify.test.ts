/**
 * Tests for {@link classifyAnthropicError} — the Anthropic-specific mapping from
 * raw SDK errors to the neutral {@link HarnessError} taxonomy.
 *
 * We construct real SDK error instances (no network) and assert the resulting
 * kind, retry verdict, and retry-after extraction. This is the seam that decides
 * which failures the harness retries, so it's worth pinning precisely.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import Anthropic from "@anthropic-ai/sdk";

import { classifyAnthropicError } from "../src/bridge/anthropic.ts";
import { HarnessError } from "../src/bridge/errors.ts";

/** Build an SDK APIError of the right subclass for a status via the SDK's own
 *  factory, so we exercise the real class hierarchy classify() checks. */
function apiError(status: number, type?: string, headers?: Headers) {
    const body = type ? { type: "error", error: { type, message: `${type} occurred` } } : undefined;
    return Anthropic.APIError.generate(status, body, undefined, headers ?? new Headers());
}

test("429 → rate_limit, retryable", () => {
    const e = classifyAnthropicError(apiError(429, "rate_limit_error"));
    assert.equal(e.kind, "rate_limit");
    assert.equal(e.retryable, true);
    assert.equal(e.status, 429);
});

test("529 overloaded → overloaded, retryable", () => {
    const e = classifyAnthropicError(apiError(529, "overloaded_error"));
    assert.equal(e.kind, "overloaded");
    assert.equal(e.retryable, true);
});

test("500 → server, retryable", () => {
    const e = classifyAnthropicError(apiError(500));
    assert.equal(e.kind, "server");
    assert.equal(e.retryable, true);
});

test("401 → auth, NOT retryable", () => {
    const e = classifyAnthropicError(apiError(401));
    assert.equal(e.kind, "auth");
    assert.equal(e.retryable, false);
});

test("400 → invalid_request, NOT retryable", () => {
    const e = classifyAnthropicError(apiError(400));
    assert.equal(e.kind, "invalid_request");
    assert.equal(e.retryable, false);
});

test("body type overloaded_error refines a 500-range status", () => {
    // A server-range status whose body says overloaded should classify as
    // overloaded (still retryable, but distinguishable in logs).
    const e = classifyAnthropicError(apiError(503, "overloaded_error"));
    assert.equal(e.kind, "overloaded");
    assert.equal(e.retryable, true);
});

test("extracts retry-after-ms header (milliseconds)", () => {
    const headers = new Headers({ "retry-after-ms": "2500" });
    const e = classifyAnthropicError(apiError(429, "rate_limit_error", headers));
    assert.equal(e.retryAfterMs, 2500);
});

test("extracts retry-after header (seconds → ms)", () => {
    const headers = new Headers({ "retry-after": "3" });
    const e = classifyAnthropicError(apiError(429, "rate_limit_error", headers));
    assert.equal(e.retryAfterMs, 3000);
});

test("connection error → network, retryable", () => {
    const e = classifyAnthropicError(
        new Anthropic.APIConnectionError({ message: "socket hang up" }),
    );
    assert.equal(e.kind, "network");
    assert.equal(e.retryable, true);
});

test("timeout → timeout, retryable", () => {
    const e = classifyAnthropicError(new Anthropic.APIConnectionTimeoutError());
    assert.equal(e.kind, "timeout");
    assert.equal(e.retryable, true);
});

test("user abort → canceled, NOT retryable", () => {
    const e = classifyAnthropicError(new Anthropic.APIUserAbortError());
    assert.equal(e.kind, "canceled");
    assert.equal(e.retryable, false);
});

test("an unrecognized error → unknown, NOT retryable", () => {
    const e = classifyAnthropicError(new Error("???"));
    assert.equal(e.kind, "unknown");
    assert.equal(e.retryable, false);
    assert.match(e.message, /\?\?\?/);
});

test("an already-classified HarnessError passes through unchanged", () => {
    const original = new HarnessError("x", { kind: "server", retryable: true });
    assert.equal(classifyAnthropicError(original), original);
});
