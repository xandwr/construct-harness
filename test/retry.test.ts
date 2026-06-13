/**
 * Tests for the provider-neutral retry policy ({@link withRetry},
 * {@link backoffDelay}) and the error taxonomy ({@link HarnessError},
 * {@link isRetryableKind}).
 *
 * Sleep and jitter are injected so these are deterministic and instant.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { HarnessError, isRetryableKind } from "../src/bridge/errors.ts";
import type { ErrorKind } from "../src/bridge/errors.ts";
import { withRetry, backoffDelay } from "../src/bridge/retry.ts";

/** Collects the sleep durations so we can assert on backoff without waiting. */
function fakeSleep() {
    const slept: number[] = [];
    return {
        slept,
        sleep: async (ms: number) => {
            slept.push(ms);
        },
    };
}

const retryable = (kind: ErrorKind = "rate_limit", retryAfterMs?: number) =>
    new HarnessError(kind, { kind, retryable: true, retryAfterMs });
const fatal = (kind: ErrorKind = "auth") => new HarnessError(kind, { kind, retryable: false });

test("isRetryableKind matches the taxonomy", () => {
    for (const k of ["rate_limit", "overloaded", "server", "network", "timeout"] as ErrorKind[]) {
        assert.equal(isRetryableKind(k), true, `${k} should be retryable`);
    }
    for (const k of ["auth", "invalid_request", "refusal", "canceled", "unknown"] as ErrorKind[]) {
        assert.equal(isRetryableKind(k), false, `${k} should not be retryable`);
    }
});

test("returns the first success without sleeping", async () => {
    const { slept, sleep } = fakeSleep();
    const out = await withRetry(async () => 42, { sleep });
    assert.equal(out, 42);
    assert.equal(slept.length, 0);
});

test("retries a retryable error then succeeds", async () => {
    const { slept, sleep } = fakeSleep();
    let calls = 0;
    const out = await withRetry(
        async () => {
            calls++;
            if (calls < 3) throw retryable();
            return "ok";
        },
        { sleep, random: () => 0.5 },
    );
    assert.equal(out, "ok");
    assert.equal(calls, 3);
    assert.equal(slept.length, 2, "should have backed off twice");
});

test("does not retry a non-retryable error", async () => {
    const { slept, sleep } = fakeSleep();
    let calls = 0;
    await assert.rejects(
        withRetry(
            async () => {
                calls++;
                throw fatal("auth");
            },
            { sleep },
        ),
        (e) => e instanceof HarnessError && e.kind === "auth",
    );
    assert.equal(calls, 1, "auth error must not be retried");
    assert.equal(slept.length, 0);
});

test("gives up after maxAttempts and rethrows the last error", async () => {
    const { slept, sleep } = fakeSleep();
    let calls = 0;
    await assert.rejects(
        withRetry(
            async () => {
                calls++;
                throw retryable("server");
            },
            { sleep, maxAttempts: 3, random: () => 0 },
        ),
        (e) => e instanceof HarnessError && e.kind === "server",
    );
    assert.equal(calls, 3, "should attempt exactly maxAttempts times");
    assert.equal(slept.length, 2, "sleeps between attempts, not after the last");
});

test("a non-HarnessError propagates without retry", async () => {
    const { slept, sleep } = fakeSleep();
    let calls = 0;
    await assert.rejects(
        withRetry(
            async () => {
                calls++;
                throw new Error("raw");
            },
            { sleep },
        ),
        /raw/,
    );
    assert.equal(calls, 1);
    assert.equal(slept.length, 0);
});

test("honors retryAfterMs over computed backoff", async () => {
    const { slept, sleep } = fakeSleep();
    let calls = 0;
    await withRetry(
        async () => {
            calls++;
            if (calls < 2) throw retryable("rate_limit", 1234);
            return "ok";
        },
        { sleep, random: () => 0.5 },
    );
    assert.deepEqual(slept, [1234], "should wait exactly the server-suggested delay");
});

test("onRetry is called with attempt, delay, and error", async () => {
    const { sleep } = fakeSleep();
    const seen: Array<{ attempt: number; delayMs: number; kind: string }> = [];
    let calls = 0;
    await withRetry(
        async () => {
            calls++;
            if (calls < 2) throw retryable("overloaded", 100);
            return 1;
        },
        {
            sleep,
            onRetry: ({ attempt, delayMs, error }) =>
                seen.push({ attempt, delayMs, kind: error.kind }),
        },
    );
    assert.deepEqual(seen, [{ attempt: 1, delayMs: 100, kind: "overloaded" }]);
});

// ── backoffDelay ─────────────────────────────────────────────────────────────

test("backoffDelay grows exponentially with full jitter", () => {
    const opts = { baseDelayMs: 100, maxDelayMs: 10_000, random: () => 1 };
    // random()===1 → top of the [d/2, d] range, i.e. the full capped value.
    assert.equal(backoffDelay(1, retryable(), opts), 100);
    assert.equal(backoffDelay(2, retryable(), opts), 200);
    assert.equal(backoffDelay(3, retryable(), opts), 400);
});

test("backoffDelay floors at half the window with random()===0", () => {
    const opts = { baseDelayMs: 100, maxDelayMs: 10_000, random: () => 0 };
    assert.equal(backoffDelay(3, retryable(), opts), 200); // 400/2
});

test("backoffDelay caps at maxDelayMs", () => {
    const opts = { baseDelayMs: 1000, maxDelayMs: 3000, random: () => 1 };
    assert.equal(backoffDelay(10, retryable(), opts), 3000);
});

test("backoffDelay caps a hostile retryAfter at maxDelayMs", () => {
    const opts = { baseDelayMs: 100, maxDelayMs: 5000, random: () => 0.5 };
    assert.equal(backoffDelay(1, retryable("rate_limit", 999_999), opts), 5000);
});
