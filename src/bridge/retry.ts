/**
 * A small, provider-neutral retry policy.
 *
 * The provider SDK has its own retries, but the harness needs a retry story it
 * controls and can reason about: one that keys off the normalized
 * {@link HarnessError} taxonomy (retry only what's `retryable`), honors a
 * provider's `retry-after` hint when present, and otherwise backs off
 * exponentially with jitter. {@link withRetry} wraps any async thunk — a
 * `generate` call, the start of a `stream` — in that policy.
 *
 * Determinism: backoff jitter draws from an injectable `random` (default
 * `Math.random`), so tests can pin it. Sleeping is likewise injectable so tests
 * don't actually wait.
 */

import { HarnessError } from "./errors.ts";

/** Knobs for {@link withRetry}. All optional; defaults suit an interactive
 *  session (a few quick retries, then give up rather than hang the user). */
export interface RetryOptions {
    /** Max attempts total, including the first. Default {@link DEFAULT_MAX_ATTEMPTS}. */
    maxAttempts?: number;
    /** Base backoff in ms; attempt n waits ~base * 2^(n-1). Default
     *  {@link DEFAULT_BASE_DELAY_MS}. */
    baseDelayMs?: number;
    /** Ceiling on any single backoff wait, in ms. Default {@link DEFAULT_MAX_DELAY_MS}. */
    maxDelayMs?: number;
    /** Source of [0,1) jitter. Injectable for deterministic tests. */
    random?: () => number;
    /** Sleep implementation. Injectable so tests don't actually wait. */
    sleep?: (ms: number) => Promise<void>;
    /** Called before each retry with the attempt that just failed (1-based),
     *  the chosen delay, and the error. For logging/telemetry. */
    onRetry?: (info: { attempt: number; delayMs: number; error: HarnessError }) => void;
}

export const DEFAULT_MAX_ATTEMPTS = 4;
export const DEFAULT_BASE_DELAY_MS = 500;
export const DEFAULT_MAX_DELAY_MS = 30_000;

const defaultSleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/**
 * Compute the backoff for a failed attempt.
 *
 * Honors the error's `retryAfterMs` when the provider supplied one (still capped
 * at `maxDelayMs` so a hostile/huge header can't park us for minutes).
 * Otherwise: exponential base*2^(attempt-1), capped, with full jitter in
 * [delay/2, delay] to avoid thundering-herd retries.
 */
export function backoffDelay(
    attempt: number,
    error: HarnessError,
    opts: Required<Pick<RetryOptions, "baseDelayMs" | "maxDelayMs">> & { random: () => number },
): number {
    if (error.retryAfterMs !== undefined && error.retryAfterMs >= 0) {
        return Math.min(error.retryAfterMs, opts.maxDelayMs);
    }
    const exp = opts.baseDelayMs * 2 ** (attempt - 1);
    const capped = Math.min(exp, opts.maxDelayMs);
    // Full jitter: a random point in [capped/2, capped].
    return Math.round(capped / 2 + opts.random() * (capped / 2));
}

/**
 * Run `fn`, retrying on retryable {@link HarnessError}s per the policy.
 *
 * Stops and rethrows immediately on a non-retryable error (auth, bad request,
 * refusal, cancel) — retrying those only wastes time and tokens. After the last
 * attempt, rethrows the final error. A thrown value that isn't a
 * {@link HarnessError} is treated as non-retryable and rethrown as-is: the
 * policy only acts on errors a provider has already classified.
 */
export async function withRetry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
    const maxAttempts = options.maxAttempts ?? DEFAULT_MAX_ATTEMPTS;
    const baseDelayMs = options.baseDelayMs ?? DEFAULT_BASE_DELAY_MS;
    const maxDelayMs = options.maxDelayMs ?? DEFAULT_MAX_DELAY_MS;
    const random = options.random ?? Math.random;
    const sleep = options.sleep ?? defaultSleep;

    let lastError: unknown;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn();
        } catch (err) {
            lastError = err;
            // Only a classified, retryable error is eligible. Anything else —
            // including a raw error that escaped classification — propagates.
            if (!(err instanceof HarnessError) || !err.retryable) throw err;
            // No attempts left: give up with the real error.
            if (attempt >= maxAttempts) throw err;

            const delayMs = backoffDelay(attempt, err, { baseDelayMs, maxDelayMs, random });
            options.onRetry?.({ attempt, delayMs, error: err });
            await sleep(delayMs);
        }
    }
    // Unreachable: the loop either returns or throws. Satisfies the type checker.
    throw lastError;
}
