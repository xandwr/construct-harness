/**
 * A provider-neutral error taxonomy for the bridge.
 *
 * Raw provider SDK errors are a grab-bag of status codes and class names that
 * the harness shouldn't have to pattern-match on directly: doing so would leak
 * Anthropic vocabulary into the loop and the REPL, the exact coupling the bridge
 * exists to prevent. Instead, each provider classifies its own failures into a
 * {@link HarnessError}: a small, stable set of {@link ErrorKind}s plus the one
 * fact the retry policy actually needs: whether the failure is worth retrying,
 * and after how long.
 *
 * This module owns the *shape*; the per-provider mapping (e.g.
 * `classifyAnthropicError`) lives in the provider module, the only place allowed
 * to import an SDK. The bridge interface promises that `generate`/`stream`
 * reject with a {@link HarnessError}, never a raw SDK error.
 */

/**
 * The normalized failure categories. Chosen so the retry policy and a
 * user-facing message can both be derived from the `kind` alone:
 *
 *  - `rate_limit`: throttled (HTTP 429). Retryable; usually carries a
 *    retry-after.
 *  - `overloaded`: the provider is temporarily overloaded (Anthropic's 529 /
 *    `overloaded_error`). Retryable.
 *  - `server`: a 5xx the provider didn't label more specifically. Retryable.
 *  - `network`: transport failure (connection reset/refused, DNS). Retryable.
 *  - `timeout`: the request timed out. Retryable.
 *  - `auth`: bad/missing key or insufficient permission (401/403). NOT
 *    retryable: retrying sends the same bad credential.
 *  - `invalid_request`: the request itself is malformed (400/404/422). NOT
 *    retryable: the same request fails the same way.
 *  - `refusal`: the model declined to respond. NOT a transport error and not
 *    retryable; surfaced so a caller can distinguish it from a crash.
 *  - `canceled`: the caller aborted the request. NOT retryable.
 *  - `unknown`: anything we couldn't classify. NOT retryable by default, so an
 *    unrecognized error never spins the retry loop.
 */
export type ErrorKind =
    | "rate_limit"
    | "overloaded"
    | "server"
    | "network"
    | "timeout"
    | "auth"
    | "invalid_request"
    | "refusal"
    | "canceled"
    | "unknown";

/** Fields a provider classifier supplies when building a {@link HarnessError}. */
export interface HarnessErrorInit {
    kind: ErrorKind;
    /** Whether the retry policy may retry this failure. */
    retryable: boolean;
    /** Suggested wait before retrying, in ms, when the provider told us (e.g.
     *  a `retry-after` header). The retry policy prefers this over its own
     *  backoff when present. */
    retryAfterMs?: number;
    /** HTTP status, when the failure was an HTTP response. */
    status?: number;
    /** The provider's own error code/type (e.g. `"rate_limit_error"`), for
     *  logs. */
    providerCode?: string;
    /** The untouched underlying error, for debugging. */
    cause?: unknown;
}

/**
 * The single error type the bridge throws. Carries the normalized
 * {@link ErrorKind}, the retry verdict, and the raw cause so nothing is lost.
 */
export class HarnessError extends Error {
    readonly kind: ErrorKind;
    readonly retryable: boolean;
    readonly retryAfterMs?: number;
    readonly status?: number;
    readonly providerCode?: string;
    readonly cause?: unknown;

    constructor(message: string, init: HarnessErrorInit) {
        super(message);
        this.name = "HarnessError";
        this.kind = init.kind;
        this.retryable = init.retryable;
        this.retryAfterMs = init.retryAfterMs;
        this.status = init.status;
        this.providerCode = init.providerCode;
        this.cause = init.cause;
    }
}

/** Whether `kind` is one the retry policy retries by default. Providers may
 *  still override per-error (e.g. a 429 with a hint), but this is the baseline
 *  and keeps the kind→retryable mapping in one place. */
export function isRetryableKind(kind: ErrorKind): boolean {
    switch (kind) {
        case "rate_limit":
        case "overloaded":
        case "server":
        case "network":
        case "timeout":
            return true;
        case "auth":
        case "invalid_request":
        case "refusal":
        case "canceled":
        case "unknown":
            return false;
    }
}
