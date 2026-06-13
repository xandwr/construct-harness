/**
 * Tests for the embeddings module: vector (de)serialization, cosine similarity,
 * and the {@link OpenAIEmbedder} over an injected fake `fetch` (no network).
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
    OpenAIEmbedder,
    EmbeddingError,
    cosineSimilarity,
    vectorToBlob,
    blobToVector,
    embedOne,
} from "../src/embeddings.ts";

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

test("vectorToBlob / blobToVector round-trips a float32 vector", () => {
    const vec = Float32Array.from([0.5, -0.25, 1, 0, 0.125]);
    const back = blobToVector(vectorToBlob(vec));
    assert.deepEqual([...back], [...vec]);
});

test("vectorToBlob does not leak a shared backing buffer", () => {
    // A view over a larger buffer must serialize only its own elements.
    const big = Float32Array.from([1, 2, 3, 4]);
    const view = big.subarray(1, 3); // [2, 3]
    const back = blobToVector(vectorToBlob(view));
    assert.deepEqual([...back], [2, 3]);
});

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

test("cosineSimilarity: identical direction is 1, opposite is -1, orthogonal is 0", () => {
    const a = Float32Array.from([1, 0]);
    assert.ok(Math.abs(cosineSimilarity(a, Float32Array.from([2, 0])) - 1) < 1e-6);
    assert.ok(Math.abs(cosineSimilarity(a, Float32Array.from([-1, 0])) + 1) < 1e-6);
    assert.ok(Math.abs(cosineSimilarity(a, Float32Array.from([0, 1]))) < 1e-6);
});

test("cosineSimilarity returns 0 for mismatched lengths or a zero vector", () => {
    assert.equal(cosineSimilarity(Float32Array.from([1, 2]), Float32Array.from([1])), 0);
    assert.equal(cosineSimilarity(Float32Array.from([0, 0]), Float32Array.from([1, 1])), 0);
});

// ---------------------------------------------------------------------------
// OpenAIEmbedder — config
// ---------------------------------------------------------------------------

test("OpenAIEmbedder requires a key and defaults model/dimensions", () => {
    assert.throws(() => new OpenAIEmbedder({ apiKey: "" }), EmbeddingError);
    const e = new OpenAIEmbedder({ apiKey: "sk-test" });
    assert.equal(e.provider, "openai");
    assert.equal(e.model, "text-embedding-3-small");
    assert.equal(e.dimensions, 1536);
});

test("OpenAIEmbedder rejects dimensions above the model's native size", () => {
    assert.throws(
        () => new OpenAIEmbedder({ apiKey: "sk-test", dimensions: 99999 }),
        EmbeddingError,
    );
    // A valid reduction is accepted and reflected in .dimensions.
    const e = new OpenAIEmbedder({ apiKey: "sk-test", dimensions: 256 });
    assert.equal(e.dimensions, 256);
});

// ---------------------------------------------------------------------------
// OpenAIEmbedder — request/response over a fake fetch
// ---------------------------------------------------------------------------

/** Build a fake fetch that returns a fixed-length unit vector per input,
 *  echoing back the request so tests can assert on what was sent. */
function fakeFetch(handler: (url: string, init: RequestInit) => Response): typeof fetch {
    return (async (url: string, init: RequestInit) =>
        handler(url, init)) as unknown as typeof fetch;
}

function okResponse(embeddings: number[][]): Response {
    const data = embeddings.map((embedding, index) => ({ embedding, index }));
    return new Response(JSON.stringify({ data }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
    });
}

test("embed sends model + inputs and returns one normalized vector per input", async () => {
    let captured: { url: string; body: Record<string, unknown> } | undefined;
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        fetchImpl: fakeFetch((url, init) => {
            captured = { url, body: JSON.parse(init.body as string) };
            // Return un-normalized vectors; the embedder must normalize them.
            return okResponse([
                [3, 4],
                [0, 10],
            ]);
        }),
    });

    const vectors = await e.embed(["hello", "world"]);
    assert.equal(vectors.length, 2);
    // [3,4] has length 5 → normalized to [0.6, 0.8].
    assert.ok(Math.abs(vectors[0][0] - 0.6) < 1e-6);
    assert.ok(Math.abs(vectors[0][1] - 0.8) < 1e-6);
    // [0,10] → [0, 1].
    assert.ok(Math.abs(vectors[1][1] - 1) < 1e-6);

    assert.ok(captured);
    assert.match(captured.url, /\/v1\/embeddings$/);
    assert.equal(captured.body.model, "text-embedding-3-small");
    assert.deepEqual(captured.body.input, ["hello", "world"]);
});

test("embed([]) makes no request and returns []", async () => {
    let called = false;
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        fetchImpl: fakeFetch(() => {
            called = true;
            return okResponse([]);
        }),
    });
    assert.deepEqual(await e.embed([]), []);
    assert.equal(called, false);
});

test("embed substitutes a space for empty strings so indices line up", async () => {
    let sentInput: unknown;
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        fetchImpl: fakeFetch((_url, init) => {
            sentInput = JSON.parse(init.body as string).input;
            return okResponse([[1, 0]]);
        }),
    });
    await e.embed([""]);
    assert.deepEqual(sentInput, [" "]);
});

test("embed reorders out-of-order response vectors by index", async () => {
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        fetchImpl: fakeFetch(
            () =>
                new Response(
                    JSON.stringify({
                        data: [
                            { embedding: [0, 1], index: 1 },
                            { embedding: [1, 0], index: 0 },
                        ],
                    }),
                    { status: 200, headers: { "Content-Type": "application/json" } },
                ),
        ),
    });
    const [first, second] = await e.embed(["a", "b"]);
    assert.ok(Math.abs(first[0] - 1) < 1e-6); // index 0 → [1,0]
    assert.ok(Math.abs(second[1] - 1) < 1e-6); // index 1 → [0,1]
});

test("embed passes the dimensions knob through when reducing", async () => {
    let body: Record<string, unknown> | undefined;
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        dimensions: 8,
        fetchImpl: fakeFetch((_url, init) => {
            body = JSON.parse(init.body as string);
            return okResponse([[1, 0]]);
        }),
    });
    await e.embed(["x"]);
    assert.equal(body?.dimensions, 8);
});

test("embedOne returns the single vector for one input", async () => {
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        fetchImpl: fakeFetch(() => okResponse([[0, 5]])),
    });
    const vec = await embedOne(e, "solo");
    assert.ok(Math.abs(vec[1] - 1) < 1e-6);
});

// ---------------------------------------------------------------------------
// OpenAIEmbedder — error paths
// ---------------------------------------------------------------------------

test("a non-2xx response becomes an EmbeddingError carrying status + detail", async () => {
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        fetchImpl: fakeFetch(() => new Response("rate limited", { status: 429 })),
    });
    await assert.rejects(e.embed(["x"]), (err: unknown) => {
        assert.ok(err instanceof EmbeddingError);
        assert.match(err.message, /429/);
        assert.match(err.message, /rate limited/);
        return true;
    });
});

test("a transport failure becomes an EmbeddingError", async () => {
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        fetchImpl: (async () => {
            throw new Error("ECONNREFUSED");
        }) as unknown as typeof fetch,
    });
    await assert.rejects(e.embed(["x"]), EmbeddingError);
});

test("a vector-count mismatch becomes an EmbeddingError", async () => {
    const e = new OpenAIEmbedder({
        apiKey: "sk-test",
        fetchImpl: fakeFetch(() => okResponse([[1, 0]])), // 1 vector for 2 inputs
    });
    await assert.rejects(e.embed(["a", "b"]), EmbeddingError);
});

test("a key from the environment is picked up when none is passed", () => {
    const prev = process.env.OPENAI_API_KEY;
    try {
        process.env.OPENAI_API_KEY = "sk-env";
        assert.doesNotThrow(() => new OpenAIEmbedder());
    } finally {
        if (prev === undefined) delete process.env.OPENAI_API_KEY;
        else process.env.OPENAI_API_KEY = prev;
    }
});
