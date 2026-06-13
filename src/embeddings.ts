/**
 * Embeddings — turn text into vectors for semantic (meaning-based) recall.
 *
 * The {@link Embedder} interface is provider-neutral on purpose, mirroring the
 * bridge's {@link ModelClient}: the {@link MemoryStore} knows how to *store* and
 * *compare* vectors but never how to *produce* them, so swapping OpenAI for a
 * local model later touches only this file. {@link OpenAIEmbedder} is the first
 * implementation, calling OpenAI's hosted `/v1/embeddings` endpoint over plain
 * `fetch` (no SDK dependency — the request shape is tiny and stable).
 *
 * Vectors are returned L2-normalized, so a downstream dot product *is* cosine
 * similarity. The store relies on that invariant.
 */

/** Thrown when embedding fails — bad config, a transport error, or a bad
 *  response. Callers can `instanceof`-check to distinguish it from storage
 *  errors and decide whether to degrade gracefully (e.g. fall back to FTS). */
export class EmbeddingError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "EmbeddingError";
    }
}

/**
 * Produces embedding vectors for text. Every vector this returns is
 * L2-normalized and exactly {@link Embedder.dimensions} long.
 */
export interface Embedder {
    /** Stable provider id, for logging (e.g. "openai"). */
    readonly provider: string;
    /** The embedding model in use. */
    readonly model: string;
    /** Vector length every result carries. Fixed for a given model. */
    readonly dimensions: number;

    /**
     * Embed a batch of strings, returning one normalized vector per input, in
     * order. An empty input array yields an empty result without a network call.
     */
    embed(texts: string[]): Promise<Float32Array[]>;
}

/** Convenience: embed a single string and return its one vector. */
export async function embedOne(embedder: Embedder, text: string): Promise<Float32Array> {
    const [vec] = await embedder.embed([text]);
    return vec;
}

/**
 * Cosine similarity of two vectors. When both are L2-normalized (as everything
 * {@link Embedder} returns), this reduces to a dot product — but we guard the
 * general case so a stray un-normalized vector can't silently skew ranking.
 * Returns a value in roughly [-1, 1]; 0 for a zero-length or mismatched vector.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) return 0;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na === 0 || nb === 0) return 0;
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return dot / denom;
}

/** L2-normalize in place and return the same array. A zero vector is left as-is
 *  (there's no meaningful direction to normalize it to). */
function normalize(vec: Float32Array): Float32Array {
    let sum = 0;
    for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
    if (sum === 0) return vec;
    const inv = 1 / Math.sqrt(sum);
    for (let i = 0; i < vec.length; i++) vec[i] *= inv;
    return vec;
}

// ── Vector ↔ BLOB serialization ───────────────────────────────────────────────
//
// SQLite stores embeddings as a raw little-endian Float32 BLOB. We keep the
// (de)serializers here next to the format they assume so the store stays
// agnostic about how a vector is laid out on disk.

/** Pack a vector into a Buffer of little-endian float32s for BLOB storage. */
export function vectorToBlob(vec: Float32Array): Uint8Array {
    // Copy through a fresh ArrayBuffer so we never persist a view over a larger,
    // shared buffer (which would write trailing garbage).
    const out = new Float32Array(vec.length);
    out.set(vec);
    return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

/** Read a Float32 vector back out of a stored BLOB. */
export function blobToVector(blob: Uint8Array): Float32Array {
    // The blob's byteLength must be a multiple of 4; a corrupt row yields a
    // shorter vector, which cosineSimilarity then treats as a non-match.
    const count = Math.floor(blob.byteLength / 4);
    const out = new Float32Array(count);
    const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
    for (let i = 0; i < count; i++) out[i] = view.getFloat32(i * 4, true);
    return out;
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

/** Per-model output dimensions, so the store can declare its vector column. */
const OPENAI_DIMENSIONS: Record<string, number> = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
};

const DEFAULT_OPENAI_MODEL = "text-embedding-3-small";
const OPENAI_ENDPOINT = "https://api.openai.com/v1/embeddings";

/** OpenAI caps a single embeddings request; batch larger inputs under this. */
const MAX_BATCH = 128;

export interface OpenAIEmbedderConfig {
    /** OpenAI secret/project key. Defaults to `OPENAI_API_KEY`, then
     *  `OPENAI_ADMIN_KEY`, from the environment. */
    apiKey?: string;
    /** Embedding model. Defaults to {@link DEFAULT_OPENAI_MODEL}. */
    model?: string;
    /**
     * Optionally shorten the output vector (text-embedding-3-* support native
     * dimension reduction). Must not exceed the model's native size. Smaller
     * vectors are cheaper to store and scan at a small recall cost.
     */
    dimensions?: number;
    /** Override the endpoint (for a proxy or gateway). */
    endpoint?: string;
    /** Injected for tests; defaults to the global `fetch`. */
    fetchImpl?: typeof fetch;
}

interface OpenAIEmbeddingResponse {
    data: Array<{ embedding: number[]; index: number }>;
}

/**
 * {@link Embedder} backed by OpenAI's hosted embeddings API.
 *
 * Batches inputs under the provider's per-request cap, normalizes every vector,
 * and surfaces any failure as an {@link EmbeddingError} so callers can fall back
 * to lexical search rather than crash a turn.
 */
export class OpenAIEmbedder implements Embedder {
    readonly provider = "openai";
    readonly model: string;
    readonly dimensions: number;

    private readonly apiKey: string;
    private readonly endpoint: string;
    private readonly fetchImpl: typeof fetch;
    /** Sent to the API only when reducing below the model's native size. */
    private readonly requestedDimensions: number | undefined;

    constructor(config: OpenAIEmbedderConfig = {}) {
        const apiKey = config.apiKey ?? process.env.OPENAI_API_KEY ?? process.env.OPENAI_ADMIN_KEY;
        if (!apiKey) {
            throw new EmbeddingError(
                "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_ADMIN_KEY), or pass apiKey",
            );
        }
        this.apiKey = apiKey;
        this.model = config.model ?? DEFAULT_OPENAI_MODEL;

        const native = OPENAI_DIMENSIONS[this.model];
        if (config.dimensions !== undefined) {
            if (!Number.isInteger(config.dimensions) || config.dimensions <= 0) {
                throw new EmbeddingError("dimensions must be a positive integer");
            }
            if (native && config.dimensions > native) {
                throw new EmbeddingError(
                    `dimensions ${config.dimensions} exceeds ${this.model}'s native ${native}`,
                );
            }
            this.dimensions = config.dimensions;
            this.requestedDimensions = config.dimensions;
        } else {
            // Fall back to a sane default for unknown models so the column width
            // is still defined; the API is the final authority on actual length.
            this.dimensions = native ?? 1536;
            this.requestedDimensions = undefined;
        }

        this.endpoint = config.endpoint ?? OPENAI_ENDPOINT;
        this.fetchImpl = config.fetchImpl ?? fetch;
    }

    async embed(texts: string[]): Promise<Float32Array[]> {
        if (texts.length === 0) return [];

        const out: Float32Array[] = [];
        for (let i = 0; i < texts.length; i += MAX_BATCH) {
            const batch = texts.slice(i, i + MAX_BATCH);
            const vectors = await this.embedBatch(batch);
            out.push(...vectors);
        }
        return out;
    }

    /** One network round-trip for a batch already known to be within the cap. */
    private async embedBatch(batch: string[]): Promise<Float32Array[]> {
        // OpenAI rejects empty strings; substitute a single space so indices in
        // the response still line up one-to-one with the input batch.
        const inputs = batch.map((t) => (t.length === 0 ? " " : t));

        const body: Record<string, unknown> = { model: this.model, input: inputs };
        if (this.requestedDimensions !== undefined) {
            body.dimensions = this.requestedDimensions;
        }

        let res: Response;
        try {
            res = await this.fetchImpl(this.endpoint, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${this.apiKey}`,
                },
                body: JSON.stringify(body),
            });
        } catch (err) {
            throw new EmbeddingError(
                `embeddings request failed: ${err instanceof Error ? err.message : String(err)}`,
            );
        }

        if (!res.ok) {
            const detail = await res.text().catch(() => "");
            throw new EmbeddingError(
                `embeddings API ${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`,
            );
        }

        let json: OpenAIEmbeddingResponse;
        try {
            json = (await res.json()) as OpenAIEmbeddingResponse;
        } catch (err) {
            throw new EmbeddingError(
                `embeddings response was not JSON: ${err instanceof Error ? err.message : String(err)}`,
            );
        }

        if (!Array.isArray(json.data) || json.data.length !== batch.length) {
            throw new EmbeddingError(
                `embeddings response had ${json.data?.length ?? 0} vectors, expected ${batch.length}`,
            );
        }

        // The API may return results out of order; place each by its `index`.
        const vectors: Float32Array[] = new Array(batch.length);
        for (const item of json.data) {
            if (!Array.isArray(item.embedding)) {
                throw new EmbeddingError("embeddings response contained a non-array embedding");
            }
            vectors[item.index] = normalize(Float32Array.from(item.embedding));
        }
        for (let i = 0; i < vectors.length; i++) {
            if (!vectors[i]) {
                throw new EmbeddingError(`embeddings response missing vector at index ${i}`);
            }
        }
        return vectors;
    }
}
