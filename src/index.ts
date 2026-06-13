import { RoleType, type Message, type ToolDef } from "./types.ts";
import { AnthropicClient } from "./bridge/anthropic.ts";
import { runLoop } from "./bridge/loop.ts";
import type { ModelClient } from "./bridge/types.ts";
import { MemoryStore } from "./memory.ts";
import { memoryTools, recallContext } from "./memoryTools.ts";
import { OpenAIEmbedder, EmbeddingError, type Embedder } from "./embeddings.ts";
import { temporalContext } from "./context.ts";

const BASE_SYSTEM =
    "You are a terse assistant. Use tools when asked about weather. " +
    "Save durable facts and preferences with memory_save, and recall them with memory_recall.";

/** Concatenate the text parts of a message — the turn's plain-text content. */
function messageText(message: Message): string {
    return message.content
        .filter((p): p is Extract<typeof p, { kind: "text" }> => p.kind === "text")
        .map((p) => p.text)
        .join(" ");
}

/**
 * Build the system turn, folding in any memories worth recalling up front.
 * Ranks recall against `userTurn` so we inject what's relevant to this turn,
 * not just the globally most-important memories.
 */
async function buildSystem(
    store: MemoryStore,
    embedder: Embedder | undefined,
    userTurn?: Message,
): Promise<Message> {
    const recalled = await recallContext(store, {
        query: userTurn ? messageText(userTurn) : undefined,
        embedder,
    });
    const text = recalled ? `${BASE_SYSTEM}\n\n${recalled}` : BASE_SYSTEM;
    return {
        sender: { role: RoleType.System },
        timestamp: Date.now(),
        content: [{ kind: "text", text }],
    };
}

const ask: Message = {
    sender: { role: RoleType.User },
    timestamp: Date.now(),
    content: [
        { kind: "text", text: "What's the weather in Dublin? Then tell me in one sentence." },
    ],
};

const weatherTool: ToolDef = {
    name: "get_weather",
    description: "Get the current weather for a city.",
    parameters: {
        type: "object",
        properties: { city: { type: "string", description: "City name" } },
        required: ["city"],
    },
    async run(args) {
        const { city } = args as { city: string };
        return { city, tempC: 14, conditions: "overcast" };
    },
};

/**
 * Construct the cloud embedder when an OpenAI key is configured, else return
 * undefined so the harness runs with purely lexical recall. A bad key surfaces
 * later, on first use, as a caught {@link EmbeddingError} (recall degrades
 * gracefully) — construction only needs the key to be present.
 */
function makeEmbedder(): Embedder | undefined {
    if (!process.env.OPENAI_API_KEY && !process.env.OPENAI_ADMIN_KEY) return undefined;
    try {
        return new OpenAIEmbedder();
    } catch (err) {
        console.warn(`embeddings disabled: ${err instanceof Error ? err.message : String(err)}`);
        return undefined;
    }
}

/** How many missing embeddings to backfill per run, to bound startup latency
 *  and embedding cost on a large store. */
const BACKFILL_LIMIT = 100;

/**
 * Embed memories that have no vector yet, one batch on startup. Best-effort: an
 * embedding outage logs and leaves those rows lexical-only rather than failing
 * the run. Returns how many were embedded.
 */
async function backfillEmbeddings(store: MemoryStore, embedder: Embedder): Promise<number> {
    const ids = store.idsMissingEmbedding(BACKFILL_LIMIT);
    if (ids.length === 0) return 0;

    const memories = ids.map((id) => store.get(id)).filter((m) => m !== undefined);
    try {
        const vectors = await embedder.embed(memories.map((m) => m.content));
        let n = 0;
        for (let i = 0; i < memories.length; i++) {
            if (store.setEmbedding(memories[i].id, vectors[i])) n++;
        }
        if (n) console.log(`embedded ${n} memor${n === 1 ? "y" : "ies"} for semantic recall`);
        return n;
    } catch (err) {
        if (err instanceof EmbeddingError) {
            console.warn(`backfill skipped: ${err.message}`);
            return 0;
        }
        throw err;
    }
}

async function main() {
    if (!process.env.ANTHROPIC_API_KEY) {
        console.log("harness boot — no ANTHROPIC_API_KEY set, skipping live call.");
        console.log("bridge wired:", new AnthropicClient().provider);
        return;
    }

    const client: ModelClient = new AnthropicClient({ model: process.env.MODEL });
    const store = new MemoryStore(process.env.MEMORY_DB ?? "db.sqlite");
    const embedder = makeEmbedder();

    try {
        // Embed any memories that don't have a vector yet (e.g. saved before
        // embeddings were wired up, or whose content was later edited) so
        // semantic recall covers the whole store, not just newly-saved rows.
        if (embedder) await backfillEmbeddings(store, embedder);

        // Inject what we know that's relevant to this turn, and let the model
        // read/write memory.
        const system = await buildSystem(store, embedder, ask);
        const tools: ToolDef[] = [weatherTool, ...memoryTools(store, embedder)];

        // Run the agentic loop: model → tool → model, all in core types.
        // Passive context (the current date/time in the user's timezone) is
        // recomputed and folded into the system prompt before every turn.
        const { final, turns } = await runLoop(client, {
            messages: [system, ask],
            tools,
            context: [temporalContext()],
        });

        for (const part of final.message.content) {
            if (part.kind === "text") process.stdout.write(part.text);
        }
        console.log(
            `\n[${final.stopReason}] ${turns} turn(s), ${final.usage.outputTokens} out tokens`,
        );
    } finally {
        store.close();
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
