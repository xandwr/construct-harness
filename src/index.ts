#!/usr/bin/env node
import { AnthropicClient } from "./bridge/anthropic.ts";
import type { ModelClient } from "./bridge/types.ts";
import { MemoryStore } from "./memory.ts";
import { GoalStore } from "./goals.ts";
import { OpenAIEmbedder, EmbeddingError, type Embedder } from "./embeddings.ts";
import { Session } from "./session.ts";
import { shellTools } from "./shellTools.ts";
import { runRepl } from "./repl.ts";

const BASE_SYSTEM =
    "You are a helpful, concise assistant: a long-lived Construct that remembers " +
    "across conversations. Save durable facts and preferences with memory_save, and " +
    "recall them with memory_recall. Don't save transient chatter. When given a task " +
    "worth holding across turns, track it with goal_set and mark it goal_update done " +
    "when achieved; your active goals are shown to you each turn. You can run commands " +
    "on the user's local machine with use__user__shell (their real files, tools, and " +
    "working directory), so reach for it to run tests, inspect or edit real files, and " +
    "drive their CLIs.";

/**
 * Construct the cloud embedder when an OpenAI key is configured, else return
 * undefined so the harness runs with purely lexical recall. A bad key surfaces
 * later, on first use, as a caught {@link EmbeddingError} (recall degrades
 * gracefully): construction only needs the key to be present.
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

/** Compaction threshold (estimated tokens) for an interactive session. Set well
 *  below the model's real context window so there's headroom for the next turn's
 *  output. Overridable via the COMPACT_AT env var. */
const DEFAULT_COMPACT_AT = 120_000;

async function main() {
    if (!process.env.ANTHROPIC_API_KEY) {
        console.log("harness boot: no ANTHROPIC_API_KEY set, skipping live session.");
        console.log("bridge wired:", new AnthropicClient().provider);
        return;
    }

    const client: ModelClient = new AnthropicClient({
        model: process.env.MODEL,
        // Surface retries on stderr so a flaky connection is visible without
        // cluttering the conversation on stdout.
        retry: {
            onRetry: ({ attempt, delayMs, error }) =>
                console.error(
                    `retrying after ${error.kind} (attempt ${attempt}, waiting ${delayMs}ms)`,
                ),
        },
    });
    const dbPath = process.env.MEMORY_DB ?? "db.sqlite";
    const store = new MemoryStore(dbPath);
    const goals = new GoalStore(dbPath);
    const embedder = makeEmbedder();

    try {
        // Embed any memories that don't have a vector yet (e.g. saved before
        // embeddings were wired up, or whose content was later edited) so
        // semantic recall covers the whole store, not just newly-saved rows.
        if (embedder) await backfillEmbeddings(store, embedder);

        const compactAt = Number(process.env.COMPACT_AT) || DEFAULT_COMPACT_AT;
        const session = new Session({
            client,
            system: BASE_SYSTEM,
            store,
            goals,
            embedder,
            // Give the Construct the user's real shell, the local counterpart to
            // the sandboxed code_execution server tool: full, unguarded access to
            // run commands on this machine.
            tools: shellTools(),
            compaction: { thresholdTokens: compactAt },
            providerOptions: { cacheSystem: true },
        });

        await runRepl(session);
    } finally {
        goals.close();
        store.close();
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
