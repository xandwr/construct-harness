/**
 * Public entry point for construct-harness (the package `exports` "." target).
 *
 * Importing this runs nothing: it only re-exports the library surface, so a
 * consumer can pull in exactly the Constructs they need. The interactive REPL
 * lives in `index.ts` (the `bin`), kept separate so importing the library never
 * starts a session. Test helpers live behind the `construct-harness/testing`
 * subpath, not here.
 *
 * Re-exported layer by layer, innermost first:
 */

// Core, provider-neutral types.
export * from "./types.ts";

// The bridge: the contract, the error taxonomy, the retry policy, the agentic
// loop, and the Anthropic implementation (the only provider so far).
export * from "./bridge/types.ts";
export * from "./bridge/errors.ts";
export * from "./bridge/retry.ts";
export * from "./bridge/loop.ts";
export * from "./bridge/anthropic.ts";

// Memory: the store, the embedder, and the tools/recall that bridge them in.
export * from "./memory.ts";
export * from "./embeddings.ts";
export * from "./memoryTools.ts";

// Context engineering: passive context, compaction, usage accounting.
export * from "./context.ts";
export * from "./compaction.ts";
export * from "./usage.ts";

// The thing you talk to, and the ways to drive it without a human in the seat.
export * from "./session.ts";
export * from "./orchestrate.ts";
export * from "./critics.ts";

// The interactive runner, for embedding a REPL in your own entry point.
export * from "./repl.ts";
