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

// Storage substrate: shared SQLite helpers, the curated memory store, the
// embedder, the append-only event log, and the tools/recall that bridge memory
// into the loop.
export * from "./sqlite.ts";
export * from "./memory.ts";
export * from "./embeddings.ts";
export * from "./events.ts";
export * from "./memoryTools.ts";
export * from "./eventTools.ts";
export * from "./goals.ts";
export * from "./goalTools.ts";

// The knowledge base: a markdown frontmatter (de)serializer, the notes store
// (a separate corpus linked to memory), the two-way file sync engine, and the
// agent tools that read/write notes.
export * from "./frontmatter.ts";
export * from "./notes.ts";
export * from "./notesFile.ts";
export * from "./notesService.ts";
export * from "./noteTools.ts";

// The local shell: an unguarded tool the loop dispatches to run commands on the
// user's real machine, the counterpart to the sandboxed code_execution server
// tool.
export * from "./shellTools.ts";

// Context engineering: passive context, the pushed working mind, compaction,
// usage accounting.
export * from "./context.ts";
export * from "./workingMind.ts";
export * from "./compaction.ts";
export * from "./usage.ts";

// The slash-command catalogue every client surface (the REPL, the web client)
// advertises; inert data describing the session-level actions a surface runs.
export * from "./commands.ts";

// The thing you talk to, and the ways to drive it without a human in the seat.
export * from "./session.ts";
export * from "./orchestrate.ts";
export * from "./critics.ts";

// Measuring the critic panel: does its verdict stay invariant to nuisance
// variables (roster order, the stakes dealt) the way an unbiased jury's would?
export * from "./biasHarness.ts";

// Dreaming: generating disposable Constructs during downtime.
export * from "./dreaming.ts";

// The interactive runner, for embedding a REPL in your own entry point.
export * from "./repl.ts";
