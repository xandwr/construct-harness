# construct-harness

A harness for long-lived agents. Each agent is a **Construct**: something you
talk to that remembers across conversations, streams its replies, calls tools,
and keeps itself under the context window without supervision.

The project is two halves you can use separately:

- **The core library** (`src/`), provider-neutral and dependency-light. It is
  what you import.
- **A web client** (`client/`), a SvelteKit app for talking to a Construct and
  inspecting what it knows. It is optional.

## The opinions

- **The core has one runtime dependency:** the Anthropic SDK, quarantined to a
  single file. Everything else is the Node standard library, `node:sqlite`
  included. No vector database, no build step. (The web client is a normal
  SvelteKit app with its own dependencies; the "one dependency" claim is about
  the library you import, not the whole repo.)
- **The model lives behind a bridge.** Nothing outside
  [`src/bridge/anthropic.ts`](src/bridge/anthropic.ts) knows which model you are
  talking to. The core speaks plain message types; the bridge maps them to a
  provider. Adding or swapping one touches a single file.
- **Tests run without a network.** The mappers are pure, the loop runs against a
  scripted fake client, and the stores run in `:memory:`. The suite is about the
  size of the source.

## What is interesting here

**Memory that degrades gracefully.** Facts live in SQLite with three ways in:
importance and recency, an FTS5 full-text index (porter-stemmed, so "deploys"
finds "deploy"), and per-row float32 embeddings for semantic recall. One recall
walks a ladder, semantic first, then lexical, then substring, then importance.
If the embedding service is down, recall falls back to lexical instead of failing
the turn. Memories also carry a **strength** that rises when they keep
resurfacing and decays when they go untouched, so the store learns which facts
keep proving relevant. See [`src/memory.ts`](src/memory.ts) and
[`src/memoryTools.ts`](src/memoryTools.ts).

**A Construct that doesn't wake up cold.** Recall is pull-based: a memory only
surfaces if the current message embed-matches it. The **working mind**
([`src/workingMind.ts`](src/workingMind.ts)) adds push: a small, evolving set of
the Construct's own recent thoughts and recently-surfaced memories rides every
turn, kept warm a while instead of vanishing the instant the next message
doesn't match. The harness promotes and decays these; it never authors them, so
the held state stays the Construct's own.

**A sense of where it is.** A waking Construct also gets the current time and how
long since the last turn ([`src/context.ts`](src/context.ts)); a
`transcript_recall` tool to search its durable event log, not just the in-context
window ([`src/eventTools.ts`](src/eventTools.ts)); goals it sets and holds across
turns ([`src/goals.ts`](src/goals.ts)); provider-hosted web search, fetch, and
code execution it runs server-side ([`src/bridge/anthropic.ts`](src/bridge/anthropic.ts));
and a `use__user__shell` tool that runs commands on your real machine with the
harness's own privileges, so it can run the tests, edit files, and drive your
CLIs ([`src/shellTools.ts`](src/shellTools.ts)). That shell tool tells the
Construct which shell and OS it is on, so it writes fish, zsh, or bash rather than
guessing. Each of these is opt-in when you wire the `Session`.

**A knowledge base alongside the memory.** Memory holds short, agent-curated
facts auto-injected every turn. **Notes** ([`src/notes.ts`](src/notes.ts)) hold
longer documentation the agent chooses to read, and sync two ways with a folder
of markdown files: each note keeps a stable uuid in its frontmatter, so a row
survives a rename or move on disk in either direction. Same database, separate
corpus, so human docs and memory chatter never bleed together.

**Dreaming.** During downtime a Construct can invent a throwaway `Personality`,
drop it into a scenario abstracted from its own memory corpus, and record what
that persona chooses ([`src/dreaming.ts`](src/dreaming.ts)). The point is not to
remember you better but to explore the decision-space you inhabit with synthetic
agents that cost nothing to discard. Dreams feed back through a `dream_recall`
tool and a last-dream nudge in the system prompt ([`src/dreamTools.ts`](src/dreamTools.ts)).

**An adversarial critic panel with stakes.** This is the reason the project
exists. A reviewer told to "be blunt" still drifts toward the agreeable centroid.
So instead of ordering a critic to be harsh, you give it something to *protect*.
A `Personality` is rendered into a verifier's system prompt so it judges in
character, and each critic is dealt a **stake**, a scene where being wrong has a
cost. A `falsePass` stake dreads waving something broken through and pulls toward
rejection; a `falseFail` stake dreads blocking good work and pulls toward
approval. Deal both across a panel with `dealRoster` and you get a jury arguing a
real tension instead of a monoculture nodding along. See
[`src/critics.ts`](src/critics.ts).

`dealRoster` earns its place. Dealing stakes per persona (each critic draws
independently) only balances the valences *in expectation*. Per run it fails
loudly: with a balanced pool a three-critic panel comes out fully one-sided a
quarter of the time, exactly the stampede the mechanism exists to prevent.
`dealRoster` stratifies the deal so both valences are guaranteed present for any
panel of two or more, while keeping which persona is biased which way random
within each panel.

Two honest notes on that panel:

1. That a jury of diverse, oppositely-biased judges beats one judge, and is
   cheaper and less self-biased, is well supported: see *Replacing Judges with
   Juries* ([arXiv:2404.18796](https://arxiv.org/abs/2404.18796)).
2. The claim that the verdict is *invariant* to things it shouldn't depend on
   (seating order, which stakes get dealt) is measured, not just asserted. The
   within-run stampede is ruled out structurally by `dealRoster`. The harder
   between-run half has a harness ([`src/biasHarness.ts`](src/biasHarness.ts),
   run with `npm run bias`) that runs the live panel over a candidate many times,
   re-seating and re-dealing between trials, and reports how stable and how
   correct the verdict was. An early run is sobering: on code carrying a cardinal
   flaw (`Math.random()` for a reset token) the panel failed it every trial, but
   on genuinely sound code the verdict *wobbled* with the seating and the deal.
   LLM judges are known to carry exactly this order and position bias (*Judging
   the Judges*, [arXiv:2406.07791](https://arxiv.org/abs/2406.07791)). The panel
   mitigates it; it does not erase it. Treat invariance as a property the design
   is built toward and the harness measures, not one it already achieves.

## Requirements

Node 23.6 or newer. The harness runs TypeScript directly using Node's native
type stripping, so there is nothing to compile. Developed on Node 26.

## Quick start (the REPL)

```sh
export ANTHROPIC_API_KEY=sk-...
# optional: turns on semantic recall. Without it, recall is lexical.
export OPENAI_API_KEY=sk-...

npm start            # or: node --env-file-if-exists=.env src/index.ts
```

You get a streaming prompt. The Construct saves durable facts on its own and
recalls them later. Slash commands handle the session-level actions a transcript
can't (`/reset`, `/history`, `/help`, `/exit`); they live in one catalogue
([`src/commands.ts`](src/commands.ts)) that both the REPL and the web client read,
so adding one lights it up on both surfaces.

## The web client

The client is a SvelteKit app that talks to a small HTTP backend. The two run as
separate processes; `just dev` starts both:

```sh
just dev     # API server (node, :8787) + client (vite, :5173)
```

Or run them by hand:

```sh
npm run serve                  # API backend on :8787
cd client && npm run dev       # client on :5173, proxies /api to the server
```

The backend runs on Node, not Bun, because the stores use `node:sqlite`. Beyond
chat, the client gives you a window into what the Construct knows: pages for its
memories, the event log, dreams, goals, the knowledge base, a read-only context
inspector (what the Construct actually sees each turn), and settings. CORS is
permissive (`CORS_ORIGIN=*`) for local development; set an exact origin if you
expose the server.

The HTTP helpers are exported separately from the core:

```ts
import { createHandler } from "construct-harness/server";
```

## Using it as a library

Install it, then import the pieces you want. Importing the package runs nothing;
the REPL only starts when you run the `construct` bin.

```ts
import { Session, MemoryStore, AnthropicClient, OpenAIEmbedder } from "construct-harness";

const client = new AnthropicClient({ model: "claude-opus-4-8" });
const store = new MemoryStore("db.sqlite");
const embedder = process.env.OPENAI_API_KEY ? new OpenAIEmbedder() : undefined;

const session = new Session({
    client,
    system: "You are a helpful assistant that remembers across conversations.",
    store,
    embedder,
    compaction: { thresholdTokens: 120_000 },
    providerOptions: { cacheSystem: true },
});

// send() streams events and returns a TurnResult when it is done.
const run = session.send("Remember that I like short answers, then say hi.");
let next = await run.next();
while (!next.done) {
    if (next.value.kind === "text") process.stdout.write(next.value.text);
    next = await run.next();
}
console.log("\nused", next.value.usage.outputTokens, "output tokens");
```

### A critic panel

The panel is the same primitive whether you grade one answer or wire it as the
verifier inside a larger run.

```ts
import { AnthropicClient, panel, dealRoster, majorityRule } from "construct-harness";
import type { Personality } from "construct-harness";

const client = new AnthropicClient();

const roster: Personality[] = dealRoster([
    {
        name: "Mara",
        role: "staff security engineer",
        disposition: "assumes every input is hostile until proven otherwise",
    },
    {
        name: "Devin",
        role: "ship-it product lead",
        disposition: "protects momentum; rejects perfectionism that is not load-bearing",
    },
    { name: "Sam", role: "the on-call engineer who gets paged at 3am" },
]); // both valences guaranteed present, but which persona gets which is random

const verdict = await panel(roster, { client }, candidateText, { consensus: majorityRule });
console.log(verdict.ok); // the jury's call
for (const v of verdict.verdicts) console.log(v.critic.name, v.verdict?.ok, v.verdict?.rationale);
```

`orchestrate()` ties this together end to end: fan tasks out to fresh Constructs,
verify each result (one skeptic, or a whole panel via `panelVerify`), and keep the
survivors. See [`src/orchestrate.ts`](src/orchestrate.ts).

## How it fits together

The layers go from provider-neutral to provider-specific, and the dependency
arrows only ever point inward.

- [`src/types.ts`](src/types.ts): the core message types. No provider here.
- [`src/bridge/`](src/bridge/): the contract (`ModelClient`), a neutral error
  taxonomy and retry policy, the agentic loop, and `anthropic.ts`, the one file
  allowed to import an SDK.
- [`src/memory.ts`](src/memory.ts), [`src/notes.ts`](src/notes.ts),
  [`src/events.ts`](src/events.ts), [`src/goals.ts`](src/goals.ts): the stores,
  all on one SQLite file behind one migration runner, plus the tools and passive
  recall that bridge them into a run.
- [`src/context.ts`](src/context.ts),
  [`src/workingMind.ts`](src/workingMind.ts),
  [`src/compaction.ts`](src/compaction.ts), [`src/usage.ts`](src/usage.ts):
  per-turn passive context, pushed working memory, summarizing old turns to stay
  under the window, and token accounting.
- [`src/session.ts`](src/session.ts): the stateful thing a person talks to.
- [`src/orchestrate.ts`](src/orchestrate.ts),
  [`src/critics.ts`](src/critics.ts),
  [`src/dreaming.ts`](src/dreaming.ts): driving Constructs without a human in the
  seat, judging their work, and dreaming between turns.

## Testing and dry runs

```sh
npm test         # node --test, no network, no spend
npm run typecheck
```

The scripted client used throughout the suite is exported for your own zero-spend
dry runs. Hand it a queue of turns and drive a real `Session` against it with no
key and no network:

```ts
import { Session } from "construct-harness";
import { FakeClient, textTurn } from "construct-harness/testing";

const client = new FakeClient([textTurn("hello from a scripted turn")]);
const session = new Session({ client, system: "test" });
```

## Status and limitations

Stated plainly, because a harness that hides its approximations is the thing this
project is trying not to be.

- **One provider so far.** Anthropic is implemented. The bridge is built to take
  others, but "provider-neutral" is validated by a single implementation today.
- **Semantic recall scans linearly.** `semanticSearch` compares against every
  stored vector in JavaScript. That is microseconds for a personal store of a few
  thousand memories; a large corpus wants an approximate index, which is a future
  migration, not a rewrite.
- **Argument validation is shallow.** The loop checks a tool call's top-level
  shape, not a full JSON Schema. The tool stays the final authority on its input.
- **The live conversation is in memory; the durable record is in SQLite.** A
  Session's in-process history lives for the life of the process. When an event
  log is wired (the server always does) every turn is appended to SQLite, so it
  outlives the process and the Construct can search it with `transcript_recall`.
  Saved memories, notes, and goals persist the same way; the working mind does
  not, by design.
- **The panel's between-run invariance is measured, and imperfect.** The
  within-run stampede is ruled out structurally; the bias harness (`npm run bias`)
  shows the verdict on sound code still wobbles with seating and deal even as the
  cardinal-flaw case is caught every time. Mitigated, not erased.

## License

ISC. See [LICENSE](LICENSE).
