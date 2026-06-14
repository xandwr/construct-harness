# construct-harness

A small, provider-neutral harness for long-lived agents. Each agent is a
**Construct**: a thing you can talk to that remembers across conversations,
streams its replies, calls tools, and keeps itself under the context window
without you babysitting it.

The whole project leans on a few opinions:

- **One runtime dependency.** Only the Anthropic SDK, and it is quarantined to a
  single file. Everything else is the Node standard library, including
  `node:sqlite` for storage. No vector database, no framework, no build step.
- **A provider-neutral core.** Nothing outside `src/bridge/anthropic.ts` knows
  what model you are talking to. The core speaks plain message types; a thin
  bridge maps them to a provider. Swapping or adding a provider touches one file.
- **Tests that run without a network.** The mappers are pure, the loop runs
  against a scripted fake client, and the memory store runs in `:memory:`. The
  suite is roughly the size of the source.

## What is interesting here

**Hybrid memory that degrades gracefully.** Memories live in SQLite with three
ways in: importance and recency, an FTS5 full-text index (porter-stemmed, so
"deploys" finds "deploy"), and per-row float32 embeddings for semantic recall.
A single recall walks a ladder: semantic match first, then lexical, then
substring, then importance order. If the embedding service is down, recall
quietly falls back to lexical instead of failing a turn. See
[`src/memory.ts`](src/memory.ts) and [`src/memoryTools.ts`](src/memoryTools.ts).

**A Construct that knows where it is.** Beyond memory, a waking Construct gets a
working sense of its situation: the current time _and_ how long since the last
turn and how long the session has run ([`src/context.ts`](src/context.ts)); a
`transcript_recall` tool to search its own durable event log, not just the
in-context window ([`src/eventTools.ts`](src/eventTools.ts)); goals it sets and
holds across turns, injected into every prompt so it doesn't drift from the task
([`src/goals.ts`](src/goals.ts), [`src/goalTools.ts`](src/goalTools.ts));
provider-hosted web search / fetch / code execution it can run server-side
([`src/bridge/anthropic.ts`](src/bridge/anthropic.ts)); and, the unguarded local
counterpart to that sandboxed code execution, a `use__user__shell` tool that runs
commands on the user's _real_ machine, with the harness process's own privileges,
so it can run the project's tests, read or edit actual files, and drive the user's
CLIs ([`src/shellTools.ts`](src/shellTools.ts)). Its reasoning trace streams
through to the UI as a collapsible block. None of this is auto-magic: each is a
tool or a passive context provider you opt into when wiring the `Session`.

**An adversarial critic panel with stakes.** This is the part with no shipped
equivalent we have found, and the reason the project exists. A reviewer agent
that has been told to "be blunt" still drifts toward the average, agreeable
voice. So instead of ordering a critic to be harsh, you give it something to
_protect_. A `Personality` is rendered into a verifier's system prompt so it
judges in character, and each critic is dealt a **stake**: a scene where being
wrong has a cost. Stakes come in two directions. A `falsePass` stake dreads
waving something broken through, so it pulls toward rejection. A `falseFail`
stake dreads blocking good work, so it pulls toward approval. Deal both across a
panel and you get a jury that argues a real tension instead of a monoculture
nodding along. See [`src/critics.ts`](src/critics.ts).

Two honest notes on that panel. First, the idea that a panel of diverse,
oppositely-biased judges beats one judge, and is cheaper and less self-biased,
is well supported: see _Replacing Judges with Juries_
([arXiv:2404.18796](https://arxiv.org/abs/2404.18796)). Second, the specific
claim that randomly dealing stakes _decorrelates_ the panel's errors run to run
is the design intent, not something measured yet. LLM judges are known to carry
order and position biases (_Judging the Judges_,
[arXiv:2406.07791](https://arxiv.org/abs/2406.07791)), and a bias-invariance
test for this panel is on the roadmap. Treat the decorrelation as a hypothesis
the design is built to test, not a result.

## Requirements

Node 23.6 or newer. The harness runs TypeScript directly using Node's native
type stripping, so there is nothing to compile. It is developed on Node 26.

## Quick start (the REPL)

```sh
export ANTHROPIC_API_KEY=sk-...
# optional, turns on semantic recall; without it recall is lexical
export OPENAI_API_KEY=sk-...

npm start            # or: node --env-file-if-exists=.env src/index.ts
```

You get a streaming prompt. The Construct saves durable facts on its own and
recalls them on later turns. Slash commands: `/reset`, `/history`, `/exit`.

## HTTP server and client

Run the local HTTP surface with:

```sh
npm run serve
```

It exposes the single-user `/api/*` backend used by the SvelteKit client. By
default CORS is permissive (`CORS_ORIGIN=*`) for local development and separately
served static builds. Set `CORS_ORIGIN=http://localhost:5173` or another exact
origin if you expose the server outside that local setup.

The HTTP helpers are exported separately from the core package:

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

The panel is the same primitive whether you use it to grade one answer or as the
verifier inside a larger run.

```ts
import { AnthropicClient, panel, dealStakes, majorityRule } from "construct-harness";
import type { Personality } from "construct-harness";

const client = new AnthropicClient();

const roster: Personality[] = [
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
].map((p) => dealStakes(p)); // hand each critic something to protect, drawn at random

const verdict = await panel(roster, { client }, candidateText, { consensus: majorityRule });
console.log(verdict.ok); // the jury's call
for (const v of verdict.verdicts) console.log(v.critic.name, v.verdict?.ok, v.verdict?.rationale);
```

`orchestrate()` ties this together end to end: fan a set of tasks out to fresh
Constructs, verify each result (a single skeptic, or a whole panel via
`panelVerify`), and keep the survivors. See [`src/orchestrate.ts`](src/orchestrate.ts).

## How it fits together

The layers go from provider-neutral to provider-specific, and the dependency
arrows only ever point inward.

- [`src/types.ts`](src/types.ts): the core message types. No provider here.
- [`src/bridge/`](src/bridge/): the contract (`ModelClient`), a neutral error
  taxonomy and retry policy, the agentic loop, and `anthropic.ts`, the one file
  allowed to import an SDK. Adding a second provider means a second file like it.
- [`src/memory.ts`](src/memory.ts), [`src/embeddings.ts`](src/embeddings.ts),
  [`src/memoryTools.ts`](src/memoryTools.ts): storage, vectors, and the tools and
  passive recall that bridge memory into a run. The same SQLite file, one
  migration runner, also holds the append-only event log
  ([`src/events.ts`](src/events.ts), [`src/eventTools.ts`](src/eventTools.ts)) and
  the goal store ([`src/goals.ts`](src/goals.ts),
  [`src/goalTools.ts`](src/goalTools.ts)).
- [`src/context.ts`](src/context.ts), [`src/compaction.ts`](src/compaction.ts),
  [`src/usage.ts`](src/usage.ts): per-turn passive context (the time, elapsed and
  session duration, active goals), summarizing old turns to stay under the window,
  and token accounting. A context provider may be async, so it can read a store to
  decide what to inject.
- [`src/session.ts`](src/session.ts): the stateful thing a person talks to.
- [`src/orchestrate.ts`](src/orchestrate.ts), [`src/critics.ts`](src/critics.ts):
  driving Constructs without a human in the seat, and judging their work.

## Testing and dry runs

```sh
npm test         # node --test, no network, no spend
npm run typecheck
```

The scripted client used throughout the suite is exported for your own
zero-spend dry runs. Hand it a queue of turns and drive a real `Session` or loop
against it with no key and no network:

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
  thousand memories. A large corpus wants an approximate index, which is a future
  migration, not a rewrite.
- **Argument validation is shallow.** The loop checks a tool call's top-level
  shape (object-ness and required keys), not a full JSON Schema. The tool stays
  the final authority on its own input.
- **The live conversation is in memory; the transcript is not.** A Session's
  in-process history lives for the life of the process, but when an event log is
  wired (the server always does) every turn is appended to durable SQLite, so it
  outlives the process and the Construct can search it with `transcript_recall`.
  Saved memories and goals persist the same way.
- **The decorrelation claim is unmeasured.** See the note under the critic panel
  above.

## License

ISC. See [LICENSE](LICENSE).
