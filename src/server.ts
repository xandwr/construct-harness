/**
 * An HTTP surface over the harness: the backend the SvelteKit client talks to.
 *
 * This is the one tier the library deliberately doesn't ship: {@link Session},
 * {@link EventStore}, and {@link MemoryStore} are UI-agnostic, and this module
 * is the thin, dependency-free adapter that exposes them over `/api/*`. It owns
 * no conversation logic of its own; every endpoint is a scoped read over a store
 * or a stream off a {@link Session}, the same way the REPL is a terminal over a
 * Session.
 *
 * Process/session model: a {@link SessionPool} of live Sessions, all sharing one
 * EventStore so every turn persists to disk. The pool starts with one fresh
 * conversation; chatting into a past conversation (a `session` id on `/api/chat`)
 * resumes it from the log into the pool, where it stays live and resumable. So
 * the client's `?session=<id>` deep-link isn't a read-only replay: sending a turn
 * there picks the conversation back up exactly where it left off. Reads
 * (`/api/sessions`, `/api/events`) report which conversations are currently live
 * (loaded in the pool) versus only on disk.
 *
 * The five endpoints, each named by the frontend stub that consumes it:
 *  - `POST /api/chat`            — send a message (optional `session` to resume
 *                                  a past conversation); reply streams as SSE.
 *  - `GET  /api/events?session=` — one conversation's transcript, oldest first.
 *  - `GET  /api/sessions`        — conversation list (id + preview + count).
 *  - `GET  /api/memories`        — the curated memory store (enriched with
 *                                  strength/provenance/embedding); `PUT`/`DELETE`
 *                                  `/api/memories/:id` to edit or forget one.
 *  - `GET  /api/log`            — the raw event log, newest first.
 *  - `GET  /api/dreams`          — the accumulated dreams, newest first.
 *  - `POST /api/dreams`          — run N dreams now (a disposable persona faces a
 *                                  scenario drawn from the corpus); each is logged.
 *  - `GET  /api/goals`           — the goal store, filtered by scope/session/status.
 *  - `POST /api/goals`           — create a goal (global, or scoped to a session).
 *  - `PUT  /api/goals/:id`       — edit a goal's content and/or status.
 *  - `DELETE /api/goals/:id`     — remove a goal.
 *  - `GET  /api/commands`        — the slash-command catalogue, for the client's
 *                                  `/` menu (mirrors {@link BUILTIN_COMMANDS}).
 *  - `GET  /api/status`          — read-only runtime status (model, tools, storage,
 *                                  features), for the settings page. No secrets.
 *  - `GET  /api/context`         — preview the context a turn would be built from
 *                                  for a draft (read-only; mutates nothing).
 *
 * Run it with `npm run serve` (see package.json). It speaks only core types and
 * the stores' public surface, so it stays as provider-neutral as everything
 * under `src/`.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { AnthropicClient, type AnthropicOptions, type ServerToolName } from "./bridge/anthropic.ts";
import { HarnessError, type ErrorKind } from "./bridge/errors.ts";
import type { ModelClient, ProviderOptions } from "./bridge/types.ts";
import { PROVIDERS, providerForModel } from "./bridge/models.ts";
import {
    RuntimeConfig,
    SERVER_TOOL_CATALOG,
    EFFORT_LEVELS,
    type LocalToolGroup,
} from "./runtimeConfig.ts";
import type { ToolDef } from "./types.ts";
import { MemoryStore, Memory, MemoryError } from "./memory.ts";
import { EventStore, MAX_ATTACHMENT_BYTES, type Event, type AttachmentMeta } from "./events.ts";
import { backfillEventEmbeddings } from "./eventTools.ts";
import {
    GoalStore,
    Goal,
    GoalError,
    isGoalStatus,
    GOAL_EVENT_KIND,
    type GoalStatus,
    type GoalChange,
    type GoalEventMeta,
} from "./goals.ts";

// Re-export the goal event kind from its home in goals.ts so existing importers
// of `server.ts` (and the test suite) keep resolving it here unchanged.
export { GOAL_EVENT_KIND };
import { OpenAIEmbedder, EmbeddingError, type Embedder } from "./embeddings.ts";
import { Session, type SessionConfig, type ImageInput } from "./session.ts";
import type { LoopEvent } from "./bridge/loop.ts";
import { NotesStore, Note, NoteError, type NoteFrontmatter } from "./notes.ts";
import { NotesService } from "./notesService.ts";
import { noteTools } from "./noteTools.ts";
import { shellTools, resolveShellPolicy, type ShellPolicyMode } from "./shellTools.ts";
import { BUILTIN_COMMANDS } from "./commands.ts";
import { dreamLoop, DREAM_EVENT_KIND, type Dream } from "./dreaming.ts";
import { SYSTEM_PROMPT } from "./systemPrompt.ts";
import { UserPresence, type Override } from "./presence.ts";
import { DowntimeDaemon } from "./downtimeDaemon.ts";
import { gardenTools } from "./memoryGarden.ts";

// The base system prompt lives in SYSTEM.md at the repo root (see
// systemPrompt.ts), not inline here, so the Construct's persona is a markdown
// file a human reads and edits. The server wires in the full tool set, so it
// loads the full prompt (memory, goals, transcript, dreams, the knowledge base,
// and both code paths).
const BASE_SYSTEM = SYSTEM_PROMPT;

/** Compaction threshold (estimated tokens), well below the model's real window
 *  so there's headroom for the next turn. Overridable via COMPACT_AT. */
const DEFAULT_COMPACT_AT = 120_000;

/** Default page size for the conversation and log reads, overridable per query. */
const DEFAULT_PAGE = 100;

/** What the server holds for its lifetime: the stores it reads, the pool of live
 *  Sessions it drives, the knowledge-base service, and a close() that releases the
 *  database handles. `notes` is optional so the server still runs without a KB
 *  (the routes 503 cleanly), but the default wiring always provides one. */
interface ServerDeps {
    /** The model client every model-driven route runs on. Chat drives it through
     *  the {@link SessionPool}; dreaming ({@link handleDreams}) drives it directly,
     *  outside any conversation, to conjure personas and abstract scenarios. Held
     *  on the deps (not only captured in the session config) so a route that isn't
     *  a chat can still reach the provider. */
    client: ModelClient;
    store: MemoryStore;
    events: EventStore;
    goals: GoalStore;
    /** The live conversations this process holds in memory, keyed by session id.
     *  Each one accepts new turns; a conversation only in the log (not here) is a
     *  past one waiting to be resumed into the pool on its next turn. */
    sessions: SessionPool;
    /** The chat turns currently streaming, each behind an AbortController the
     *  `/api/chat/cancel` route fires so the human can stop a reply mid-stream and
     *  the server still saves what was produced. See {@link InflightTurns}. */
    inflight: InflightTurns;
    /** The human's presence (Online/Away/DND/Offline), computed from when they
     *  last sent a message plus any manual override they pinned. Touched on every
     *  user turn (the chat route), read by `/api/presence`, written by a PUT to
     *  the same. In-process for the process's life; see {@link UserPresence}. */
    presence: UserPresence;
    /** The downtime daemon: watches {@link presence} and, once the human has been
     *  genuinely away a while, puts the silence to use — dreaming, mining the
     *  Construct's recurring concerns, gardening the memory store. Started in
     *  {@link buildDeps} and stopped on {@link close}. See {@link DowntimeDaemon}. */
    daemon: DowntimeDaemon;
    notes?: NotesService;
    notesStore?: NotesStore;
    /** The live runtime knobs the settings page reads and writes: the model in
     *  use, which server/local tools are on, the effort level. Optional so a test
     *  can omit it (the settings routes then 503); the default wiring provides
     *  one. Reads off it report the *current* runtime, not the boot snapshot. */
    runtime?: RuntimeConfig;
    corsOrigin?: string;
    /** A snapshot of the *static* runtime configuration this process booted with,
     *  for the read-only status endpoint (see {@link handleStatus}). The dynamic
     *  parts (live session ids, store counts, schema version) are read off the
     *  live deps at request time, not frozen here; this carries only what's fixed
     *  at boot (model, db paths, thresholds, which features are on). Optional so a
     *  test can omit it; the route then reports just the dynamic parts. */
    status?: StatusConfig;
    close(): void;
}

/**
 * The genuinely static slice of runtime configuration the status endpoint
 * reports: what this process booted with that doesn't change while it runs —
 * where data lives, the compaction threshold, whether embedding is configured,
 * the standing features, the shell policy. Captured once in {@link buildDeps},
 * so the handler stays a pure read and a test can supply its own snapshot.
 *
 * The *mutable* knobs the settings page turns — the model, the server tools, the
 * local tools, the effort level — deliberately live on {@link RuntimeConfig}, not
 * here, so the status route reads them live and a switch shows up immediately.
 * Deliberately carries no secrets — the embedder is reported as a yes/no, never
 * the key behind it.
 */
export interface StatusConfig {
    /** The sqlite file every store shares (MEMORY_DB). */
    memoryDb: string;
    /** The knowledge-base markdown folder (KB_DIR), or null when no KB is wired. */
    kbDir: string | null;
    /** The compaction threshold in estimated tokens (COMPACT_AT). */
    compactAt: number;
    /** Whether an embedder is configured (semantic recall on), reported as a flag
     *  so no key ever leaves the process. */
    embeddingConfigured: boolean;
    /** Standing feature flags the Sessions run with, so the status page reflects
     *  what a turn actually does rather than a hardcoded guess. */
    dreamsEnabled: boolean;
    transcriptRecall: boolean;
    workingMind: boolean;
    /** How the local shell is governed: the policy mode and the cwd roots it's
     *  confined to (if any), so the status page shows whether `use__user__shell`
     *  is unrestricted, restricted, or read-only. */
    shellPolicy: { mode: ShellPolicyMode; allowedCwdRoots: string[] };
}

/** Builds the shared {@link SessionConfig} every live conversation runs under,
 *  carrying the `events` log so {@link Session.resume} has a transcript to
 *  rehydrate from. It deliberately leaves `sessionId` unset (a fresh
 *  conversation gets a fresh id); the {@link SessionPool} pins the id when
 *  resuming a specific conversation. {@link buildDeps} supplies the closure with
 *  everything wired in. */
export type SessionConfigBase = () => SessionConfig & { events: EventStore };

/**
 * The set of live Sessions a server process drives, keyed by session id.
 *
 * The harness keeps every conversation in one shared log, but a {@link Session}
 * (the in-memory thing that holds working context and accepts turns) is more
 * expensive: it carries the rehydrated history and pending embeds. So we keep a
 * pool, not one-Session-per-process and not one-per-conversation-ever: a
 * conversation becomes live the first time it's chatted into ({@link resolve}
 * resumes it from the log), stays live for the process's life, and any
 * conversation in the log can be resumed this way. {@link has} lets a read
 * endpoint report which conversations are currently live (in the pool) versus
 * merely on disk.
 */
export class SessionPool {
    private readonly sessions = new Map<string, Session>();
    /** Builds the shared config every conversation runs under (see
     *  {@link SessionConfigBase}); the pool pins the id for a resume. */
    private readonly config: SessionConfigBase;
    /** The conversation a chat with no session id lands on, so a client that
     *  never names one keeps talking to the same conversation across turns. */
    private readonly defaultId: string;
    /** Supplies the concern candidates a freshly-live Session is seeded with —
     *  the topics the Construct keeps raising unprompted, mined during downtime
     *  (the daemon's {@link DowntimeDaemon.concerns}). Optional: with none, a
     *  Session simply starts with an empty concern band. The pool calls it each
     *  time a conversation becomes live, so a Construct picking a conversation back
     *  up wakes already holding what it keeps returning to. */
    private readonly seedConcerns?: () => readonly string[];

    /** Register one brand-new live conversation at boot via {@link config}, so
     *  the server always has an addressable live conversation even before anyone
     *  resumes a past one; its id becomes the default. `seedConcerns`, when given,
     *  seeds each conversation's concern band as it becomes live. */
    constructor(config: SessionConfigBase, seedConcerns?: () => readonly string[]) {
        this.config = config;
        this.seedConcerns = seedConcerns;
        const initial = new Session(config());
        this.seed(initial);
        this.sessions.set(initial.id, initial);
        this.defaultId = initial.id;
    }

    /** Seed a newly-live Session's concern band from the provider, if one is
     *  wired. Best-effort: a provider hiccup must not block bringing a
     *  conversation live. */
    private seed(session: Session): void {
        if (!this.seedConcerns) return;
        try {
            session.seedConcerns(this.seedConcerns());
        } catch {
            // A concern seed is an enhancement, never a gate on the conversation.
        }
    }

    /** Whether a conversation is currently live (has an in-memory Session in the
     *  pool), as opposed to only existing in the log. */
    has(id: string): boolean {
        return this.sessions.has(id);
    }

    /** The ids of every live conversation in the pool. */
    ids(): string[] {
        return [...this.sessions.keys()];
    }

    /**
     * Get the live Session for `id`, resuming it from the log into the pool if it
     * isn't live yet. With no `id` the caller wants "a live conversation to talk
     * to": return the default conversation so a client that never sends a session
     * id keeps landing on the same one across turns.
     *
     * Resuming rehydrates the conversation's prior turns into the new Session's
     * working context (see {@link Session.resume}), so the first turn after a
     * resume builds on the real exchange, not an empty history. A brand-new id
     * (one with no events in the log) resumes to an empty history, which is the
     * correct behavior: it just becomes a fresh live conversation under that id.
     */
    async resolve(id: string | undefined): Promise<Session> {
        if (!id) return this.sessions.get(this.defaultId)!;
        const live = this.sessions.get(id);
        if (live) return live;
        const resumed = await Session.resume({ ...this.config(), sessionId: id });
        // Seed its concern band before it takes its first turn, so the resumed
        // conversation wakes holding what the Construct keeps returning to.
        this.seed(resumed);
        this.sessions.set(resumed.id, resumed);
        return resumed;
    }

    /**
     * Get a Session for `id` for a *read-only* purpose (the context inspector),
     * without registering a new live conversation in the pool. If the conversation
     * is already live, return it (so the preview reflects its real in-memory
     * state — its working mind, its rehydrated history). Otherwise resume a
     * *transient* Session from the log and return it WITHOUT adding it to the pool:
     * inspecting a past conversation must not silently bring it live, the way
     * sending a turn (via {@link resolve}) deliberately does. With no `id`, preview
     * the default conversation.
     */
    async peek(id: string | undefined): Promise<Session> {
        if (!id) return this.sessions.get(this.defaultId)!;
        const live = this.sessions.get(id);
        if (live) return live;
        // Transient: resumed for this read only, never pooled.
        return Session.resume({ ...this.config(), sessionId: id });
    }
}

/**
 * The set of chat turns currently streaming, keyed by session id, each behind an
 * {@link AbortController} the cancel route can fire.
 *
 * A streamed reply has no request the client can simply drop to stop cleanly:
 * the connection dropping is racy and wouldn't give the server a chance to *save*
 * the partial reply. So cancellation is an explicit, addressable action — the
 * client POSTs `/api/chat/cancel` with the session id, which {@link abort}s the
 * controller this registry holds for that turn. Aborting routes through the
 * bridge's cancel path (a `canceled` HarnessError), so the Session keeps the
 * prose already streamed and logs the cancellation, rather than the turn just
 * vanishing.
 *
 * One turn per session at a time: a session streams a single reply, so the latest
 * `begin` for an id replaces any stale entry. Entries are removed when the turn
 * settles (see the chat route's `finally`), so the map only ever holds genuinely
 * in-flight turns.
 */
export class InflightTurns {
    private readonly controllers = new Map<string, AbortController>();

    /** Register a starting turn for `session` and return its {@link AbortSignal}
     *  to thread into {@link streamChat}. Replaces any prior controller for the
     *  same session (a session has at most one streaming reply at a time). */
    begin(session: string): AbortSignal {
        const controller = new AbortController();
        this.controllers.set(session, controller);
        return controller.signal;
    }

    /** Abort the in-flight turn for `session`, if one is streaming. Returns whether
     *  a turn was actually aborted (false when nothing is in flight for that id, so
     *  the cancel route can 404 a stale/duplicate request honestly). The entry is
     *  left for the turn's own `finally` to remove, so a double-cancel is a no-op. */
    cancel(session: string): boolean {
        const controller = this.controllers.get(session);
        if (!controller) return false;
        controller.abort();
        return true;
    }

    /** Drop the registry entry for a settled turn. Idempotent; called from the
     *  chat route's `finally` whether the turn finished, errored, or was cancelled. */
    end(session: string): void {
        this.controllers.delete(session);
    }

    /** Whether a turn is currently streaming for `session`. */
    has(session: string): boolean {
        return this.controllers.has(session);
    }
}

/** Where the knowledge-base markdown folder lives, mirroring MEMORY_DB. */
const DEFAULT_KB_DIR = "kb";

/** A local (harness-owned) tool group plus the factory that builds its ToolDefs.
 *  {@link buildDeps} defines one per toggleable group; the session config rebuilds
 *  the enabled subset per Session, and the runtime config is told the resolved
 *  tool names (the build's output) for the settings list. The build is a thunk so
 *  toggling never rewires a live Session — only the next Session's tool list. */
interface LocalToolGroupWiring extends Omit<LocalToolGroup, "toolNames"> {
    build: () => ToolDef[];
}

/** The provider-hosted tools enabled by default: live web access so the Construct
 *  can answer about the current world. Code execution is opt-in (it spins up a
 *  sandbox and bills accordingly), so it's left out of the default set. */
const DEFAULT_SERVER_TOOLS: ServerToolName[] = ["web_search", "web_fetch"];

/** The full set a caller may name in SERVER_TOOLS, for validation. */
const KNOWN_SERVER_TOOLS: ServerToolName[] = ["web_search", "web_fetch", "code_execution"];

/**
 * Resolve which provider-hosted tools to enable from the `SERVER_TOOLS` env var.
 * Unset uses {@link DEFAULT_SERVER_TOOLS} (web access on). A comma-separated list
 * picks an explicit set (e.g. `web_search,code_execution`); `none` (or an empty
 * value) disables them entirely. Unknown names are dropped with a warning so a
 * typo degrades to fewer tools rather than a crash.
 */
export function resolveServerTools(raw: string | undefined): ServerToolName[] {
    if (raw === undefined) return DEFAULT_SERVER_TOOLS;
    const trimmed = raw.trim().toLowerCase();
    if (trimmed === "" || trimmed === "none") return [];
    const out: ServerToolName[] = [];
    for (const part of trimmed
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)) {
        if ((KNOWN_SERVER_TOOLS as string[]).includes(part)) {
            out.push(part as ServerToolName);
        } else {
            console.warn(`SERVER_TOOLS: ignoring unknown tool "${part}"`);
        }
    }
    return out;
}

/**
 * Construct the cloud embedder when an OpenAI key is configured, else undefined
 * so recall stays purely lexical. Mirrors index.ts: a bad key surfaces later as
 * a caught {@link EmbeddingError}, so construction only needs the key present.
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

/**
 * Wire up the stores and the single live Session this server drives. The
 * EventStore and MemoryStore share one database file (one schema, one migration
 * runner), so opening either brings the whole schema current; we open both and
 * hand them to a Session, which then logs every turn it runs into the events
 * table. That wiring is what makes chats persist to disk: without the EventStore
 * a Session is purely in-memory and `/api/sessions` would have nothing to read.
 */
function buildDeps(): ServerDeps {
    const client: ModelClient = new AnthropicClient({
        model: process.env.MODEL,
        retry: {
            onRetry: ({ attempt, delayMs, error }) =>
                console.error(
                    `retrying after ${error.kind} (attempt ${attempt}, waiting ${delayMs}ms)`,
                ),
        },
    });

    const dbPath = process.env.MEMORY_DB ?? "db.sqlite";
    const store = new MemoryStore(dbPath);
    const events = new EventStore(dbPath);
    // Log each goal write into the shared event log, so adding or deleting a goal
    // leaves a trace the way a message or a dream does. One sink on the single
    // shared store covers both write paths: the agent's goal tools and the
    // human-editable /api/goals routes.
    const goals = new GoalStore({ location: dbPath, onChange: goalEventSink(events) });
    const embedder = makeEmbedder();
    const compactAt = Number(process.env.COMPACT_AT) || DEFAULT_COMPACT_AT;
    const serverTools = resolveServerTools(process.env.SERVER_TOOLS);

    // The knowledge base shares the same database file (one schema, one migration
    // runner) and a markdown folder on disk. The NotesService starts its watcher
    // asynchronously: an initial scan adopts any files created while the process
    // was down, then live two-way sync begins. A failure to start the watcher is
    // logged but non-fatal: the in-app KB still works over the API.
    const kbDir = process.env.KB_DIR ?? DEFAULT_KB_DIR;
    const notesStore = new NotesStore(dbPath);
    const notes = new NotesService({ store: notesStore, root: kbDir, embedder });

    // The governance the local shell runs under, resolved from the environment.
    // Unset is `unrestricted` (the historical behavior): the policy only tightens
    // when an operator opts in via SHELL_POLICY / SHELL_ALLOWED_ROOTS / the caps.
    const shellPolicy = resolveShellPolicy(process.env);

    // The local (harness-owned) tools, grouped so the runtime config can toggle
    // each group on or off for newly built Sessions. Each group carries a factory
    // that builds its ToolDefs on demand, so the session config closure rebuilds
    // exactly the enabled subset per Session — the same wiring the settings page
    // turns into a list of switches. The KB note tools are one group; the local
    // shell (under the resolved policy) is another.
    const localGroups: LocalToolGroupWiring[] = [
        {
            key: "notes",
            label: "knowledge base",
            note: "save / recall / link markdown notes",
            build: () => noteTools(notes, notesStore, embedder),
        },
        {
            key: "shell",
            label: "local shell",
            note: "run commands on the user's real machine",
            build: () => shellTools({ policy: shellPolicy }),
        },
    ];
    // Both groups are on at boot — the historical wiring. The runtime config can
    // disable one; that takes effect for conversations started after the toggle.
    const enabledLocalKeys = localGroups.map((g) => g.key);

    // The single ProviderOptions object every Session in the pool shares. The
    // runtime config mutates THIS object in place (server tools, effort), so a
    // change reaches live conversations: a Session reads `cfg.providerOptions`
    // each turn off this same reference. Cache the system prefix; turn on adaptive
    // thinking with a summarized display so the streaming path emits readable
    // `thinking` deltas the client can show (see streamChat); start with the
    // resolved server tools (provider-hosted web access the model runs in-turn).
    const sharedProviderOptions: ProviderOptions & AnthropicOptions = {
        cacheSystem: true,
        thinking: true,
        thinkingDisplay: true,
        serverTools,
    };

    // The live knobs the settings page turns. It owns model switching (through
    // the client's setter), the shared provider options above, and which local
    // tool groups new Sessions wire in. Built before the session config so the
    // closure can read its enablement set per Session.
    const runtime = new RuntimeConfig(
        client,
        sharedProviderOptions,
        localGroups.map(({ key, label, note, build }) => ({
            key,
            label,
            note,
            toolNames: build().map((t) => t.name),
        })),
        enabledLocalKeys,
    );

    // One wiring shared by every live conversation. Every Session in the pool —
    // the boot one and every resumed past conversation — runs under this exact
    // configuration; the pool pins the `sessionId` when resuming a specific
    // conversation, leaving it fresh here. Keeping it in one closure is what
    // guarantees a resumed conversation behaves identically to the one it's
    // continuing: same tools, same compaction, same provider options.
    const sessionConfig: SessionConfigBase = () => ({
        client,
        system: BASE_SYSTEM,
        store,
        events,
        goals,
        embedder,
        // The agent opts into the KB and the local shell, each gated by the
        // runtime config: a group the settings page disabled isn't wired into a
        // Session built after the toggle. The note tools (save/update/recall/link)
        // are not auto-injected into context the way memories are; the shell
        // (use__user__shell) is the unguarded local counterpart to the sandboxed
        // code_execution server tool enabled via providerOptions.serverTools.
        // garden_review is always on: it's the read side of memory gardening (the
        // downtime daemon flags consolidation candidates; this lets the Construct
        // review and act on them with the memory tools it already has).
        tools: [
            ...localGroups.filter((g) => runtime.isLocalEnabled(g.key)).flatMap((g) => g.build()),
            ...gardenTools(events),
        ],
        compaction: { thresholdTokens: compactAt },
        // The shared, mutable provider options (see above): live edits land here.
        providerOptions: sharedProviderOptions,
    });

    // The human's presence. Seed last-activity with boot time so a freshly
    // launched daemon reads Online rather than "never seen": today the client and
    // daemon launch together, so a booted process means a present human. The first
    // user message re-touches it; 15 minutes of silence flips it to Away. Built
    // before the daemon (which gates on it) and the pool (which seeds from the
    // daemon's mined concerns).
    const presence = new UserPresence(Date.now());

    // The downtime daemon: bridges the human stepping away to the Construct using
    // the silence. Constructed before the pool so the pool can seed each live
    // conversation's concern band from the daemon's mined candidates. Started
    // below, once the deps are assembled; stopped on close().
    const daemon = new DowntimeDaemon({
        presence,
        events,
        store,
        client,
        embedder,
        onDream: (d) =>
            console.log(`  dreamed during downtime as ${d.persona.name ?? "(unknown)"}`),
        onGarden: (pairs) =>
            console.log(`  flagged ${pairs.length} memory-consolidation candidate(s)`),
    });

    // The pool seeds each newly-live conversation's concern band from the daemon's
    // latest mined concerns, so a Construct picking a conversation back up wakes
    // holding what it keeps returning to (empty until the first downtime mining).
    const sessions = new SessionPool(sessionConfig, () => daemon.concerns());
    // Tracks streaming turns so the cancel route can abort one by session id.
    const inflight = new InflightTurns();

    notes
        .start()
        .then(() => console.log(`  knowledge base watching: ${notes.kbRoot}`))
        .catch((err) =>
            console.warn(
                `knowledge-base sync disabled: ${err instanceof Error ? err.message : String(err)}`,
            ),
        );

    // The Session embeds each new message turn as it's logged, but a log that
    // predates this (turns recorded before an embedder was wired up) is
    // lexical-only until backfilled. Run one bounded catch-up pass at startup so
    // semantic transcript_recall covers the existing transcript too. Fire-and-
    // forget like the watcher above: an embedding outage just leaves those rows
    // lexical, and a slow embed must not block the server accepting requests.
    if (embedder) {
        backfillEventEmbeddings(events, embedder)
            .then((n) => {
                if (n) console.log(`  embedded ${n} past event${n === 1 ? "" : "s"} for recall`);
            })
            .catch((err) =>
                console.warn(
                    `event-embedding backfill skipped: ${err instanceof Error ? err.message : String(err)}`,
                ),
            );
    }

    // Arm the downtime daemon: from here it watches presence and, once the human
    // has been away long enough, dreams / mines concerns / gardens memory. Its
    // interval is unref'd, so it never on its own keeps the process alive.
    daemon.start();

    return {
        client,
        store,
        events,
        goals,
        sessions,
        inflight,
        presence,
        daemon,
        notes,
        notesStore,
        runtime,
        corsOrigin: process.env.CORS_ORIGIN,
        // The static config snapshot for /api/status: the fields that genuinely
        // don't change while the process runs — where the data lives, the
        // compaction threshold, whether embedding is configured, the standing
        // features, the shell policy. The *mutable* fields (model, server tools,
        // local tools, effort) are NOT frozen here; the status route reads them
        // live off `runtime` so the page reflects a switch the moment it lands.
        status: {
            memoryDb: dbPath,
            kbDir,
            compactAt,
            embeddingConfigured: embedder !== undefined,
            // These mirror the Session defaults the config above leaves on: dreams
            // and transcript recall default true when an events log is present
            // (it always is here), and the working mind is on unless disabled.
            dreamsEnabled: true,
            transcriptRecall: true,
            workingMind: true,
            shellPolicy: {
                mode: shellPolicy.mode ?? "unrestricted",
                allowedCwdRoots: shellPolicy.allowedCwdRoots ?? [],
            },
        },
        close() {
            // Stop the downtime daemon first so no dream/garden tick races the
            // store closing. A dream already in flight is left to finish (the
            // daemon doesn't interrupt it); stop() only disarms the next tick.
            daemon.stop();
            // Stop the watcher next so no inbound event races the store closing.
            notes.close();
            // Close the handles. All point at the same file; closing checkpoints
            // the WAL, so order only affects which one truncates it. Either order
            // is correct; this is deterministic.
            events.close();
            goals.close();
            notesStore.close();
            store.close();
        },
    };
}

// ── HTTP plumbing ───────────────────────────────────────────────────────────

/** Map a harness error kind to the closest HTTP status, so the client can tell
 *  a bad key (401) from a rate-limit (429) from a transient upstream blip (502)
 *  without parsing prose. Anything unclassified is a 500. */
function statusForKind(kind: ErrorKind): number {
    switch (kind) {
        case "auth":
            return 401;
        case "invalid_request":
            return 400;
        case "rate_limit":
            return 429;
        case "overloaded":
        case "server":
            return 502;
        case "network":
        case "timeout":
            return 504;
        case "canceled":
            return 499; // nginx's "client closed request"; fitting for an abort.
        default:
            return 500;
    }
}

/** Permissive CORS, so the client works whether it reaches us same-origin (via
 *  the Vite dev proxy) or cross-origin (a separately-served static build). The
 *  harness is a local, single-user tool; there's no credentialed session to
 *  protect with a strict origin allow-list. */
function cors(res: ServerResponse, origin = "*"): void {
    res.setHeader("Access-Control-Allow-Origin", origin || "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    if (origin && origin !== "*") res.setHeader("Vary", "Origin");
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
    const text = JSON.stringify(body);
    res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
    res.end(text);
}

/** Body-size ceiling for the chat endpoint. A single photo's base64 dwarfs the
 *  default 256 KB; this bounds a whole multi-image turn while {@link
 *  MAX_ATTACHMENT_BYTES} still caps each image. base64 inflates bytes ~1.37×, so
 *  this leaves headroom for a few images plus the text and JSON framing. */
const CHAT_BODY_LIMIT = 32 * 1024 * 1024;

/** Thrown when an inbound image fails validation (bad type, oversized, or
 *  undecodable base64). Distinguished from a generic malformed-body error so the
 *  /api/chat handler can return a specific 400 message. */
class ImageParseError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "ImageParseError";
    }
}

/** Parse and validate the `images` field of a chat body into {@link ImageInput}s.
 *  Each wire image is `{ mediaType, data (base64), filename? }`; the base64 is
 *  decoded to raw bytes here (the store keeps bytes, not base64). Throws {@link
 *  ImageParseError} on anything malformed. `undefined`/absent means no images. */
function parseImages(raw: unknown): ImageInput[] {
    if (raw === undefined || raw === null) return [];
    if (!Array.isArray(raw)) throw new ImageParseError("images must be an array");
    const out: ImageInput[] = [];
    for (const item of raw) {
        if (item === null || typeof item !== "object") {
            throw new ImageParseError("each image must be an object");
        }
        const { mediaType, data, filename } = item as Record<string, unknown>;
        if (mediaType !== "image/jpeg" && mediaType !== "image/png") {
            throw new ImageParseError("image mediaType must be 'image/jpeg' or 'image/png'");
        }
        if (typeof data !== "string" || data === "") {
            throw new ImageParseError("image data must be a non-empty base64 string");
        }
        // Buffer.from is lenient (it skips invalid chars), so reject anything that
        // isn't well-formed base64 up front rather than silently storing garbage.
        if (!/^[A-Za-z0-9+/]+={0,2}$/.test(data)) {
            throw new ImageParseError("image data must be valid base64");
        }
        const bytes = Buffer.from(data, "base64");
        if (bytes.length === 0) throw new ImageParseError("image data decoded to empty");
        if (bytes.length > MAX_ATTACHMENT_BYTES) {
            throw new ImageParseError(`image exceeds ${MAX_ATTACHMENT_BYTES} bytes`);
        }
        out.push({
            mediaType,
            data: bytes,
            filename: typeof filename === "string" && filename ? filename : undefined,
        });
    }
    return out;
}

/** Read a request body to a string, bounded so a runaway upload can't exhaust
 *  memory. Rejects past the cap rather than truncating silently. */
function readBody(req: IncomingMessage, limit = 256 * 1024): Promise<string> {
    return new Promise((resolve, reject) => {
        let size = 0;
        const chunks: Buffer[] = [];
        req.on("data", (chunk: Buffer) => {
            size += chunk.length;
            if (size > limit) {
                reject(new Error("request body too large"));
                req.destroy();
                return;
            }
            chunks.push(chunk);
        });
        req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
        req.on("error", reject);
    });
}

// ── Read views over the log ─────────────────────────────────────────────────

/** Serialize an event for the wire: a stable JSON shape the client renders. The
 *  store's `Event` is already plain data; this just fixes the field set.
 *
 *  `attachments` (when given) is the list of image refs the event carries —
 *  metadata only, each a `{ id, mediaType, filename }` the client turns into a
 *  `/api/attachments/<id>` URL. The bytes never ride in this JSON. */
function eventToJson(e: Event, attachments: AttachmentMeta[] = []) {
    return {
        id: e.id,
        ts: e.ts,
        kind: e.kind,
        role: e.role ?? null,
        content: e.content,
        meta: e.meta ?? null,
        session: e.session ?? null,
        correlation: e.correlation ?? null,
        attachments: attachments.map((a) => ({
            id: a.id,
            mediaType: a.mediaType,
            filename: a.filename ?? null,
        })),
    };
}

/**
 * Serialize a `dream` event into the dreams-applet wire shape.
 *
 * A dream is logged as one event ({@link dreamOnce}): its `content` is the
 * persona's choice (verbatim), and its `meta` carries the structured record the
 * loop wrote — `{ persona, scenario, sourceMemoryIds }`. The dreams view wants
 * that structure flattened into named fields rather than the raw event, so the
 * applet renders persona/scenario/choice directly without re-deriving them.
 *
 * `meta` is read defensively: the EventStore degrades a corrupt `meta` to
 * `undefined` on read, and even a well-formed event might (in principle) carry a
 * shape we don't expect, so every field falls back to a safe default rather than
 * throwing. The persona is passed through as-is (it's a {@link Personality}, with
 * optional dealt stakes); the applet reads its `name`/`role` and ignores the
 * rest.
 */
function dreamEventToJson(e: Event) {
    const meta = (e.meta ?? {}) as {
        persona?: unknown;
        scenario?: unknown;
        sourceMemoryIds?: unknown;
    };
    const persona =
        meta.persona && typeof meta.persona === "object" ? meta.persona : { name: "(unknown)" };
    const sources = Array.isArray(meta.sourceMemoryIds)
        ? meta.sourceMemoryIds.filter((x): x is number => typeof x === "number")
        : [];
    return {
        id: e.id,
        ts: e.ts,
        persona,
        scenario: typeof meta.scenario === "string" ? meta.scenario : "",
        // The persona's choice rides in the event content (FTS-searchable there).
        choice: e.content,
        sourceMemoryIds: sources,
    };
}

/**
 * Serialize a memory for the wire, enriched with the curation signals the
 * Memory page surfaces: its effective (decayed) strength right now, when it last
 * surfaced, its provenance (the event it was curated from, and that event's
 * session), and whether it carries an embedding. The provenance event is looked
 * up over the shared {@link EventStore} so the page can offer a jump to the
 * source conversation. `now` is threaded so every memory in one list reads its
 * strength against the same instant.
 */
function memoryToJson(m: Memory, deps: Pick<ServerDeps, "store" | "events">, now = Date.now()) {
    const eventId = deps.store.provenanceOf(m.id);
    // Resolve the provenance event's session (for the deep-link) when the event
    // still exists in the log. A nulled/missing link leaves both null.
    const sourceEvent = eventId !== undefined ? deps.events.get(eventId) : undefined;
    return {
        id: m.id,
        content: m.content,
        tags: m.tags,
        importance: m.importance ?? null,
        created: m.created,
        updated: m.updated,
        // The decayed strength ranking actually uses, not the raw stored number,
        // so the page shows what the Construct effectively feels about this memory.
        strength: deps.store.strengthOf(m.id, now) ?? m.strength,
        lastSurfaced: m.lastSurfaced ?? null,
        provenance:
            eventId === undefined
                ? null
                : {
                      eventId,
                      session: sourceEvent?.session ?? null,
                  },
        hasEmbedding: deps.store.hasEmbedding(m.id),
    };
}

/**
 * Build the {@link GoalEventSink} that records a goal write into the event log,
 * so adding or deleting a goal leaves the same kind of trace a message or a
 * dream does. Wired into the single shared {@link GoalStore}, it covers *both*
 * write paths at once: the agent's `goal_set`/`goal_update` tools and the
 * human-editable `/api/goals` routes run through that one store.
 *
 * The append is best-effort by the store's contract (a throwing sink is
 * swallowed there), but we also guard here: an event-log hiccup must never
 * surface as a failed goal write. The goal's `session` rides onto the event so a
 * session-scoped goal's change shows up in that conversation's transcript, while
 * a shared (global) goal's change stays unscoped, visible in the cross-session
 * view the way the goal itself is.
 */
export function goalEventSink(
    events: EventStore,
): (change: GoalChange, goal: Goal, now: number) => void {
    // Past tense for the human-readable content line; the structured `change`
    // lives in meta for programmatic readers.
    const verb: Record<GoalChange, string> = {
        created: "added",
        deleted: "deleted",
        status: "status changed",
        edited: "edited",
    };
    return (change, goal, now) => {
        try {
            const meta: GoalEventMeta = { change, goalId: goal.id, status: goal.status };
            events.append({
                kind: GOAL_EVENT_KIND,
                role: "agent",
                content: `Goal ${verb[change]}: ${goal.content}`,
                meta,
                session: goal.session,
                ts: now,
            });
        } catch (err) {
            console.warn(
                `goal event not logged: ${err instanceof Error ? err.message : String(err)}`,
            );
        }
    };
}

/** Serialize a goal for the wire: the full record the Goals page edits. A global
 *  (shared) goal has `session: null`; a session-scoped one carries its id. */
function goalToJson(g: Goal) {
    return {
        id: g.id,
        content: g.content,
        status: g.status,
        session: g.session ?? null,
        created: g.created,
        updated: g.updated,
    };
}

/** Serialize a note for the list wire shape (no body, for a compact list). */
function noteToSummaryJson(n: Note) {
    return {
        id: n.id,
        uuid: n.uuid,
        path: n.path,
        title: n.title,
        frontmatter: n.frontmatter,
        created: n.created,
        updated: n.updated,
    };
}

/** Serialize a note for the detail wire shape (with body and its links). */
function noteToDetailJson(notesStore: NotesStore, n: Note) {
    return {
        ...noteToSummaryJson(n),
        content: n.content,
        links: notesStore.linksFrom(n.id).map((l) => ({
            id: l.id,
            toMemory: l.toMemory,
            toNote: l.toNote,
            kind: l.kind,
        })),
    };
}

/**
 * Group the log into one row per conversation for the conversations applet.
 *
 * The log has no `sessions` table by design (a session is just a value in the
 * `session` column), so we derive the list: scan recent events, and for each
 * distinct session keep its newest timestamp, its event count, and a preview
 * line (the most recent user message, falling back to any content). Newest
 * conversation first, matching the stub's ordering.
 *
 * This reads a bounded window (`scan` events) rather than the whole log, so the
 * list reflects recent activity without an unbounded query; the count is the
 * count *within that window*, which is what a recent-activity list wants.
 */
function summarizeSessions(events: EventStore, scan: number) {
    const recent = events.recent({ limit: scan }); // newest first
    const order: string[] = [];
    const acc = new Map<
        string,
        { session: string; when: number; count: number; preview: string }
    >();

    for (const e of recent) {
        const id = e.session;
        if (!id) continue; // events with no session id aren't conversations.
        let row = acc.get(id);
        if (!row) {
            // First time we see this session, and because `recent` is newest
            // first, this is its latest event: the right timestamp to sort by.
            row = { session: id, when: e.ts, count: 0, preview: "" };
            acc.set(id, row);
            order.push(id);
        }
        row.count++;
        // Prefer the most recent user message as the preview (what the person
        // asked); since we walk newest-first, the first user message we meet for
        // a session is its latest, so only fill the slot once.
        if (!row.preview && e.kind === "message" && e.role === "user") {
            row.preview = e.content;
        }
    }

    // Any session whose user turns fell outside the scan window still deserves a
    // preview; fall back to its newest content line.
    for (const e of recent) {
        const id = e.session;
        if (!id) continue;
        const row = acc.get(id);
        if (row && !row.preview) row.preview = e.content;
    }

    return order.map((id) => acc.get(id)!);
}

// ── SSE for chat ────────────────────────────────────────────────────────────

/** Frame and write one SSE event. Each `LoopEvent` the Session yields becomes a
 *  named SSE event carrying a JSON payload, so the client can switch on the same
 *  `kind` vocabulary the REPL renders. */
function sse(res: ServerResponse, event: string, data: unknown): void {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
}

/**
 * Stream one chat turn over SSE.
 *
 * Drives {@link Session.send}, translating each {@link LoopEvent} into an SSE
 * frame the client consumes: `text` deltas as they arrive (the reply, typed
 * out), tool lifecycle as `tool` events, and a terminal `done` carrying the
 * turn's accounting. An error mid-stream (the model fails after the headers are
 * already sent, so we can't change the status) is reported as an `error` SSE
 * event and the stream is closed: the client distinguishes it from `done`.
 *
 * Crucially, the Session was built with an EventStore, so simply running this
 * turn persists the whole exchange (the user message, every tool call/result,
 * the reply) to disk as a side effect. Nothing here writes to the log directly.
 */
async function streamChat(
    session: Session,
    text: string,
    res: ServerResponse,
    images: ImageInput[] = [],
    signal?: AbortSignal,
): Promise<void> {
    res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
    });
    // Announce the session id up front so the client can deep-link this
    // conversation (and so a brand-new session shows up addressable immediately).
    sse(res, "open", { session: session.id });

    try {
        const turn = session.send(text, images, signal);
        let next = await turn.next();
        while (!next.done) {
            const event: LoopEvent = next.value;
            switch (event.kind) {
                case "text":
                    sse(res, "text", { text: event.text });
                    break;
                case "thinking":
                    // The model's reasoning trace, streamed so the client can show
                    // it live (collapsible). Only present when thinking is enabled
                    // on the provider (see buildDeps' providerOptions).
                    sse(res, "thinking", { text: event.text });
                    break;
                case "tool_start":
                    sse(res, "tool", { phase: "start", name: event.name, args: event.args });
                    break;
                case "tool_end":
                    sse(res, "tool", {
                        phase: "end",
                        name: event.name,
                        isError: event.isError,
                    });
                    break;
                case "compacted":
                    sse(res, "compacted", { turn: event.turn });
                    break;
                case "cancelled":
                    // The human pressed "stop": the Session has saved the partial
                    // reply (the `text` deltas already streamed) and logged a
                    // ResponseCancelled marker. Tell the client so it can mark the
                    // reply cancelled while keeping what's on screen; a `done` frame
                    // still follows with the turn's final accounting.
                    sse(res, "cancelled", { text: event.text });
                    break;
                // tool_call_start / tool_call_args / turn_start / loop_done are not
                // surfaced to the client (loop_done's payload is folded into the
                // `done` frame below, from the TurnResult).
            }
            next = await turn.next();
        }
        const result = next.value;
        sse(res, "done", {
            text: result.text,
            modelTurns: result.modelTurns,
            stoppedAtMaxTurns: result.stoppedAtMaxTurns,
            compactions: result.compactions,
            usage: result.usage,
            cancelled: result.cancelled,
        });
    } catch (err) {
        // Headers are already sent, so we can't set a status; report the failure
        // in-band. Surface the neutral HarnessError kind when we have one so the
        // client can tell auth from rate-limit from a generic blip.
        const kind = err instanceof HarnessError ? err.kind : "unknown";
        const message = err instanceof Error ? err.message : String(err);
        sse(res, "error", { kind, message });
    } finally {
        res.end();
    }
}

// ── Knowledge-base routes ────────────────────────────────────────────────────

/** Read a request body and parse it as a JSON object, or return null on any
 *  failure (the caller then 400s). Bounded by {@link readBody}. */
async function readJsonObject(req: IncomingMessage): Promise<Record<string, unknown> | null> {
    try {
        const body = await readBody(req);
        const parsed: unknown = JSON.parse(body || "{}");
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
        return parsed as Record<string, unknown>;
    } catch {
        return null;
    }
}

/**
 * Handle every `/api/notes` and `/api/notes/:uuid` request: the KB's read and
 * write surface. All mutations go through {@link NotesService}, so a write here
 * and a file saved in an editor converge to the same row (the unified write
 * path). Translates {@link NoteError} (bad input, a path/uuid clash) to a 400 so
 * the client gets a real message rather than an opaque 500.
 *
 *  - GET    /api/notes            list (optional `q` search, `prefix` folder)
 *  - GET    /api/notes/:uuid      one note with body + links
 *  - POST   /api/notes            create { title, content, path?, frontmatter? }
 *  - PUT    /api/notes/:uuid      update { title?, content?, path?, frontmatter? }
 *  - DELETE /api/notes/:uuid      delete
 */
async function handleNotes(
    req: IncomingMessage,
    res: ServerResponse,
    deps: ServerDeps,
    url: URL,
    path: string,
): Promise<void> {
    const { notes, notesStore } = deps;
    if (!notes || !notesStore) {
        sendJson(res, 503, { error: "knowledge base is not configured" });
        return;
    }

    // The collection endpoint: list (GET) and create (POST).
    if (path === "/api/notes") {
        if (req.method === "GET") {
            const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
            const q = url.searchParams.get("q");
            const prefix = url.searchParams.get("prefix") ?? undefined;
            const rows = (
                q
                    ? notesStore.search(q, { limit, pathPrefix: prefix })
                    : notesStore.all({ limit, pathPrefix: prefix })
            ).map(noteToSummaryJson);
            sendJson(res, 200, {
                notes: rows,
                total: notesStore.count({ q: q ?? undefined, pathPrefix: prefix }),
            });
            return;
        }
        if (req.method === "POST") {
            const body = await readJsonObject(req);
            if (!body) {
                sendJson(res, 400, { error: "invalid JSON body" });
                return;
            }
            if (typeof body.title !== "string" || body.title.trim() === "") {
                sendJson(res, 400, { error: "title must be a non-empty string" });
                return;
            }
            try {
                const result = await notes.create({
                    title: body.title,
                    content: typeof body.content === "string" ? body.content : "",
                    path: typeof body.path === "string" ? body.path : undefined,
                    frontmatter: asFrontmatter(body.frontmatter),
                });
                sendJson(res, 201, { note: noteToDetailJson(notesStore, result.note) });
            } catch (err) {
                sendNoteError(res, err);
            }
            return;
        }
        sendJson(res, 405, { error: `method ${req.method} not allowed on /api/notes` });
        return;
    }

    // The item endpoint: /api/notes/:uuid (read, update, delete).
    const uuid = decodeURIComponent(path.slice("/api/notes/".length));
    if (uuid === "" || uuid.includes("/")) {
        sendJson(res, 404, { error: "note not found" });
        return;
    }
    const existing = notesStore.getByUuid(uuid);

    if (req.method === "GET") {
        if (!existing) {
            sendJson(res, 404, { error: "note not found" });
            return;
        }
        sendJson(res, 200, { note: noteToDetailJson(notesStore, existing) });
        return;
    }

    if (req.method === "PUT") {
        if (!existing) {
            sendJson(res, 404, { error: "note not found" });
            return;
        }
        const body = await readJsonObject(req);
        if (!body) {
            sendJson(res, 400, { error: "invalid JSON body" });
            return;
        }
        // Build a patch from only the provided fields, so an omitted field is left
        // untouched rather than cleared.
        const patch: Parameters<NotesService["update"]>[1] = {};
        if (typeof body.title === "string") patch.title = body.title;
        if (typeof body.content === "string") patch.content = body.content;
        if (typeof body.path === "string") patch.path = body.path;
        if ("frontmatter" in body) patch.frontmatter = asFrontmatter(body.frontmatter);
        try {
            const result = await notes.update(existing.id, patch);
            if (!result) {
                sendJson(res, 404, { error: "note not found" });
                return;
            }
            sendJson(res, 200, { note: noteToDetailJson(notesStore, result.note) });
        } catch (err) {
            sendNoteError(res, err);
        }
        return;
    }

    if (req.method === "DELETE") {
        if (!existing) {
            sendJson(res, 404, { error: "note not found" });
            return;
        }
        const removed = await notes.remove(existing.id);
        sendJson(res, 200, { deleted: removed });
        return;
    }

    sendJson(res, 405, { error: `method ${req.method} not allowed on ${path}` });
}

/** Coerce an unknown request field into a frontmatter map, dropping any value
 *  that isn't a supported scalar or string array (the store re-validates too). */
function asFrontmatter(value: unknown): NoteFrontmatter {
    if (value === null || typeof value !== "object" || Array.isArray(value)) return {};
    const out: NoteFrontmatter = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        if (v === null || typeof v === "string" || typeof v === "boolean") {
            out[k] = v;
        } else if (typeof v === "number" && Number.isFinite(v)) {
            out[k] = v;
        } else if (Array.isArray(v) && v.every((x) => typeof x === "string")) {
            out[k] = v as string[];
        }
    }
    return out;
}

/** Map a NoteError to a 400 (bad input / clash); anything else to a 500. */
function sendNoteError(res: ServerResponse, err: unknown): void {
    if (err instanceof NoteError) {
        sendJson(res, 400, { error: err.message });
        return;
    }
    const message = err instanceof Error ? err.message : String(err);
    sendJson(res, 500, { error: message });
}

// ── Dreams routes ────────────────────────────────────────────────────────────

/** The largest number of dreams a single POST may ask for. A dream is several
 *  model turns, so an unbounded `count` would be an open-ended spend; this caps
 *  one request to a reasonable batch (a client wanting more dreams sends another
 *  request). Reads (GET) are bounded separately by {@link clampPage}. */
const MAX_DREAM_BATCH = 10;

/**
 * Handle the dreams applet's two routes: read the accumulated dreams and run new
 * ones on demand.
 *
 *  - GET  /api/dreams           the logged dreams, newest first. A scoped read of
 *                               the event log filtered to {@link DREAM_EVENT_KIND},
 *                               each event flattened to its structured record
 *                               (persona / scenario / choice) by
 *                               {@link dreamEventToJson}. `limit` bounds the page.
 *  - POST /api/dreams           run `count` dreams now and return them. Drives
 *                               {@link dreamLoop} directly against the shared
 *                               stores and client — no conversation, the way
 *                               dreaming is meant to run (during downtime, with no
 *                               one watching). Each dream appends a `dream` event
 *                               as a side effect, so a subsequent GET reflects them.
 *
 * The POST tolerates per-dream failures the way the loop does: a malformed dream
 * is recorded in the result's `failures`, not fatal, so a single bad dream
 * doesn't sink the batch. The whole batch only fails (a real error status) if the
 * client itself throws — a transport/auth problem the client should see classified.
 */
async function handleDreams(
    req: IncomingMessage,
    res: ServerResponse,
    deps: ServerDeps,
    url: URL,
): Promise<void> {
    if (req.method === "GET") {
        const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
        // recent() is newest-first and filters by kind in the store, so the dreams
        // view is the dream events alone, freshest at the top — the order a "what
        // has the Construct been dreaming" list wants.
        const rows = deps.events.recent({ kind: DREAM_EVENT_KIND, limit }).map(dreamEventToJson);
        sendJson(res, 200, { dreams: rows, total: deps.events.count({ kind: DREAM_EVENT_KIND }) });
        return;
    }

    if (req.method === "POST") {
        const body = await readJsonObject(req);
        if (!body) {
            sendJson(res, 400, { error: "invalid JSON body" });
            return;
        }
        // `count` defaults to one dream; clamp to [1, MAX_DREAM_BATCH] so a missing
        // or silly value still runs exactly one rather than erroring or running away.
        const requested = typeof body.count === "number" ? Math.floor(body.count) : 1;
        const count = Math.min(Math.max(1, requested), MAX_DREAM_BATCH);
        // `deal` opts the dreamer into stakes (a biased dreamer); a truthy `deal`
        // flag deals the default count, an object passes through its `count`.
        const deal =
            body.deal === true
                ? {}
                : body.deal && typeof body.deal === "object"
                  ? (body.deal as { count?: number })
                  : undefined;

        const result = await dreamLoop({
            client: deps.client,
            store: deps.store,
            events: deps.events,
            count,
            deal,
        });

        sendJson(res, 200, {
            // Return the dreams this batch produced in the same flattened shape the
            // GET serves, so the client can prepend them without a re-fetch. They're
            // already on the log too (dreamOnce appended each), so a refresh agrees.
            dreams: result.dreams.map((d: Dream) => dreamEventToJson(d.event)),
            // Surface the misses rather than hiding them: a batch that asked for 3
            // and produced 1 should say so, with each failure's reason.
            failures: result.failures.map((f) => ({
                index: f.index,
                error: f.error instanceof Error ? f.error.message : String(f.error),
            })),
        });
        return;
    }

    sendJson(res, 405, { error: `method ${req.method} not allowed on /api/dreams` });
}

// ── Goals routes ─────────────────────────────────────────────────────────────

/**
 * Handle every `/api/goals` and `/api/goals/:id` request: the human-editable
 * surface over the {@link GoalStore}. Goals are the harness's most *immediate*
 * standing context — unlike a memory the agent chose to keep or a note it wrote,
 * a goal here is intent a human can set, sharpen, or retire live, and the next
 * turn reads it (see goalContext). Two ownership tiers, both editable here:
 *
 *  - **global** (`session: null`) — shared goals every conversation sees.
 *  - **session** (`session: <id>`) — goals scoped to one conversation, the same
 *    rows the agent's goal_set writes against that session.
 *
 *  - GET    /api/goals?scope=&session=&status=   list, filtered
 *  - POST   /api/goals  { content, session? }    create (no session ⇒ global)
 *  - PUT    /api/goals/:id { content?, status? } edit text and/or status
 *  - DELETE /api/goals/:id                        remove (prefer status=abandoned)
 *
 * Mirrors {@link handleNotes}: list/read are GETs, writes funnel through the
 * store and translate {@link GoalError} into a clean 400 the client can show.
 */
async function handleGoals(
    req: IncomingMessage,
    res: ServerResponse,
    deps: ServerDeps,
    url: URL,
    path: string,
): Promise<void> {
    if (path === "/api/goals") {
        if (req.method === "GET") {
            // `scope` picks the ownership tier the read sees:
            //  - all (default): every goal, across global and all sessions.
            //  - global: only shared goals (session IS NULL).
            //  - session: only the goals of `session` (requires it).
            const scopeParam = url.searchParams.get("scope");
            const sessionParam = url.searchParams.get("session") ?? undefined;
            const statusParam = url.searchParams.get("status");
            const status = isGoalStatus(statusParam) ? statusParam : undefined;
            // A bad status filter is a client error worth naming, not a silent
            // "every status" that hides the typo.
            if (statusParam !== null && status === undefined) {
                sendJson(res, 400, { error: "status must be one of active, done, abandoned" });
                return;
            }
            const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);

            const query: Parameters<GoalStore["list"]>[0] = { status, limit };
            if (scopeParam === "global") {
                query.scope = "global";
            } else if (scopeParam === "session") {
                if (!sessionParam) {
                    sendJson(res, 400, { error: "scope=session requires a session query param" });
                    return;
                }
                query.scope = "session";
                query.session = sessionParam;
            } else if (sessionParam) {
                // No explicit scope but a session given: the legacy "this session's
                // goals" filter, kept for the deep-link from a conversation.
                query.session = sessionParam;
            }

            const goals = deps.goals.list(query).map(goalToJson);
            sendJson(res, 200, { goals, total: deps.goals.count({ status }) });
            return;
        }

        if (req.method === "POST") {
            const body = await readJsonObject(req);
            if (!body) {
                sendJson(res, 400, { error: "invalid JSON body" });
                return;
            }
            if (typeof body.content !== "string") {
                sendJson(res, 400, { error: "content must be a string" });
                return;
            }
            // An explicit empty/whitespace `session` means "no session" (global),
            // not a session literally named "". Only a non-empty string scopes it.
            const session =
                typeof body.session === "string" && body.session.trim() !== ""
                    ? body.session
                    : undefined;
            try {
                const goal = deps.goals.create({ content: body.content, session });
                sendJson(res, 201, { goal: goalToJson(goal) });
            } catch (err) {
                sendGoalError(res, err);
            }
            return;
        }

        sendJson(res, 405, { error: `method ${req.method} not allowed on /api/goals` });
        return;
    }

    // The item endpoint: /api/goals/:id (update, delete).
    const idText = decodeURIComponent(path.slice("/api/goals/".length));
    const id = Number(idText);
    if (!idText || !Number.isInteger(id) || id <= 0) {
        sendJson(res, 404, { error: "goal not found" });
        return;
    }

    if (req.method === "PUT") {
        const body = await readJsonObject(req);
        if (!body) {
            sendJson(res, 400, { error: "invalid JSON body" });
            return;
        }
        if (body.content === undefined && body.status === undefined) {
            sendJson(res, 400, { error: "provide content and/or status to update" });
            return;
        }
        let status: GoalStatus | undefined;
        if (body.status !== undefined) {
            if (!isGoalStatus(body.status)) {
                sendJson(res, 400, { error: "status must be one of active, done, abandoned" });
                return;
            }
            status = body.status;
        }
        try {
            // Apply the content edit and the status change independently, the way
            // goal_update does; either touching a missing id means a 404.
            let goal: Goal | undefined;
            let found = false;
            if (typeof body.content === "string") {
                goal = deps.goals.edit(id, body.content);
                found = found || goal !== undefined;
            } else if (body.content !== undefined) {
                sendJson(res, 400, { error: "content must be a string" });
                return;
            }
            if (status !== undefined) {
                goal = deps.goals.setStatus(id, status);
                found = found || goal !== undefined;
            }
            if (!goal) {
                if (found) {
                    sendJson(res, 500, { error: "update failed" });
                } else {
                    sendJson(res, 404, { error: `no goal with id ${id}` });
                }
                return;
            }
            sendJson(res, 200, { goal: goalToJson(goal) });
        } catch (err) {
            sendGoalError(res, err);
        }
        return;
    }

    if (req.method === "DELETE") {
        const removed = deps.goals.delete(id);
        if (!removed) {
            sendJson(res, 404, { error: `no goal with id ${id}` });
            return;
        }
        sendJson(res, 200, { deleted: true });
        return;
    }

    sendJson(res, 405, { error: `method ${req.method} not allowed on ${path}` });
}

/** Map a {@link GoalError} (bad input the store refused) to a 400 with its
 *  message; anything else is a real 500. Mirrors {@link sendNoteError}. */
function sendGoalError(res: ServerResponse, err: unknown): void {
    if (err instanceof GoalError) {
        sendJson(res, 400, { error: err.message });
        return;
    }
    const message = err instanceof Error ? err.message : String(err);
    sendJson(res, 500, { error: message });
}

// ── Status route ─────────────────────────────────────────────────────────────

/**
 * Assemble the runtime status: the truth about what this process is *right now*,
 * and the catalogue the settings page's controls render from. Three layers:
 *
 *  - **live knobs** (read off {@link RuntimeConfig} now): the model in use, which
 *    server and local tools are enabled, the effort level. These reflect the
 *    current runtime, so a model switch or tool toggle shows up the moment it
 *    lands — they are NOT the boot snapshot. With no runtime (a bare test), they
 *    fall back to the client's model and empty/default tool sets.
 *  - **catalogue** (static, from the model + tool catalogues): the providers and
 *    their model variants for the two dropdowns, the server-tool and local-tool
 *    lists for the toggles, and the effort levels — so the page never hardcodes
 *    a model id or tool name. Each tool/model entry carries an `enabled`/current
 *    flag so the controls render in the right state.
 *  - **static + dynamic context** (the boot snapshot and live deps): where data
 *    lives, the compaction threshold, the standing features, the shell policy
 *    (static); the schema version and live session ids (dynamic).
 *
 * Read-only, side-effect free: it never touches a Session or mutates a store, so
 * the page can poll it freely. Secret-free — the embedder is a yes/no, never a key.
 */
function handleStatus(res: ServerResponse, deps: ServerDeps): void {
    const s = deps.status;
    const rt = deps.runtime;
    // The schema version is the same across every store (one file, one
    // user_version); read it off the memory store, which always exists.
    const schemaVersion = deps.store.version;
    const liveSessions = deps.sessions.ids();

    // The model in use right now, and the provider that serves it, read live off
    // the runtime (or the client when no runtime is wired). The provider value
    // drives which model dropdown is shown.
    const currentModel = rt?.model ?? deps.client.model;
    const currentProvider = providerForModel(currentModel)?.id ?? null;

    // The enabled server tools and the effort level, live off the runtime.
    const enabledServerTools = rt?.serverTools ?? [];
    const effort = rt?.effort ?? null;

    sendJson(res, 200, {
        provider: {
            // Which provider and model the next turn uses, plus the capabilities
            // the provider reports (small, secret-free) so the page can show
            // thinking/serverTools/streaming support rather than a hardcoded guess.
            id: currentProvider,
            model: currentModel,
            capabilities: deps.client.capabilities,
        },
        // The catalogue the two dropdowns render from: every provider and its
        // model variants, each marked `current` when it's the one in use. Static
        // provider knowledge, so the page picks up a new provider/model without a
        // frontend change.
        providers: PROVIDERS.map((p) => ({
            id: p.id,
            label: p.label,
            models: p.models.map((m) => ({
                id: m.id,
                label: m.label,
                contextWindow: m.contextWindow,
                maxOutput: m.maxOutput,
                current: m.id === currentModel,
            })),
        })),
        // The server-tool toggles: the full catalogue, each flagged with whether
        // it's currently enabled. A toggle here lands on every live conversation's
        // next turn (the options object is shared and mutated in place).
        serverTools: SERVER_TOOL_CATALOG.map((t) => ({
            id: t.id,
            label: t.label,
            note: t.note,
            enabled: enabledServerTools.includes(t.id),
        })),
        // The local-tool toggles: the harness-owned groups, each with the tool
        // names it covers and whether it's on. Toggling applies to conversations
        // started after the change (a live Session captured its tools at start).
        localTools: (rt?.localGroups ?? []).map((g) => ({
            key: g.key,
            label: g.label,
            note: g.note,
            toolNames: g.toolNames,
            enabled: rt!.isLocalEnabled(g.key),
        })),
        // The effort levels for that dropdown, plus the level in use (null = the
        // provider default, "high").
        effort: { current: effort, levels: EFFORT_LEVELS },
        storage: {
            memoryDb: s?.memoryDb ?? null,
            kbDir: s?.kbDir ?? null,
            schemaVersion,
            // The current row counts: cheap indexed COUNT(*)s, useful orientation
            // ("how much has this Construct accumulated") and a liveness check.
            memories: deps.store.count(),
            events: deps.events.count(),
            goals: deps.goals.count(),
        },
        compactAt: s?.compactAt ?? null,
        embeddingConfigured: s?.embeddingConfigured ?? false,
        features: {
            dreams: s?.dreamsEnabled ?? false,
            transcriptRecall: s?.transcriptRecall ?? false,
            workingMind: s?.workingMind ?? false,
        },
        // How the local shell is governed. Defaults to unrestricted (the
        // historical behavior) when no snapshot is supplied.
        shellPolicy: s?.shellPolicy ?? { mode: "unrestricted", allowedCwdRoots: [] },
        liveSessions,
    });
}

// ── Settings write route ─────────────────────────────────────────────────────

/**
 * The settings page's write surface: turn the live runtime knobs.
 *
 *  - PATCH /api/settings  { model?, serverTools?, effort?, localTools? }
 *
 * A single PATCH applies any subset of the knobs, each validated by
 * {@link RuntimeConfig}: an unknown model id, server-tool name, effort level, or
 * local-tool key is a 400 (the bad field is named) and nothing changes. Applied
 * fields take effect immediately — the model and server tools/effort on every
 * conversation's next turn (the client and options object are process-wide and
 * read per turn), local-tool toggles on conversations started after the change.
 *
 * `model` carries just the model id; the provider is derived from it (the
 * provider dropdown's value follows the model). `effort: null` clears the level
 * back to the provider default. `localTools` is a map of group-key → enabled.
 *
 * After applying, it returns the same payload {@link handleStatus} serves, so the
 * client refreshes its whole view from one response.
 */
async function handleSettings(
    req: IncomingMessage,
    res: ServerResponse,
    deps: ServerDeps,
): Promise<void> {
    const rt = deps.runtime;
    if (!rt) {
        sendJson(res, 503, { error: "runtime configuration is not available" });
        return;
    }
    if (req.method !== "PATCH") {
        sendJson(res, 405, { error: `method ${req.method} not allowed on /api/settings` });
        return;
    }
    const body = await readJsonObject(req);
    if (!body) {
        sendJson(res, 400, { error: "invalid JSON body" });
        return;
    }

    try {
        // Each field is optional and applied only when present. Validation lives
        // in RuntimeConfig, which throws a classified HarnessError on bad input;
        // the catch below turns that into a 400 naming the offending field.
        if (body.model !== undefined) {
            if (typeof body.model !== "string") {
                sendJson(res, 400, { error: "model must be a string" });
                return;
            }
            rt.setModel(body.model);
        }
        if (body.serverTools !== undefined) {
            if (
                !Array.isArray(body.serverTools) ||
                body.serverTools.some((t) => typeof t !== "string")
            ) {
                sendJson(res, 400, { error: "serverTools must be an array of strings" });
                return;
            }
            rt.setServerTools(body.serverTools as string[]);
        }
        if (body.effort !== undefined) {
            if (body.effort !== null && typeof body.effort !== "string") {
                sendJson(res, 400, { error: "effort must be a string or null" });
                return;
            }
            rt.setEffort(body.effort as string | null);
        }
        if (body.localTools !== undefined) {
            if (
                body.localTools === null ||
                typeof body.localTools !== "object" ||
                Array.isArray(body.localTools)
            ) {
                sendJson(res, 400, { error: "localTools must be an object of key → boolean" });
                return;
            }
            for (const [key, enabled] of Object.entries(
                body.localTools as Record<string, unknown>,
            )) {
                if (typeof enabled !== "boolean") {
                    sendJson(res, 400, { error: `localTools.${key} must be a boolean` });
                    return;
                }
                rt.setLocalEnabled(key, enabled);
            }
        }
    } catch (err) {
        if (err instanceof HarnessError) {
            sendJson(res, statusForKind(err.kind), { error: err.message });
            return;
        }
        const message = err instanceof Error ? err.message : String(err);
        sendJson(res, 500, { error: message });
        return;
    }

    // Echo the full status back so the client re-renders from one response.
    handleStatus(res, deps);
}

// ── Presence route ───────────────────────────────────────────────────────────

/**
 * The human's presence surface: read it, or pin/clear a manual override.
 *
 *  - GET  /api/presence            the computed presence right now.
 *  - PUT  /api/presence  { state } pin a manual status, where `state` is
 *                                  `online` (clears the override back to the
 *                                  automatic Online/Away-by-activity axis), `dnd`,
 *                                  or `offline`. `away` is rejected: it's derived
 *                                  from silence, not a status you announce.
 *
 * Both compute against one `Date.now()` and return the same {@link Presence}
 * shape, so the client renders the dot and updates from one reply. The state is
 * in-process (see {@link UserPresence}); a restart re-seeds Online from boot.
 */
async function handlePresence(
    req: IncomingMessage,
    res: ServerResponse,
    deps: ServerDeps,
): Promise<void> {
    if (req.method === "GET") {
        sendJson(res, 200, deps.presence.read(Date.now()));
        return;
    }
    if (req.method === "PUT") {
        const body = await readJsonObject(req);
        if (!body) {
            sendJson(res, 400, { error: "invalid JSON body" });
            return;
        }
        const want = body.state;
        // Only the pinnable states are accepted. `online` is the clear-to-automatic
        // value; `away` is intentionally absent (derived, not announced).
        if (want !== "online" && want !== "dnd" && want !== "offline") {
            sendJson(res, 400, {
                error: 'state must be "online" (clears the override), "dnd", or "offline"',
            });
            return;
        }
        sendJson(res, 200, deps.presence.setOverride(want as "online" | Override, Date.now()));
        return;
    }
    sendJson(res, 405, { error: `method ${req.method} not allowed on /api/presence` });
}

// ── Context inspector route ──────────────────────────────────────────────────

/**
 * Preview the context a turn would be built from, for the context inspector: what
 * the Construct actually sees before it answers, assembled but never sent.
 *
 *  - GET /api/context?session=<id>&q=<draft>
 *
 * Delegates to {@link Session.inspectContext}, which is read-only by construction
 * (it recalls memory without reinforcing, renders the working mind without
 * ticking, and touches no goal or event). To keep the *whole* request read-only,
 * we {@link SessionPool.peek} rather than {@link SessionPool.resolve}: a past
 * conversation is resumed transiently for this read and never brought live in the
 * pool, so opening the inspector on an old conversation doesn't change what's
 * loaded. `q` defaults to empty (recall against an empty draft still shows the
 * standing injections: goals, the last dream, the working mind).
 */
async function handleContext(res: ServerResponse, deps: ServerDeps, url: URL): Promise<void> {
    const session = url.searchParams.get("session") ?? undefined;
    const q = url.searchParams.get("q") ?? "";
    const target = await deps.sessions.peek(session);
    const inspection = await target.inspectContext(q);
    sendJson(res, 200, inspection);
}

// ── Memory curation routes ───────────────────────────────────────────────────

/**
 * Handle every `/api/memories` and `/api/memories/:id` request: the read and
 * curation surface over the {@link MemoryStore}, the human counterpart to the
 * agent's memory_save/forget tools. Where the agent curates its own memory as it
 * runs, this lets a human inspect and correct that curation: see each memory's
 * earned strength and provenance, sharpen its text or tags, or forget it.
 *
 *  - GET    /api/memories            list (optional `q` search), each enriched
 *                                    with strength / lastSurfaced / provenance /
 *                                    embedding-present.
 *  - GET    /api/memories/:id        one memory (same enriched shape) plus the
 *                                    source event when it has provenance.
 *  - PUT    /api/memories/:id        edit { content?, tags?, importance? }.
 *  - DELETE /api/memories/:id        forget it.
 *
 * Strength is read against one shared `now` per request so a list ranks
 * consistently. Mirrors {@link handleGoals}: writes translate {@link MemoryError}
 * into a clean 400.
 */
async function handleMemories(
    req: IncomingMessage,
    res: ServerResponse,
    deps: ServerDeps,
    url: URL,
    path: string,
): Promise<void> {
    const now = Date.now();

    if (path === "/api/memories") {
        if (req.method === "GET") {
            const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
            const q = url.searchParams.get("q");
            const rows = (q ? deps.store.search(q, { limit }) : deps.store.all({ limit })).map(
                (m) => memoryToJson(m, deps, now),
            );
            sendJson(res, 200, { memories: rows, total: deps.store.count() });
            return;
        }
        sendJson(res, 405, { error: `method ${req.method} not allowed on /api/memories` });
        return;
    }

    // The item endpoint: /api/memories/:id (detail, update, delete).
    const idText = decodeURIComponent(path.slice("/api/memories/".length));
    const id = Number(idText);
    if (!idText || !Number.isInteger(id) || id <= 0) {
        sendJson(res, 404, { error: "memory not found" });
        return;
    }

    if (req.method === "GET") {
        const memory = deps.store.get(id);
        if (!memory) {
            sendJson(res, 404, { error: "memory not found" });
            return;
        }
        // Detail adds the source event itself (content + when) so the page can show
        // what the memory was curated from without a second round-trip.
        const base = memoryToJson(memory, deps, now);
        const sourceEvent =
            base.provenance && base.provenance.eventId !== undefined
                ? deps.events.get(base.provenance.eventId)
                : undefined;
        sendJson(res, 200, {
            memory: base,
            sourceEvent: sourceEvent ? eventToJson(sourceEvent) : null,
        });
        return;
    }

    if (req.method === "PUT") {
        const body = await readJsonObject(req);
        if (!body) {
            sendJson(res, 400, { error: "invalid JSON body" });
            return;
        }
        const patch: Partial<Pick<Memory, "content" | "tags" | "importance">> = {};
        if (body.content !== undefined) {
            if (typeof body.content !== "string") {
                sendJson(res, 400, { error: "content must be a string" });
                return;
            }
            patch.content = body.content;
        }
        if (body.tags !== undefined) {
            if (!Array.isArray(body.tags) || body.tags.some((t) => typeof t !== "string")) {
                sendJson(res, 400, { error: "tags must be an array of strings" });
                return;
            }
            patch.tags = body.tags as string[];
        }
        if (body.importance !== undefined) {
            // null clears importance (the key stays present, so the store's update
            // sees a deliberate clear); a number sets it. Anything else is a 400.
            if (body.importance !== null && typeof body.importance !== "number") {
                sendJson(res, 400, { error: "importance must be a number or null" });
                return;
            }
            patch.importance = body.importance === null ? undefined : body.importance;
        }
        if (Object.keys(patch).length === 0) {
            sendJson(res, 400, { error: "provide content, tags, and/or importance to update" });
            return;
        }
        try {
            const updated = deps.store.update(id, patch, now);
            if (!updated) {
                sendJson(res, 404, { error: `no memory with id ${id}` });
                return;
            }
            sendJson(res, 200, { memory: memoryToJson(updated, deps, now) });
        } catch (err) {
            sendMemoryError(res, err);
        }
        return;
    }

    if (req.method === "DELETE") {
        const removed = deps.store.delete(id);
        if (!removed) {
            sendJson(res, 404, { error: `no memory with id ${id}` });
            return;
        }
        sendJson(res, 200, { deleted: true });
        return;
    }

    sendJson(res, 405, { error: `method ${req.method} not allowed on ${path}` });
}

/** Map a {@link MemoryError} (bad input the store refused) to a 400; anything
 *  else is a real 500. Mirrors {@link sendGoalError}. */
function sendMemoryError(res: ServerResponse, err: unknown): void {
    if (err instanceof MemoryError) {
        sendJson(res, 400, { error: err.message });
        return;
    }
    const message = err instanceof Error ? err.message : String(err);
    sendJson(res, 500, { error: message });
}

// ── Router ──────────────────────────────────────────────────────────────────

/** Build the request handler over a set of deps. Pure routing: it owns no
 *  state beyond the deps, so it's straightforward to exercise in a test by
 *  pointing it at an in-memory store and a fake client. */
export function createHandler(deps: ServerDeps) {
    return async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
        cors(res, deps.corsOrigin);
        if (req.method === "OPTIONS") {
            res.writeHead(204);
            res.end();
            return;
        }

        const url = new URL(req.url ?? "/", "http://localhost");
        const path = url.pathname;

        try {
            if (req.method === "GET" && path === "/api/health") {
                sendJson(res, 200, { ok: true, sessions: deps.sessions.ids() });
                return;
            }

            // The truthful runtime status: model, tools, storage, features. A
            // read-only snapshot the settings page renders in place of hardcoded
            // rows. No secrets leave (the embedder is a yes/no).
            if (req.method === "GET" && path === "/api/status") {
                handleStatus(res, deps);
                return;
            }

            // The settings write surface: PATCH the live runtime knobs (model,
            // server tools, effort, local tools). Each applied field takes effect
            // immediately — model/server-tools/effort on every conversation's next
            // turn, local-tool toggles on conversations started after the change.
            // Echoes the full status back so the client refreshes from one reply.
            if (path === "/api/settings") {
                await handleSettings(req, res, deps);
                return;
            }

            // The human's presence. GET computes it from when they last sent a
            // message plus any pinned override; PUT pins/clears that override
            // (online clears back to automatic, dnd/offline pin). The status dot
            // the client shows reads from here and polls so Away surfaces after
            // silence without a message.
            if (path === "/api/presence") {
                await handlePresence(req, res, deps);
                return;
            }

            // The context inspector: a read-only preview of what a turn would be
            // built from for a draft, with per-section token estimates and source
            // ids. Assembles the context but never sends it, and mutates nothing
            // (no reinforce, no working-mind tick, no event append).
            if (req.method === "GET" && path === "/api/context") {
                await handleContext(res, deps, url);
                return;
            }

            if (req.method === "GET" && path === "/api/sessions") {
                const scan = clampPage(url.searchParams.get("scan"), 1000, 5000);
                const live = deps.sessions.ids();
                const liveSet = new Set(live);
                const sessions = summarizeSessions(deps.events, scan).map((s) => ({
                    ...s,
                    // Mark every conversation currently held live in the pool
                    // (in-memory, accepting turns). The rest live only in the log
                    // until they're resumed — which any of them can be, by sending
                    // a turn into them. So `live` is "loaded now", not "the only one
                    // you can talk to".
                    live: liveSet.has(s.session),
                }));
                sendJson(res, 200, { sessions, live });
                return;
            }

            if (req.method === "GET" && path === "/api/events") {
                const session = url.searchParams.get("session");
                if (!session) {
                    sendJson(res, 400, { error: "session query param is required" });
                    return;
                }
                const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
                // Oldest first: natural reading order for a transcript. Enrich each
                // event with its image attachment refs (cheap metadata-only reads;
                // only user messages ever carry any) so a replay can render them.
                const rows = deps.events
                    .recent({ session, limit })
                    .reverse()
                    .map((e) => eventToJson(e, deps.events.attachmentsFor(e.id)));
                sendJson(res, 200, {
                    session,
                    live: deps.sessions.has(session),
                    events: rows,
                });
                return;
            }

            // Memory: list/detail reads plus the human curation writes (edit,
            // forget), each memory enriched with strength / provenance / embedding.
            if (path === "/api/memories" || path.startsWith("/api/memories/")) {
                await handleMemories(req, res, deps, url, path);
                return;
            }

            if (req.method === "GET" && path === "/api/log") {
                const limit = clampPage(url.searchParams.get("limit"), DEFAULT_PAGE, 1000);
                const kind = url.searchParams.get("kind") ?? undefined;
                const rows = deps.events
                    .recent({ limit, kind })
                    .map((e) => eventToJson(e, deps.events.attachmentsFor(e.id)));
                sendJson(res, 200, { events: rows, total: deps.events.count() });
                return;
            }

            // The slash-command catalogue: a static read of the registry, so the
            // client can list the same commands (name, signature parts, params)
            // in its `/` menu that the REPL prints under `/help`. No per-session
            // state: the menu is the same for every conversation.
            if (req.method === "GET" && path === "/api/commands") {
                sendJson(res, 200, { commands: BUILTIN_COMMANDS });
                return;
            }

            // Knowledge-base routes: list/read are GETs; create/update/delete are
            // the harness's first write endpoints, all funneling through the one
            // NotesService write path so a UI write and a file save converge.
            if (path === "/api/notes" || path.startsWith("/api/notes/")) {
                await handleNotes(req, res, deps, url, path);
                return;
            }

            // Dreams: GET reads the accumulated dreams from the log; POST runs new
            // ones on demand by driving dreamLoop directly (no conversation), which
            // appends each as a dream event the GET then reflects.
            if (path === "/api/dreams") {
                await handleDreams(req, res, deps, url);
                return;
            }

            // Goals: the human-editable standing-context surface. List/read are
            // GETs; create/edit/delete mutate the GoalStore directly (the same rows
            // the agent's goal tools write), so a UI edit and an agent goal_set
            // converge on one store the next turn reads.
            if (path === "/api/goals" || path.startsWith("/api/goals/")) {
                await handleGoals(req, res, deps, url, path);
                return;
            }

            if (req.method === "POST" && path === "/api/chat") {
                let text: string;
                let wantSession: string | undefined;
                let images: ImageInput[] = [];
                try {
                    // Raised well past the default 256 KB: a single photo's base64
                    // dwarfs it. The per-image byte cap (MAX_ATTACHMENT_BYTES) still
                    // bounds each image; this bounds the whole multi-image payload.
                    const body = await readBody(req, CHAT_BODY_LIMIT);
                    const parsed = JSON.parse(body || "{}") as {
                        message?: unknown;
                        session?: unknown;
                        images?: unknown;
                    };
                    // A turn must carry *something*: text or at least one image.
                    const hasText =
                        typeof parsed.message === "string" && parsed.message.trim() !== "";
                    images = parseImages(parsed.images);
                    if (!hasText && images.length === 0) {
                        sendJson(res, 400, {
                            error: "message must be a non-empty string or include images",
                        });
                        return;
                    }
                    text = typeof parsed.message === "string" ? parsed.message : "";
                    // An optional session id resumes (or continues) that
                    // conversation; omitted, the turn lands on the default live one.
                    if (typeof parsed.session === "string" && parsed.session !== "") {
                        wantSession = parsed.session;
                    }
                } catch (err) {
                    // A bad image (wrong type, too big, undecodable base64) is a
                    // client error worth naming; everything else is a malformed body.
                    const msg = err instanceof ImageParseError ? err.message : "invalid JSON body";
                    sendJson(res, 400, { error: msg });
                    return;
                }
                // Resolve the conversation before streaming: resuming reads the log
                // and rehydrates history, which can fail (a store error), so do it
                // here where we can still send a real error status. Once streamChat
                // writes the SSE head we can only report errors in-band.
                const session = await deps.sessions.resolve(wantSession);
                // The human just sent something, so they're here: advance presence
                // (Online again, Away countdown restarted) and lift an `offline`
                // override — you can't be talking and offline. A pinned `dnd`
                // survives: present and not-to-be-disturbed are compatible.
                deps.presence.touch(Date.now());
                // Register this turn so `/api/chat/cancel` can stop it by session id,
                // and hand its abort signal to the stream. The `finally` drops the
                // registry entry however the turn ends (done, error, or cancelled),
                // so the map never holds a settled turn.
                const cancelSignal = deps.inflight.begin(session.id);
                try {
                    await streamChat(session, text, res, images, cancelSignal);
                } finally {
                    deps.inflight.end(session.id);
                }
                return;
            }

            // Stop an in-flight streamed reply. The chat SSE stream can't carry a
            // request back to the server, so cancellation is this separate, small
            // call: the client POSTs the session id of the turn to stop, and we
            // abort that turn's controller. The abort routes through the bridge's
            // cancel path, so the Session saves the partial reply (the prose already
            // streamed) and logs a ResponseCancelled marker — the original chat
            // stream then sends its own `cancelled` then `done` frames and closes.
            // This route just fires the abort; it doesn't touch the SSE stream.
            if (req.method === "POST" && path === "/api/chat/cancel") {
                const body = await readJsonObject(req);
                const session = body && typeof body.session === "string" ? body.session : "";
                if (!session) {
                    sendJson(res, 400, { error: "session must be a non-empty string" });
                    return;
                }
                // 404 when nothing is in flight for that id (a stale or duplicate
                // cancel), so the client can tell "stopped it" from "too late".
                const cancelled = deps.inflight.cancel(session);
                if (!cancelled) {
                    sendJson(res, 404, { error: `no in-flight turn for session ${session}` });
                    return;
                }
                sendJson(res, 200, { cancelled: true, session });
                return;
            }

            // Serve one image attachment's bytes by id, so a replayed conversation
            // can render the pictures a past user turn carried. The bytes live in
            // the event log's attachment side table; this is the only read that
            // pulls them. Content-Type is the stored MIME; cache hard since an
            // attachment is immutable once written.
            if (req.method === "GET" && path.startsWith("/api/attachments/")) {
                const id = Number(path.slice("/api/attachments/".length));
                if (!Number.isInteger(id) || id <= 0) {
                    sendJson(res, 400, { error: "attachment id must be a positive integer" });
                    return;
                }
                const att = deps.events.getAttachment(id);
                if (!att) {
                    sendJson(res, 404, { error: `no attachment ${id}` });
                    return;
                }
                res.writeHead(200, {
                    "Content-Type": att.mediaType,
                    "Content-Length": String(att.data.length),
                    "Cache-Control": "public, max-age=31536000, immutable",
                });
                res.end(Buffer.from(att.data));
                return;
            }

            sendJson(res, 404, { error: `no route for ${req.method} ${path}` });
        } catch (err) {
            // A failure before we've sent any bytes: classify it to a status so
            // the client gets a meaningful code, not a bare 500, where we can.
            if (res.headersSent) {
                res.end();
                return;
            }
            const status = err instanceof HarnessError ? statusForKind(err.kind) : 500;
            const message = err instanceof Error ? err.message : String(err);
            sendJson(res, status, { error: message });
        }
    };
}

/** Parse a `limit`/`scan`-style query param, falling back to `def` and capping
 *  at `max`, so a client can't ask for an unbounded read. */
function clampPage(raw: string | null, def: number, max: number): number {
    const n = raw === null ? def : Number(raw);
    if (!Number.isFinite(n) || n <= 0) return def;
    return Math.min(Math.floor(n), max);
}

/** Boot the server: build deps, start listening, and tear the stores down
 *  cleanly on SIGINT/SIGTERM so the WAL is checkpointed on exit. */
function main(): void {
    if (!process.env.ANTHROPIC_API_KEY) {
        console.error("construct-harness server: ANTHROPIC_API_KEY is not set; chat will 401.");
    }

    const deps = buildDeps();
    const handler = createHandler(deps);
    const server = createServer((req, res) => {
        handler(req, res).catch((err) => {
            console.error("unhandled request error:", err);
            if (!res.headersSent) {
                res.writeHead(500, { "Content-Type": "application/json" });
                res.end(JSON.stringify({ error: "internal error" }));
            } else {
                res.end();
            }
        });
    });

    const port = Number(process.env.PORT) || 8787;
    server.listen(port, () => {
        console.log(`construct-harness server listening on http://localhost:${port}`);
        console.log(`  live session: ${deps.sessions.ids().join(", ")}`);
    });

    let closing = false;
    const shutdown = (signal: string) => {
        if (closing) return;
        closing = true;
        console.log(`\n${signal} received, shutting down…`);
        server.close(() => {
            deps.close();
            process.exit(0);
        });
        // If connections linger (an open SSE stream), don't hang forever.
        setTimeout(() => {
            deps.close();
            process.exit(0);
        }, 3000).unref();
    };
    process.on("SIGINT", () => shutdown("SIGINT"));
    process.on("SIGTERM", () => shutdown("SIGTERM"));
}

// Only boot when run as the entry point, so importing this module (e.g. from a
// test exercising createHandler) never starts a listener. import.meta.main is
// the Node entry-point check; the argv fallback covers runtimes without it.
const isEntry =
    (import.meta as unknown as { main?: boolean }).main ?? process.argv[1]?.endsWith("server.ts");
if (isEntry) main();
