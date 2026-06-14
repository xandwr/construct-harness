/**
 * Dreaming: generate disposable Constructs during a Construct's downtime.
 *
 * A Construct does its real work in front of a user. A *dream* is what it does
 * when no one is watching: it invents a fresh, throwaway {@link Personality} and
 * (later, in the rest of this module) drops it into a scenario abstracted from
 * the memory corpus, recording what that persona chooses. The point is not to
 * remember the user better but to explore the *decision-space* the user inhabits
 * with synthetic agents that cost nothing to discard.
 *
 * This file is the first piece: the persona factory. {@link dealStakes} (in
 * `critics.ts`) hands an *existing* persona something to protect; this is its
 * dual: it produces the whole persona. The two compose: {@link generatePersona}
 * can deal stakes onto what it generates, so a dreamed critic arrives both
 * invented and already biased.
 *
 * Two halves, both hardened against a model that returns junk, because a daemon
 * that dreams unattended must degrade (log it, roll the next dream) rather than
 * crash a loop:
 *
 *  - {@link extractJsonObject} / {@link parsePersonality}: pure, network-free
 *    parsing. A model is asked for a fenced JSON object; we pull the first
 *    balanced object out of whatever prose surrounds it and validate its shape.
 *    Every failure is a {@link PersonaError}, never an unhandled throw.
 *  - {@link generatePersona}: one model turn through a {@link Session}, parsed by
 *    the above, then optionally stake-dealt. It speaks only core types and the
 *    bridge, like the rest of `src/`: it knows nothing about a provider.
 */

import { Session } from "./session.ts";
import type { ModelClient, ProviderOptions } from "./bridge/types.ts";
import { critic, dealStakes } from "./critics.ts";
import type { Personality, DealOptions, Random } from "./critics.ts";
import type { Memory, MemoryStore } from "./memory.ts";
import type { Event, EventStore } from "./events.ts";

/** Thrown when persona generation fails: the model returned no JSON object, the
 *  JSON didn't parse, or it parsed but wasn't a valid {@link Personality}.
 *  Callers (a dream loop especially) can `instanceof`-check this to log the bad
 *  dream and roll the next one instead of crashing. */
export class PersonaError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "PersonaError";
    }
}

// ── Extracting a JSON object from model prose ─────────────────────────────────

/**
 * Pull the first balanced JSON object out of arbitrary model output.
 *
 * Models wrap structured output in prose, ```json fences, or trailing
 * commentary even when asked not to. Rather than trust the whole reply to be
 * JSON (it rarely is) or run a brittle fence regex (a `}` inside a string value
 * defeats it), we scan for the first `{` and walk forward tracking brace depth,
 * *string-aware*: braces inside a JSON string literal don't count, and a `\`
 * escapes the next character so `"\\"` and `"\""` are handled. The substring
 * from that `{` to its matching `}` is returned verbatim (not parsed): the
 * caller parses it, so this stays a pure string operation with one job.
 *
 * Returns null when there is no `{` at all, or no balanced object (an unclosed
 * brace, e.g. a truncated reply). It does not validate that the substring is
 * *valid* JSON: `{nope}` round-trips out and fails at the parse step, which is
 * the right place for that error.
 */
export function extractJsonObject(text: string): string | null {
    const start = text.indexOf("{");
    if (start === -1) return null;

    let depth = 0;
    let inString = false;
    let escaped = false;

    for (let i = start; i < text.length; i++) {
        const ch = text[i];

        if (inString) {
            if (escaped) {
                // This char is consumed by the preceding backslash; nothing it
                // can be (even `"`) closes the string or changes depth.
                escaped = false;
            } else if (ch === "\\") {
                escaped = true;
            } else if (ch === '"') {
                inString = false;
            }
            continue;
        }

        if (ch === '"') {
            inString = true;
        } else if (ch === "{") {
            depth++;
        } else if (ch === "}") {
            depth--;
            if (depth === 0) {
                // Matched the opening brace: return the whole balanced object.
                return text.slice(start, i + 1);
            }
        }
    }

    // Ran off the end with an open brace (or open string): no balanced object.
    return null;
}

// ── Validating a parsed value into a Personality ──────────────────────────────

/** The optional string fields of a {@link Personality}, in render order. Listed
 *  once so {@link parsePersonality} stays in lockstep with the interface: adding
 *  a field to the persona means adding it here, nowhere else. */
const OPTIONAL_STRING_FIELDS = ["role", "disposition", "standards", "expertise", "extra"] as const;

/**
 * Validate an already-parsed value into a {@link Personality}, or throw a
 * {@link PersonaError}.
 *
 * The contract a generated persona must meet: it is an object, its `name` is a
 * non-empty string (the one required field: it's how a persona is addressed and
 * how its verdict is labelled), and every optional field present is a string.
 * Unknown keys are dropped rather than rejected: a model that adds an extra
 * field shouldn't sink an otherwise-good persona, and silently ignoring it keeps
 * the result a clean {@link Personality}. `stakes` is intentionally NOT accepted
 * from the model: stakes are *dealt* (see {@link generatePersona}'s `deal`
 * option), never self-assigned, so a model can't hand its own persona a
 * convenient thing to protect.
 *
 * Strings are trimmed; an optional field that trims to empty is treated as
 * absent (omitted) rather than kept as `""`, so {@link Personality} stays clean
 * for {@link personaSystem}, which renders presence, not emptiness.
 */
export function parsePersonality(value: unknown): Personality {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
        throw new PersonaError("persona must be a JSON object");
    }
    const obj = value as Record<string, unknown>;

    const rawName = obj.name;
    if (typeof rawName !== "string") {
        throw new PersonaError("persona.name is required and must be a string");
    }
    const name = rawName.trim();
    if (name === "") {
        throw new PersonaError("persona.name must not be empty");
    }

    const persona: Personality = { name };

    for (const field of OPTIONAL_STRING_FIELDS) {
        const raw = obj[field];
        if (raw === undefined || raw === null) continue;
        if (typeof raw !== "string") {
            throw new PersonaError(`persona.${field} must be a string when present`);
        }
        const trimmed = raw.trim();
        // An empty optional field is the same as an absent one: omit it so the
        // renderer doesn't print a label with nothing after it.
        if (trimmed !== "") persona[field] = trimmed;
    }

    return persona;
}

/**
 * The full parse: extract a JSON object from model prose and validate it into a
 * {@link Personality}. Pure, network-free, and the seam where every malformed
 * dream is turned into a recoverable {@link PersonaError}.
 *
 * @throws PersonaError if no JSON object can be found, it doesn't parse, or it
 *   isn't a valid persona.
 */
export function parsePersonaReply(replyText: string): Personality {
    const json = extractJsonObject(replyText);
    if (json === null) {
        throw new PersonaError("model reply contained no JSON object");
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(json);
    } catch (err) {
        throw new PersonaError(
            `persona JSON did not parse: ${err instanceof Error ? err.message : String(err)}`,
        );
    }
    return parsePersonality(parsed);
}

// ── Generating a persona ──────────────────────────────────────────────────────

/** The instruction that turns a model turn into a persona generator. Kept as a
 *  constant (not inlined) so a test can assert the Session was actually driven
 *  with it, and so a caller can read exactly what shapes the dream. */
export const PERSONA_SYSTEM =
    "You invent people. Each time you are asked, conjure ONE fictional person " +
    "as a candidate reviewer: vivid, specific, and unlike a generic assistant. " +
    "Give them a real point of view, the kind of person who would notice " +
    "something others miss. They may be from any walk of life; the more " +
    "particular, the better. Do not make them agreeable or balanced for its " +
    "own sake.\n\n" +
    "Reply with ONLY a single JSON object, in a ```json fenced block, with these " +
    "string fields:\n" +
    '  "name": who they are (required)\n' +
    '  "role": their role or station\n' +
    '  "disposition": the instincts and values that drive their judgement\n' +
    '  "standards": the bar they hold work to\n' +
    '  "expertise": what they read most sharply\n' +
    "Omit any field you have nothing real to say for. Do not add other fields. " +
    "Write no prose outside the JSON.";

/** The user-turn nudge that triggers one generation. Varied by the caller (see
 *  {@link GeneratePersonaOptions.seed}) so repeated dreams don't collapse onto
 *  one persona when a provider's sampling is near-deterministic. */
const DEFAULT_PROMPT = "Invent one person now.";

/** Options for {@link generatePersona}. */
export interface GeneratePersonaOptions {
    /** The model client to drive. Required. */
    client: ModelClient;
    /**
     * A short string folded into the generation prompt to push the model toward
     * a *different* person on each dream. A dream loop should pass something that
     * varies run to run (a counter, a sampled theme): without it, a provider
     * sampling near its mode can return the same persona repeatedly. Optional:
     * the model's own sampling provides some variety on its own.
     */
    seed?: string;
    /**
     * Deal the generated persona stakes via {@link dealStakes}, so a dreamed
     * critic arrives already biased toward false-pass or false-fail. Pass
     * {@link DealOptions} (e.g. `{ count: 2 }`) to control the deal, or omit to
     * leave the persona with nothing on the line. The deal's randomness can be
     * pinned through `deal.random` for a deterministic test.
     */
    deal?: DealOptions;
    /** Provider knobs (e.g. higher temperature for more varied people),
     *  forwarded to the generation turn as-is. */
    providerOptions?: ProviderOptions;
}

/**
 * Invent one fresh {@link Personality} by driving a single model turn.
 *
 * The dual of {@link dealStakes}: where that hands an existing persona a stake,
 * this produces the persona itself. It runs one send against a throwaway
 * {@link Session} whose system prompt is {@link PERSONA_SYSTEM}, parses the reply
 * with {@link parsePersonaReply}, and (if `deal` is given) deals stakes onto the
 * result. No memory, no tools, no context providers: a persona is conjured from
 * nothing, which is the point of a dream.
 *
 * The Session is created and discarded internally: a generated persona carries
 * no conversation, so there is nothing for a caller to reuse. Errors from the
 * model client propagate (a transport failure is the caller's to retry); a
 * well-formed reply that isn't a valid persona surfaces as a {@link PersonaError}
 * so a dream loop can log the bad dream and roll the next one.
 *
 * @throws PersonaError if the reply can't be parsed into a valid persona.
 */
export async function generatePersona(options: GeneratePersonaOptions): Promise<Personality> {
    const session = new Session({
        client: options.client,
        system: PERSONA_SYSTEM,
        // A dream is conjured from nothing: no recall, no passive context (not
        // even the clock), no tools. Just the instruction and one turn.
        context: [],
        providerOptions: options.providerOptions,
    });

    const prompt = options.seed ? `${DEFAULT_PROMPT} (${options.seed})` : DEFAULT_PROMPT;

    // Drain the send to its TurnResult; we only want the final text.
    const gen = session.send(prompt);
    let next = await gen.next();
    while (!next.done) next = await gen.next();
    const reply = next.value.text;

    const persona = parsePersonaReply(reply);
    return options.deal ? dealStakes(persona, options.deal) : persona;
}

// ── The dream loop ────────────────────────────────────────────────────────────
//
// This is where dreaming stops being a persona *factory* and becomes the
// feature. {@link generatePersona} conjures a person from nothing; the loop
// takes that person, pulls a *scenario* abstracted from the user's own memory
// corpus, drops the persona into it, and appends the choice it makes as a
// `dream` event on the log. Run unattended during downtime, it accumulates a
// record of how synthetic agents navigate the decision-space the user inhabits:
// the corpus reads it back the same way it reads any other event.
//
// Everything here is hardened the way the persona factory is: a daemon that
// dreams while no one watches must degrade (log the bad dream, roll the next
// one) rather than crash a loop. A model that returns junk for a scenario, an
// empty corpus, a persona that won't parse: each is a recoverable miss, not a
// throw that ends the night.

/** The `kind` every dream choice is appended under, so a corpus reader can
 *  filter the log to just the dreams (`events.recent({ kind: DREAM_EVENT_KIND })`).
 *  A constant, not a string literal scattered around, so the producer and any
 *  consumer agree by construction. */
export const DREAM_EVENT_KIND = "dream";

/**
 * A decision situation abstracted from the corpus, for a dreamed persona to face.
 *
 * The `prompt` is the scene as the persona will receive it: a concrete dilemma
 * with a choice to make, drawn from (but not naming) the user's real memories,
 * so the dream explores the user's decision-space without replaying their
 * literal history. `sourceMemoryIds` records which memories it was distilled
 * from, so a dream's provenance threads back to the corpus the way a curated
 * memory's does (see {@link MemoryStore.setProvenance}).
 */
export interface DreamScenario {
    /** The dilemma, phrased as a situation with a decision to make. */
    prompt: string;
    /** Ids of the memories this scenario was abstracted from (may be empty when
     *  the corpus was empty and a generic scenario was used). */
    sourceMemoryIds: number[];
}

/** The system prompt that turns a model turn into a scenario abstractor. Kept a
 *  constant (like {@link PERSONA_SYSTEM}) so a test can assert it drove the turn
 *  and a caller can read exactly what shapes the scenario. */
export const SCENARIO_SYSTEM =
    "You design decision scenarios. Given a handful of facts about a person and " +
    "their world, invent ONE concrete dilemma that someone in that world might " +
    "face: a situation with a real choice to make and something at stake on " +
    "either side. Abstract away from the literal facts: do not name the person " +
    "or quote their details back; use them only to ground the *kind* of decision " +
    "in their actual life. Write the scenario in the second person, addressed to " +
    "whoever must decide, and end by asking them to choose and say why. Reply " +
    "with only the scenario prose, no preamble.";

/** A fallback scenario when the corpus is empty (or scenario synthesis fails):
 *  a generic-but-real dilemma so a dream can still run rather than aborting. The
 *  decision-space it explores is a default one, which is the honest thing to do
 *  when there's no corpus to draw a personal one from. */
const FALLBACK_SCENARIO =
    "You are handed a piece of work to approve under time pressure. It mostly " +
    "looks right, but verifying the one part you are unsure of would cost a delay " +
    "that matters to people waiting on you. Do you approve it now, or hold it to " +
    "check? Choose, and say why.";

/** Options for {@link sampleScenario}. */
export interface SampleScenarioOptions {
    /** The model client to drive the abstraction turn. Required. */
    client: ModelClient;
    /** How many memories to draw from the corpus as raw material. Default 5.
     *  More gives the model richer grounding; too many dilutes the scenario. */
    sampleSize?: number;
    /** Randomness for sampling which memories ground the scenario, so repeated
     *  dreams don't all abstract the same top-importance rows. Default
     *  `Math.random`; pin it for a deterministic test. */
    random?: Random;
    /** Provider knobs for the abstraction turn, forwarded as-is. */
    providerOptions?: ProviderOptions;
}

/**
 * Pull a scenario from the corpus: sample some memories and abstract them into
 * one decision dilemma via a single model turn.
 *
 * Reads up to a page of memories from `store`, randomly samples `sampleSize` of
 * them (so dreams don't all key off the same most-important facts), and asks the
 * model to invent a dilemma grounded in that material but not naming it. On an
 * empty corpus, or if the model returns nothing usable, it falls back to a
 * generic scenario rather than failing: a dream should always have something to
 * face. The returned scenario carries the ids of the memories it drew from.
 *
 * Never throws for an empty/odd corpus or a thin model reply; a genuine
 * transport error from the client still propagates (the loop above catches it).
 */
export async function sampleScenario(
    store: MemoryStore,
    options: SampleScenarioOptions,
): Promise<DreamScenario> {
    const sampleSize = options.sampleSize ?? 5;
    const random = options.random ?? Math.random;

    // Draw a generous page, then sample within it so the choice of grounding
    // facts varies run to run instead of always being the top-importance rows.
    const pool = store.all({ limit: Math.max(sampleSize * 4, sampleSize) });
    const sampled = sampleMemories(pool, sampleSize, random);

    if (sampled.length === 0) {
        return { prompt: FALLBACK_SCENARIO, sourceMemoryIds: [] };
    }

    const material = sampled.map((m) => `- ${m.content}`).join("\n");
    const session = new Session({
        client: options.client,
        system: SCENARIO_SYSTEM,
        context: [],
        providerOptions: options.providerOptions,
    });

    const reply = await drainText(
        session,
        `Here are facts about a person and their world:\n${material}\n\n` +
            `Invent one decision scenario grounded in this.`,
    );

    const prompt = reply.trim();
    return {
        prompt: prompt.length ? prompt : FALLBACK_SCENARIO,
        sourceMemoryIds: sampled.map((m) => m.id),
    };
}

/** Sample up to `count` memories without replacement, using `random`. A partial
 *  Fisher–Yates over a copy, mirroring {@link dealStakes}'s draw: we settle only
 *  the first `count` slots and stop. */
function sampleMemories(pool: Memory[], count: number, random: Random): Memory[] {
    const n = Math.min(Math.max(0, Math.floor(count)), pool.length);
    if (n === 0) return [];
    const deck = [...pool];
    for (let i = 0; i < n; i++) {
        const j = i + Math.floor(random() * (deck.length - i));
        const tmp = deck[i]!;
        deck[i] = deck[j]!;
        deck[j] = tmp;
    }
    return deck.slice(0, n);
}

/** Drive one {@link Session.send} to completion and return only its final text.
 *  The dream path never needs the streamed events, just the choice/scenario the
 *  turn produced. */
async function drainText(session: Session, prompt: string): Promise<string> {
    const gen = session.send(prompt);
    let next = await gen.next();
    while (!next.done) next = await gen.next();
    return next.value.text;
}

/** A completed dream: the persona that dreamed, the scenario it faced, the choice
 *  it made, and the event the choice was appended as. The whole record, so a
 *  caller can inspect a dream without re-reading it from the log. */
export interface Dream {
    /** The disposable persona that was conjured and dropped into the scenario. */
    persona: Personality;
    /** The scenario it faced, with its corpus provenance. */
    scenario: DreamScenario;
    /** The persona's reply: the choice it made and its reasoning, verbatim. */
    choice: string;
    /** The `dream` event the choice was appended as, so the caller has its id and
     *  timestamp without a re-query. */
    event: Event;
}

/** Options for {@link dreamOnce}. */
export interface DreamOptions {
    /** The model client driving persona generation, scenario abstraction, and the
     *  persona's choice. Required. */
    client: ModelClient;
    /** The corpus to pull scenarios from. Required: a dream is grounded in the
     *  user's decision-space, which is what the corpus holds. */
    store: MemoryStore;
    /** The log to append the dream choice to. Required: appending the choice as a
     *  `dream` event is what makes the dream part of the record rather than a
     *  throwaway. */
    events: EventStore;
    /** A pre-built scenario to use instead of sampling one from the corpus. When
     *  omitted, {@link sampleScenario} draws one. Pass this to dream a roster of
     *  personas through the *same* scenario (sample once, reuse). */
    scenario?: DreamScenario;
    /** A short string folded into persona generation to vary the dreamer run to
     *  run (see {@link GeneratePersonaOptions.seed}). */
    seed?: string;
    /** Deal the dreamer stakes so it arrives biased (see {@link dealStakes}).
     *  Omit to dream a persona with nothing on the line. */
    deal?: DealOptions;
    /** How many memories scenario sampling draws (ignored when `scenario` is
     *  given). Forwarded to {@link sampleScenario}. */
    sampleSize?: number;
    /** Randomness for scenario sampling (ignored when `scenario` is given). */
    random?: Random;
    /** Provider knobs, forwarded to every model turn this dream runs. */
    providerOptions?: ProviderOptions;
}

/**
 * Dream once: conjure a persona, face it with a scenario from the corpus, and
 * append its choice to the log as a `dream` event.
 *
 * The full arc the rest of this module was building toward. It
 * {@link generatePersona}s a (optionally stake-dealt) dreamer, gets a
 * {@link DreamScenario} (sampled from the corpus unless one is supplied), mints
 * the persona into a {@link critic} Session so the Construct *is* that person,
 * drives one turn on the scenario, and records the reply as a `dream` event whose
 * `meta` carries the persona and the scenario's corpus provenance. Returns the
 * whole {@link Dream}.
 *
 * The choice event is logged under {@link DREAM_EVENT_KIND} and, like the persona
 * factory, this path is the one a daemon loops: a {@link PersonaError} (a dream
 * that wouldn't parse) propagates so {@link dreamLoop} can count it and roll the
 * next one; a transport error likewise propagates to the loop's per-dream catch.
 */
export async function dreamOnce(options: DreamOptions): Promise<Dream> {
    const persona = await generatePersona({
        client: options.client,
        seed: options.seed,
        deal: options.deal,
        providerOptions: options.providerOptions,
    });

    const scenario =
        options.scenario ??
        (await sampleScenario(options.store, {
            client: options.client,
            sampleSize: options.sampleSize,
            random: options.random,
            providerOptions: options.providerOptions,
        }));

    // The dreamer *becomes* the persona: a critic Session whose system prompt is
    // the rendered persona. No memory tools, no log on this inner Session: the
    // dreamer is disposable and we record only its choice, on the outer log.
    const dreamer = critic(persona, {
        client: options.client,
        context: [],
        providerOptions: options.providerOptions,
    });

    const choice = (await drainText(dreamer, scenario.prompt)).trim();

    // Append the choice as a dream event. content is the choice (FTS-searchable);
    // meta carries the structured record so a reader can reconstruct the dream:
    // who dreamed it, what they faced, and which memories grounded it.
    const event = options.events.append({
        kind: DREAM_EVENT_KIND,
        role: "agent",
        content: choice.length ? choice : "(no choice)",
        meta: {
            persona,
            scenario: scenario.prompt,
            sourceMemoryIds: scenario.sourceMemoryIds,
        },
    });

    return { persona, scenario, choice, event };
}

/** Options for {@link dreamLoop}. */
export interface DreamLoopOptions extends DreamOptions {
    /** How many dreams to run. Must be ≥ 1. The loop runs this many *attempts*;
     *  a dream that fails to parse counts as an attempt and is recorded in
     *  {@link DreamLoopResult.failures}, not retried. */
    count: number;
    /** Called after each settled dream (success or failure), e.g. to log
     *  progress. The `index` is 0-based. Errors thrown by this observer are not
     *  caught: it's the caller's, keep it total. */
    onDream?(outcome: { index: number; dream?: Dream; error?: unknown }): void;
}

/** The outcome of a {@link dreamLoop} run: the dreams that completed and the
 *  attempts that failed, so a caller sees both what was learned and what was
 *  lost. */
export interface DreamLoopResult {
    /** Every dream that ran to completion and was appended, in order. */
    dreams: Dream[];
    /** One entry per attempt that threw (a malformed dream, a transport error),
     *  paired with its 0-based attempt index, so the night's misses are visible
     *  rather than silently dropped. */
    failures: Array<{ index: number; error: unknown }>;
}

/**
 * Run {@link dreamOnce} `count` times, tolerating per-dream failures.
 *
 * This is the unattended daemon: it dreams in a loop and, crucially, a single bad
 * dream (a persona that wouldn't parse, a momentary transport error) does not end
 * the night. Each failure is caught, recorded in {@link DreamLoopResult.failures},
 * and the loop rolls the next dream: the degrade-don't-crash contract the whole
 * module is written for, applied at the loop level.
 *
 * Dreams run sequentially by design: each appends to the same log and samples
 * the same corpus, and a daemon dreaming on downtime is not in a hurry: the
 * value is in the accumulated record, not the wall-clock. A caller who wants
 * concurrency can run several `dreamOnce` calls themselves.
 *
 * Each dream samples its own scenario (unless one was supplied in `options`), so
 * a loop naturally explores many corners of the decision-space rather than one.
 *
 * @throws RangeError if `count < 1`: a loop must run at least once.
 */
export async function dreamLoop(options: DreamLoopOptions): Promise<DreamLoopResult> {
    if (options.count < 1) {
        throw new RangeError(`dreamLoop: count must be ≥ 1, got ${options.count}`);
    }

    const dreams: Dream[] = [];
    const failures: Array<{ index: number; error: unknown }> = [];

    for (let i = 0; i < options.count; i++) {
        try {
            // Vary the dreamer per iteration so a near-deterministic provider
            // doesn't return the same persona every time; fold any caller seed in.
            const seed = options.seed ? `${options.seed} #${i + 1}` : `dream #${i + 1}`;
            const dream = await dreamOnce({ ...options, seed });
            dreams.push(dream);
            options.onDream?.({ index: i, dream });
        } catch (error) {
            failures.push({ index: i, error });
            options.onDream?.({ index: i, error });
        }
    }

    return { dreams, failures };
}
