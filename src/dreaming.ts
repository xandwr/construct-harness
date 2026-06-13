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
import { dealStakes } from "./critics.ts";
import type { Personality, DealOptions } from "./critics.ts";

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
