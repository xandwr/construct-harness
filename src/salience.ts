/**
 * Salience mining: finding the topics the Construct keeps raising *on its own*.
 *
 * The working mind has a `concern` band (see {@link ./workingMind.ts}) for things
 * the Construct returns to unprompted across conversations. The honest difficulty
 * the band was deferred for is that the harness *can't author* a concern — a
 * summarizer writing "you are worried about X" would be the harness inventing the
 * Construct's interior, the exact failure the working mind exists to avoid. But
 * the *evidence* of a concern is already on the log: it's a topic that appears in
 * the Construct's own messages, that the user's preceding message didn't
 * introduce, recurring across many sessions. The harness can faithfully *identify
 * candidates* from that evidence; it just can't decide they're real.
 *
 * That division is the whole design, and it's what keeps this honest:
 *  - This module *mines candidates*: it scans recent agent messages, for each
 *    extracts the phrases the user's immediately-preceding turn did NOT contain
 *    (so a topic the user raised is excluded — that's prompted, not a concern),
 *    and counts how many distinct sessions each unprompted phrase recurs in.
 *    A phrase recurring unprompted in {@link DEFAULT_MIN_SESSIONS}+ sessions is a
 *    candidate.
 *  - The Construct *decides*: a candidate is `note`d into the mind's concern band
 *    when a session starts, but the band decays by recency+reinforcement like
 *    every other. If the Construct keeps raising it, it stays warm; if it stops,
 *    it cools and slips out on its own. The harness never writes the band
 *    directly and never refreshes a concern's warmth — only the Construct
 *    continuing to bring it up does. So a mis-mined candidate simply fades, and a
 *    real concern persists exactly as long as it's real.
 *
 * The mining is deliberately a lexical, deterministic proxy, not a model call: it
 * runs during downtime (the daemon) and produces only *candidate phrases lifted
 * verbatim from the Construct's own words*. It does not paraphrase, summarize, or
 * label — the text of a concern is the Construct's noun phrase as it wrote it, so
 * even the candidate is the Construct's, not the harness's. A semantic version
 * (embed agent vs prior-user message, score the novel terms) is a possible later
 * refinement; the lexical proxy is enough to surface a topic that keeps coming up.
 *
 * Speaks only core types and the {@link EventStore}'s read surface. No embedder,
 * no model, no provider: pure text over log rows.
 */

import { EventStore } from "./events.ts";
import type { Event } from "./events.ts";

/** How many distinct sessions an unprompted phrase must recur in to count as a
 *  concern candidate. Three, matching the design: twice could be coincidence, a
 *  third unprompted recurrence across a *different* conversation is the signal of
 *  something the Construct carries rather than something one chat happened to be
 *  about. */
export const DEFAULT_MIN_SESSIONS = 3;

/** How many recent agent messages to scan per mining pass. Bounded so the pass
 *  stays cheap on a long log; the daemon runs it during downtime, not on a turn's
 *  hot path, but it's still a linear scan we keep modest. */
export const DEFAULT_SCAN = 400;

/** The longest a candidate phrase may be (chars). A concern is a topic — a noun
 *  phrase — not a sentence; cap it so a run-on clause that slipped the phrase
 *  splitter can't become a "concern". */
const MAX_PHRASE_CHARS = 48;

/** Words too generic to anchor a concern on their own. A concern phrase made
 *  entirely of these carries no topic, so it's dropped. Deliberately small: the
 *  multi-session recurrence test does most of the filtering; this just removes
 *  the obvious filler that would otherwise recur trivially. */
const STOPWORDS: ReadonlySet<string> = new Set([
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "them",
    "your",
    "my",
    "our",
    "their",
    "me",
    "us",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
    "must",
    "have",
    "has",
    "had",
    "not",
    "no",
    "yes",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "there",
    "here",
    "just",
    "about",
    "into",
    "than",
    "too",
    "very",
    "more",
    "most",
    "some",
    "any",
    "all",
    "one",
    "out",
    "up",
    "down",
    "over",
    "also",
    "like",
    "get",
    "got",
    "thing",
    "things",
    "way",
    "ways",
    "lot",
    "bit",
    "kind",
    "sort",
]);

/** A mined concern candidate: the phrase (the Construct's own words, lowercased
 *  for identity), the number of distinct sessions it recurred *unprompted* in,
 *  and a representative verbatim form (the first surface spelling seen) to show
 *  in the mind. */
export interface ConcernCandidate {
    /** Normalized phrase identity (lowercased, whitespace-folded). */
    phrase: string;
    /** The verbatim form first seen, shown to the Construct (the band renders
     *  this, not the lowercased identity). */
    text: string;
    /** Distinct sessions this phrase appeared in, unprompted. The recurrence
     *  signal: ≥ the threshold to be a candidate. */
    sessions: number;
}

/** Options for {@link mineConcerns}. */
export interface MineConcernsOptions {
    /** Distinct-session recurrence threshold. Default {@link DEFAULT_MIN_SESSIONS}. */
    minSessions?: number;
    /** How many recent agent messages to scan. Default {@link DEFAULT_SCAN}. */
    scan?: number;
}

/** Split text into lowercased word tokens (letters, digits, apostrophes,
 *  hyphens within words). The unit both phrase extraction and the "did the user
 *  say it" test work in. */
function tokenize(text: string): string[] {
    const matches = text.toLowerCase().match(/[a-z0-9][a-z0-9'-]*/g);
    return matches ?? [];
}

/** Fold a phrase to its stable identity for counting (lowercased, whitespace-
 *  collapsed, trimmed). */
function normalizePhrase(text: string): string {
    return text.toLowerCase().replace(/\s+/g, " ").trim();
}

/**
 * Extract candidate noun-ish phrases from one agent message: runs of content
 * words (2–4 tokens) between stopword/punctuation breaks, plus salient single
 * content words. A blunt proxy for noun phrases — we don't POS-tag — but the
 * multi-session recurrence test downstream is what actually decides a concern, so
 * over-generating here is fine: noise that doesn't recur across sessions is
 * dropped anyway.
 */
export function extractPhrases(text: string): string[] {
    const phrases: string[] = [];
    // Split on punctuation into clauses, then within each clause grow runs of
    // non-stopword tokens.
    const clauses = text.toLowerCase().split(/[.,;:!?()[\]{}"“”'`\n]+/);
    for (const clause of clauses) {
        const tokens = (clause.match(/[a-z0-9][a-z0-9'-]*/g) ?? []).filter(Boolean);
        let run: string[] = [];
        const flush = () => {
            // Drop a leading/trailing stopword-only edge isn't needed: a run never
            // contains a stopword (we break on them). Emit 1–4 token windows.
            for (let len = Math.min(run.length, 4); len >= 1; len--) {
                for (let i = 0; i + len <= run.length; i++) {
                    const phrase = run.slice(i, i + len).join(" ");
                    if (phrase.length <= MAX_PHRASE_CHARS) phrases.push(phrase);
                }
            }
            run = [];
        };
        for (const tok of tokens) {
            if (STOPWORDS.has(tok) || tok.length < 3) {
                flush();
            } else {
                run.push(tok);
            }
        }
        flush();
    }
    // A lone stopword phrase can't happen (we break on them); but a single short
    // token can slip through length-3 filter as e.g. "the"→no. Keep only phrases
    // with at least one ≥4-char content token so "ai", "ok" don't become concerns.
    return phrases.filter((p) => p.split(" ").some((t) => t.length >= 4));
}

/**
 * The events the miner reads, paired so each agent message knows the user turn
 * that immediately preceded it *in the same session*. Built by walking a
 * session's events in order and pairing each agent message with the last user
 * message before it.
 */
interface AgentTurn {
    session: string;
    agentText: string;
    /** Tokens of the user message right before this agent message in the session,
     *  for the "was this prompted?" test. Empty when the agent spoke first. */
    priorUserTokens: Set<string>;
}

/**
 * Pair agent messages with their preceding user message, per session. Walks the
 * given events oldest-first within each session, tracking the most recent user
 * message so each agent message can be tested against what the user had just
 * said. Events with no session are ignored (a concern is a cross-conversation
 * signal; an unscoped event has no conversation to attribute it to).
 */
function pairTurns(events: Event[]): AgentTurn[] {
    // Group by session, preserving order. `events` arrives newest-first from
    // recent(); reverse per session so we walk each conversation forward.
    const bySession = new Map<string, Event[]>();
    for (const e of events) {
        if (e.kind !== "message" || !e.session) continue;
        const list = bySession.get(e.session) ?? [];
        list.push(e);
        bySession.set(e.session, list);
    }
    const turns: AgentTurn[] = [];
    for (const [session, list] of bySession) {
        // recent() is newest-first; reverse to reading order.
        list.reverse();
        let priorUser: Set<string> = new Set();
        for (const e of list) {
            if (e.role === "user") {
                priorUser = new Set(tokenize(e.content));
            } else {
                // agent (or any non-user) message: a turn the Construct produced.
                turns.push({ session, agentText: e.content, priorUserTokens: priorUser });
            }
        }
    }
    return turns;
}

/** Whether a phrase was *prompted* by the preceding user turn: true when every
 *  content token of the phrase appears in the user's message. A phrase the user
 *  introduced isn't a concern of the Construct's, it's a response to the user. */
function wasPrompted(phrase: string, priorUserTokens: Set<string>): boolean {
    if (priorUserTokens.size === 0) return false; // agent spoke first: not prompted.
    const tokens = phrase.split(" ");
    return tokens.every((t) => priorUserTokens.has(t));
}

/**
 * Mine concern candidates from the log: phrases the Construct raised that the
 * user's immediately-preceding turn didn't, recurring across at least
 * `minSessions` distinct conversations.
 *
 * The algorithm, all lexical and deterministic:
 *  1. Read the recent message events, pair each agent message with the user
 *     message before it in the same session.
 *  2. For each agent message, extract candidate phrases and drop the ones the
 *     prior user turn introduced (those are prompted).
 *  3. For each surviving phrase, record the *distinct session* it appeared in and
 *     keep the first verbatim form seen.
 *  4. Return the phrases whose distinct-session count meets the threshold,
 *     strongest (most sessions) first.
 *
 * Counting distinct sessions (not raw occurrences) is the load-bearing choice: a
 * phrase the Construct repeats ten times in one conversation is that
 * conversation's topic, not a standing concern; the same phrase surfacing
 * unprompted in three *different* conversations is. Best-effort: a store read
 * failure yields an empty candidate list rather than throwing (the daemon must
 * degrade, not crash).
 */
export function mineConcerns(
    store: EventStore,
    options: MineConcernsOptions = {},
): ConcernCandidate[] {
    const minSessions = options.minSessions ?? DEFAULT_MIN_SESSIONS;
    const scan = options.scan ?? DEFAULT_SCAN;

    let events: Event[];
    try {
        events = store.recent({ kind: "message", limit: scan });
    } catch {
        return [];
    }

    const turns = pairTurns(events);

    // phrase identity → { sessions it appeared in (unprompted), first verbatim form }
    const acc = new Map<string, { sessions: Set<string>; text: string }>();
    for (const turn of turns) {
        // Distinct phrases in this one message, so repeating a phrase within a
        // single message doesn't inflate anything (we add the session, a set).
        const seen = new Set<string>();
        for (const raw of extractPhrases(turn.agentText)) {
            const id = normalizePhrase(raw);
            if (!id || seen.has(id)) continue;
            if (wasPrompted(id, turn.priorUserTokens)) continue;
            seen.add(id);
            const entry = acc.get(id) ?? { sessions: new Set<string>(), text: id };
            entry.sessions.add(turn.session);
            acc.set(id, entry);
        }
    }

    const candidates: ConcernCandidate[] = [];
    for (const [phrase, entry] of acc) {
        if (entry.sessions.size >= minSessions) {
            candidates.push({ phrase, text: entry.text, sessions: entry.sessions.size });
        }
    }
    // Strongest first (most distinct sessions), then alphabetically for a stable
    // order among ties (determinism: no Date/random in this module).
    candidates.sort((a, b) => b.sessions - a.sessions || a.phrase.localeCompare(b.phrase));
    return candidates;
}
