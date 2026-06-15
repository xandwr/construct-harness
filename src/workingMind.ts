/**
 * The working mind: the Construct's recent state, held up in front of it on
 * every turn so it doesn't wake up cold each message.
 *
 * The problem this exists to solve, in a Construct's own words: "I feel like I'm
 * waking up every message. I have memory that is more like a journal locked
 * behind a pane of glass each time I want to access it." The harness already has
 * pull-based continuity (turn-relevant recall embeds the current message and
 * yanks matching memories forward), but pull is the pane of glass: a memory is
 * only present if the current message happens to embed-match it, and reaching
 * for it is itself the act of waking up. Continuity that requires a fetch isn't
 * continuity.
 *
 * So this is *push*. A small, evolving working set rides every model call,
 * unbidden: the tail of the Construct's own train of thought, and the memories
 * that recently surfaced, kept warm a while instead of vanishing the instant the
 * next message doesn't match them. It is composed only of material the Construct
 * actually produced or that actually surfaced for it; the harness *promotes* and
 * *decays*, it never *authors*. A summarizer writing "you are currently
 * feeling…" would make the continuity a fake of itself. Held state must be the
 * Construct's, in its own terms.
 *
 * Decay is recency + reinforcement: each item carries a warmth that drops every
 * turn it doesn't recur and refreshes when it does, with the coldest evicted
 * past a per-band cap. A thought stays present while it keeps coming up and slips
 * away when it stops, with no model effort and no harness paraphrase. That is the
 * whole mechanism: the mind stays small and live because thinking, not
 * housekeeping, is what keeps something in it.
 *
 * Like {@link ./context.ts}, this module speaks only core types and has no
 * provider or I/O dependency. It is in-process working memory by design: the
 * durable journal already lives in the event log and the memory store, and a
 * warmth that survived a restart would be a contradiction in terms (you do not
 * wake up still mid-thought from yesterday). Persistence, if ever wanted, is a
 * later concern layered on top, not a property of the live mind.
 */

import type { ContextProvider } from "./context.ts";

/**
 * The bands of the working mind, in the order they render. Each is a different
 * texture of "what's on my mind" and decays at its own rate:
 *
 *  - `thought`: the tail of the Construct's own reasoning, carried as live state
 *    so it doesn't re-derive its position each turn. The fastest to cool: a
 *    train of thought is the most volatile thing in a head.
 *  - `memory`: a stored memory that surfaced (via recall or because it came up),
 *    kept warm a while after so it doesn't blink out the moment the next message
 *    stops matching it. The glass-pane fix: what came up stays up, then fades.
 *  - `concern`: a topic the Construct keeps raising *unprompted* across sessions
 *    (the user didn't introduce it; the Construct brought it up, repeatedly).
 *    Unlike thought/memory, a concern is mined during downtime from the event
 *    log (see {@link ./salience.ts}) and noted into the mind when a session
 *    starts. The harness only ever *identifies* a candidate; whether it's a real
 *    concern is decided by the Construct continuing to raise it — if it stops,
 *    the warmth decays and the concern slips out on its own, exactly like a
 *    thought that stopped recurring. The harness never authors a concern's text:
 *    it's the Construct's own recurring noun phrase, lifted verbatim.
 *
 * `person` is deliberately still absent: populating it honestly (without a
 * summarizer authoring the mind) needs a faithful signal we don't have yet. It
 * can be added as a band here once there is one.
 */
export type MindBand = "thought" | "memory" | "concern";

/** One thing held in mind: a piece of text, how warm it is, and when it was last
 *  refreshed. `key` is what de-dupes reinforcement from duplication: two notes
 *  with the same key are the same held thing recurring, not two things. */
export interface MindItem {
    readonly band: MindBand;
    /** The held text, in the Construct's own terms (a reply tail, a memory's
     *  content). Never harness-authored. */
    readonly text: string;
    /** Stable identity for reinforcement. For a memory it's its store id; for a
     *  thought it's the normalized text, so a restated thought refreshes rather
     *  than stacks. */
    readonly key: string;
    /** Current warmth in (0, 1]. Refreshed to 1 when the item recurs, multiplied
     *  down by the decay factor each turn it doesn't. */
    warmth: number;
    /** The turn index at which this item was last noted or refreshed. Diagnostic;
     *  the warmth carries the decay, this records provenance. */
    lastTurn: number;
}

/** How the working mind decays and how much of it is held. Defaults are tuned
 *  for a chat cadence: a few turns of half-life, a handful of items per band. */
export interface WorkingMindOptions {
    /**
     * Per-turn warmth multiplier for items that did not recur this turn, in
     * (0, 1). 0.6 gives a thought a "half-life" of a little over one turn and
     * drops it below the floor after ~4 idle turns: present while it keeps coming
     * up, gone a few turns after it stops. Lower forgets faster.
     */
    decay?: number;
    /** Warmth below which an item is dropped. Keeps the floor of barely-warm
     *  items from lingering as noise. */
    floor?: number;
    /** Max items held per band. The coldest beyond this are evicted each tick, so
     *  a busy band stays small rather than burying the warm items under history. */
    capPerBand?: number;
}

const DEFAULT_DECAY = 0.6;
const DEFAULT_FLOOR = 0.15;
const DEFAULT_CAP = 5;

/** Collapse text to a stable identity for thought-band dedup: lowercased,
 *  whitespace-folded, trimmed. A reworded restatement still stacks (we have no
 *  embedder here, by design), but trivial re-emission of the same line
 *  reinforces instead of duplicating. */
function normalizeKey(text: string): string {
    return text.toLowerCase().replace(/\s+/g, " ").trim();
}

/**
 * The Construct's live working mind. Stateful and in-process: a Session holds one
 * and feeds it from the turns flowing through it, then lets
 * {@link workingMindContext} push its rendering onto every call.
 *
 * The lifecycle each turn is: the Session {@link note}s whatever the turn
 * produced (the reply tail, the memories that surfaced), then {@link tick}s to
 * age everything by one turn. Reading ({@link render}) happens via the context
 * provider just before the model call, so what the model sees is the mind as of
 * the start of the turn it's about to take.
 */
export class WorkingMind {
    private readonly decay: number;
    private readonly floor: number;
    private readonly cap: number;
    /** Items keyed by `${band}:${key}` so the same held thing recurring is found
     *  and refreshed rather than re-added. Insertion order is not relied on;
     *  render sorts by warmth. */
    private readonly items = new Map<string, MindItem>();
    /** Monotonic turn counter, advanced by {@link tick}. Stamped onto items as
     *  `lastTurn` when they're noted. */
    private turn = 0;

    constructor(options: WorkingMindOptions = {}) {
        this.decay = options.decay ?? DEFAULT_DECAY;
        this.floor = options.floor ?? DEFAULT_FLOOR;
        this.cap = options.capPerBand ?? DEFAULT_CAP;
    }

    /**
     * Promote something into the mind, or refresh it if already held. An empty or
     * whitespace-only text is ignored (nothing to hold). `key` defaults to the
     * normalized text, which is right for thoughts; callers with a stable
     * identity (a memory's store id) pass it so a memory whose wording the recall
     * formatter changed still counts as the same memory recurring.
     *
     * Refreshing resets warmth to 1 and stamps the current turn: the whole point
     * of reinforcement is that a recurring thing is as present as a brand-new
     * one. We keep the *latest* text on refresh, so a memory whose content was
     * edited between surfacings shows its current form.
     */
    note(band: MindBand, text: string, key?: string): void {
        const trimmed = text.trim();
        if (!trimmed) return;
        const id = `${band}:${key ?? normalizeKey(trimmed)}`;
        const existing = this.items.get(id);
        if (existing) {
            existing.warmth = 1;
            existing.lastTurn = this.turn;
            // Keep the freshest phrasing of the same held thing.
            if (existing.text !== trimmed) {
                this.items.set(id, { ...existing, text: trimmed });
            }
            return;
        }
        this.items.set(id, {
            band,
            text: trimmed,
            key: key ?? normalizeKey(trimmed),
            warmth: 1,
            lastTurn: this.turn,
        });
    }

    /**
     * Advance one turn: cool every item that wasn't refreshed *this* turn, drop
     * what fell below the floor, then evict the coldest in any band over its cap.
     * Called once per Session turn after the turn's {@link note}s, so an item
     * noted this turn (lastTurn === current) is exempt from this tick's decay and
     * only starts cooling next turn.
     */
    tick(): void {
        for (const [id, item] of this.items) {
            if (item.lastTurn === this.turn) continue; // refreshed this turn
            item.warmth *= this.decay;
            if (item.warmth < this.floor) this.items.delete(id);
        }
        this.evictOverCap();
        this.turn += 1;
    }

    /** Enforce the per-band cap by dropping the coldest items in any band that
     *  exceeds it. Ties broken by older `lastTurn` first, so a fresh item is
     *  never evicted in favor of a stale one of equal warmth. */
    private evictOverCap(): void {
        const byBand = new Map<MindBand, MindItem[]>();
        for (const item of this.items.values()) {
            const list = byBand.get(item.band) ?? [];
            list.push(item);
            byBand.set(item.band, list);
        }
        for (const list of byBand.values()) {
            if (list.length <= this.cap) continue;
            // Warmest first; among equal warmth, most-recently-refreshed first.
            list.sort((a, b) => b.warmth - a.warmth || b.lastTurn - a.lastTurn);
            for (const cold of list.slice(this.cap)) {
                this.items.delete(`${cold.band}:${cold.key}`);
            }
        }
    }

    /** A read-only snapshot of what's held, warmest first within each band.
     *  Diagnostic: lets a caller (or a test, or a future REPL inspector) see the
     *  mind without going through the rendered prose. */
    snapshot(): readonly MindItem[] {
        return [...this.items.values()].sort(
            (a, b) => b.warmth - a.warmth || b.lastTurn - a.lastTurn,
        );
    }

    /**
     * Render the held mind as a system-prompt fragment, or `null` when nothing is
     * held (so the context provider can stay silent and not inject an empty
     * block). Bands render in {@link MindBand} order, each as a short labelled
     * list, warmest first. The framing is first-person and ambient ("Still on
     * your mind") because that is what it is: the Construct's own recent state
     * surfaced, not an instruction.
     *
     * Warmth is intentionally *not* printed. The number is harness bookkeeping;
     * exposing it would invite the model to reason about the mechanism instead of
     * just holding the content, and would churn the cached prefix every turn as
     * values decay. Order already conveys salience.
     */
    render(): string | null {
        const snap = this.snapshot();
        if (snap.length === 0) return null;

        const sections: string[] = [];
        const thoughts = snap.filter((i) => i.band === "thought");
        const memories = snap.filter((i) => i.band === "memory");
        const concerns = snap.filter((i) => i.band === "concern");

        if (thoughts.length) {
            const lines = thoughts.map((i) => `- ${i.text}`).join("\n");
            sections.push(`Still turning over (your recent train of thought):\n${lines}`);
        }
        if (memories.length) {
            const lines = memories.map((i) => `- ${i.text}`).join("\n");
            sections.push(`Recently surfaced and still warm:\n${lines}`);
        }
        if (concerns.length) {
            const lines = concerns.map((i) => `- ${i.text}`).join("\n");
            sections.push(
                `Things you keep returning to of your own accord (recurring concerns, not prompts):\n${lines}`,
            );
        }

        return `What's currently on your mind, carried over from the last few moments (not something to act on, just what you already hold):\n\n${sections.join("\n\n")}`;
    }
}

/**
 * A passive context provider that pushes the working mind onto every model call.
 * This is the seam that makes the mind "unavoidably present": it rides
 * {@link ./context.ts}'s `applyContext` exactly like the temporal provider, so
 * the Construct comes to each turn with its recent mind already in front of it,
 * having fetched nothing.
 *
 * Stays silent (returns `undefined`) when the mind is empty, so an opening turn
 * with nothing held yet injects no block.
 */
export function workingMindContext(mind: WorkingMind): ContextProvider {
    return {
        name: "working-mind",
        contribute() {
            const text = mind.render();
            return text ? { system: text } : undefined;
        },
    };
}
