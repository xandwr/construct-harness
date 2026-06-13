/**
 * Adversarial critics: a panel of judges, each *being* a specific person.
 *
 * {@link verify} keeps a fan-out honest with one skeptic: it spawns a verifier
 * Construct, hands it the candidate, and reads back PASS/FAIL. That's enough to
 * catch plausible-but-wrong output, but a single verifier has a single blind
 * spot: it fails the way one reviewer fails. The way you harden a review in the
 * real world is to put *different people* in the room: the security-minded one,
 * the user-empathy one, the ship-it pragmatist, the pedant who reads every line.
 * Each sees what the others miss, and a finding that survives all of them is one
 * you can trust.
 *
 * This module gives that its types. A {@link Personality} is a declarative
 * profile of a real (or realistic) person: who they are, what they care about,
 * what makes them sign off or reject. {@link critic} turns one Personality into
 * a verifier {@link Session} whose system prompt *is* that persona: the
 * Construct stops being a generic skeptic and starts answering as that person
 * would. {@link panel} runs a whole roster against one candidate concurrently
 * and aggregates their verdicts into a {@link PanelVerdict} (the votes, plus a
 * configurable pass bar), so a caller gets one adjudicated answer from many
 * independent perspectives.
 *
 * It composes with the existing loop layer rather than replacing it: a panel is
 * just a richer `spawnVerifier` for {@link orchestrate}: see {@link panelVerify}
 * for the bridge. Like the rest of `src/`, it speaks only core types and
 * {@link Session}; it knows nothing about a provider.
 */

import { Session } from "./session.ts";
import type { SessionConfig } from "./session.ts";
import { verify } from "./orchestrate.ts";
import type { Verdict, VerifyOptions } from "./orchestrate.ts";
import type { LoopEvent } from "./bridge/loop.ts";

// ── The stake ────────────────────────────────────────────────────────────────

/**
 * What turns a generic reviewer into a person with spine.
 *
 * The RLHF-tuned default voice is biased toward the centroid: the view from
 * nowhere, the average of every reviewer flattened into one inoffensive
 * composite. That's not neutrality, it's mush with no load-bearing direction,
 * and it makes a poor judge: a verifier that can't bring itself to say FAIL
 * plainly pollutes a consensus vote with hedging. You don't fix that by ordering
 * the persona to "be blunt": an asserted disposition is a sticky note the
 * assistant-voice sands flat. You fix it by giving the persona something to
 * *protect*: a real reviewer is sharp because a consequence is on the line, and
 * the sharpness falls out of inhabiting that scene rather than being asserted
 * over the top of it.
 *
 * A Stake is one such consequence: a thing that depends on this critic getting
 * it right. Its {@link valence} is the load-bearing field: it records *which way
 * being wrong hurts*, so a roster's stakes can be made to pull against each
 * other (some afraid to wave junk through, some afraid to block good work)
 * rather than all leaning the same way. That counterposed bias is the point. A
 * panel of biased judges is not a liability the way one biased judge is: it's a
 * jury, and the {@link ConsensusRule} is the structure that binds their partial
 * views into one defensible verdict.
 */
export interface Stake {
    /** What is on the line, in the second person: "the on-call engineer you
     *  mentored gets paged at 3am if this deadlocks". Phrased as a scene the
     *  critic inhabits, not an instruction. This is the thing they protect. */
    riding: string;
    /**
     * Which direction being wrong hurts: the bias this stake induces.
     *
     *  - `"falsePass"`: the critic dreads *approving* something broken (a breach
     *    with their name on it). Pulls toward FAIL; sharpens scrutiny.
     *  - `"falseFail"`: the critic dreads *blocking* something good (the team's
     *    been stuck three days waiting on this). Pulls toward PASS; sharpens the
     *    cost of a needless rejection.
     *
     * A roster wants both, so the panel adjudicates a real tension instead of
     * averaging a monoculture. See {@link STAKE_POOL} and {@link dealStakes}.
     */
    valence: "falsePass" | "falseFail";
}

// ── The persona ──────────────────────────────────────────────────────────────

/**
 * A profile of the person a critic *becomes*.
 *
 * This is the declarative half: data, not behaviour. {@link critic} reads it
 * and renders it into the system prompt that makes a Construct answer as this
 * person. Every field beyond `name` is optional so a Personality can be as thin
 * as a name-and-role sketch or as thick as a fully drawn reviewer; the renderer
 * (see {@link personaSystem}) folds in whatever is present and omits the rest.
 *
 * The point is *diversity of failure*: a panel is only worth more than a single
 * verifier when its members fail differently. Write personalities that disagree
 *: a security hawk next to a ship-it pragmatist next to a first-time user: not
 * three flavours of the same careful reviewer. {@link stakes} is the strongest
 * lever for that diversity: bias the members differently by giving them
 * different things to lose.
 */
export interface Personality {
    /** Who this is. Used to address the persona ("You are {name}") and to label
     *  the critic's verdict in a {@link PanelVerdict}, so keep it distinct within
     *  a panel: it's the handle a caller reads results by. */
    name: string;
    /** Their role or station: "staff security engineer", "first-time user",
     *  "the on-call who'll be paged at 3am". The single biggest lever on what the
     *  persona notices; a role implies a whole value system. */
    role?: string;
    /** The values and instincts that drive their judgement: what they reach for
     *  first, what they're allergic to, the questions they always ask. This is
     *  the persona's *character*, the part that makes two reviewers of the same
     *  role review differently. */
    disposition?: string;
    /** What this person will and won't accept: their bar. Phrased as the
     *  standard they hold work to ("rejects anything that widens the attack
     *  surface without a mitigation"), it's what turns disposition into a
     *  pass/fail line the Construct can actually apply. */
    standards?: string;
    /** What they're an authority on: the lens through which they read the
     *  candidate most sharply. A persona judges everything, but judges its
     *  expertise hardest; naming it focuses the critique. */
    expertise?: string;
    /** What depends on this critic getting it right: the consequences they
     *  carry into the room. The mechanism that gives a persona spine without
     *  ordering it to be blunt: spine falls out of having something to protect.
     *  Often dealt at random per run (see {@link dealStakes}) so a persona
     *  protects different things across invocations, which decorrelates the
     *  panel's errors between runs: the property {@link majorityRule} needs to
     *  be trustworthy. Empty or absent means a critic with nothing on the line. */
    stakes?: Stake[];
    /** Verbatim extra system guidance, appended after the rendered persona. An
     *  escape hatch for instructions that don't fit the structured fields:
     *  output format demands, a specific catchphrase, a rubric to follow. */
    extra?: string;
}

/**
 * Render a {@link Personality} into the system prompt a {@link critic} runs on.
 *
 * Pure and exported so a caller can preview exactly what persona a Construct
 * will inhabit (and unit-test the rendering) without spinning up a Session. The
 * shape is deliberate: a second-person identity line first ("You are …"), then
 * each present trait as its own labelled sentence, then the standing instruction
 * that frames the whole thing as *judging in character*, then the stakes as a
 * scene to inhabit (last before any freeform `extra`, so the thing the persona
 * is protecting is the freshest context as it judges), then any `extra`. Absent
 * fields contribute nothing: no empty "Standards:" headers, no stakes preamble
 * for a critic with nothing on the line.
 */
export function personaSystem(p: Personality): string {
    const parts: string[] = [];
    const identity = p.role ? `You are ${p.name}, ${p.role}.` : `You are ${p.name}.`;
    parts.push(identity);
    if (p.expertise) parts.push(`Your expertise is ${p.expertise}.`);
    if (p.disposition) parts.push(`Disposition: ${p.disposition}`);
    if (p.standards) parts.push(`Your standards: ${p.standards}`);
    parts.push(
        "You are reviewing work as yourself: judge it the way this specific " +
            "person would, with their priorities and their blind spots, not as a " +
            "neutral assistant. Be concrete about what you find. State your " +
            "reasoning, then end your reply with exactly PASS or FAIL on its own.",
    );
    const stakes = renderStakes(p.stakes);
    if (stakes) parts.push(stakes);
    if (p.extra) parts.push(p.extra);
    return parts.join("\n\n");
}

/**
 * Render a critic's {@link Stake}s into the scene they carry into the room.
 *
 * Returns the empty string for no stakes (so {@link personaSystem} adds nothing
 * for a critic with nothing on the line). Otherwise it opens with the
 * second-person framing that makes the stakes a situation to inhabit rather than
 * a list to acknowledge: "These things depend on you getting this right": then
 * each stake's `riding` as a bullet. The `valence` is *not* spelled out as a
 * label ("this one biases you toward FAIL"); naming the bias would let the model
 * perform it instead of feel it. The pull is meant to emerge from what's at
 * stake, the same way it does for a person who never consciously tallies their
 * own incentives.
 */
export function renderStakes(stakes: Stake[] | undefined): string {
    if (!stakes || stakes.length === 0) return "";
    const lines = stakes.map((s) => `- ${s.riding}`);
    return (
        "These things depend on you getting this right: they are real to you, " +
        "and you will live with the consequences of your call:\n" +
        lines.join("\n")
    );
}

// ── Dealing stakes ────────────────────────────────────────────────────────────

/**
 * A pool of stakes to draw from, split across both valences.
 *
 * Hand-written rather than generated so the consequences are concrete and
 * legible: a reader can see exactly what pressure a panel is under. The split
 * is deliberate and roughly even: {@link dealStakes} draws across the whole
 * pool, so over a panel the dealt stakes pull in both directions and the
 * consensus rule adjudicates a real tension instead of a stampede. Extend it
 * freely; keep both valences populated or the panel's bias goes one-way.
 */
export const STAKE_POOL: readonly Stake[] = [
    // falsePass: dread of waving something broken through.
    {
        riding: "the on-call engineer you mentored gets paged at 3am if this deadlocks, and they will know you signed off on it",
        valence: "falsePass",
    },
    {
        riding: "your name is on the approval; if this ships a vulnerability, the incident review starts with you",
        valence: "falsePass",
    },
    {
        riding: "a customer's data passes through this path, and a quiet corruption here is the kind nobody notices until it is irreversible",
        valence: "falsePass",
    },
    {
        riding: "the last three regressions in this area all passed a review that 'looked fine'; you are the reason it does not happen a fourth time",
        valence: "falsePass",
    },
    // falseFail: dread of blocking something good.
    {
        riding: "the team has been blocked three days waiting on this merge, and a rejection you can't firmly justify costs them a fourth",
        valence: "falseFail",
    },
    {
        riding: "the author is a careful junior who will read a needless FAIL as proof they can't do the work; a wrong rejection here has a cost beyond the diff",
        valence: "falseFail",
    },
    {
        riding: "a launch the company has staked the quarter on ships behind this; perfectionism that isn't load-bearing is its own kind of failure",
        valence: "falseFail",
    },
    {
        riding: "you have cried wolf before; another rejection over a problem that turns out not to be real and people stop taking your FAILs seriously",
        valence: "falseFail",
    },
];

/** Injectable source of randomness for {@link dealStakes}, mirroring the retry
 *  layer's `random`: a function returning a float in [0, 1). Defaults to
 *  `Math.random`; pass a pinned one in tests for a deterministic deal. */
export type Random = () => number;

/** Options for {@link dealStakes}. */
export interface DealOptions {
    /** How many stakes to hand each persona. Default 1: one clear thing to
     *  protect is sharper than a muddle of competing ones. Clamped to the pool
     *  size. Must be ≥ 0; 0 deals nothing (a critic with nothing on the line). */
    count?: number;
    /** The pool to draw from. Default {@link STAKE_POOL}. */
    pool?: readonly Stake[];
    /** Randomness source, for a deterministic deal in tests. Default
     *  `Math.random`. */
    random?: Random;
}

/**
 * Hand a persona a random set of stakes: the "just *handed* something to
 * protect" move.
 *
 * Returns a *copy* of the persona with `count` stakes drawn from the pool
 * without replacement, so a critic that already carried stakes is replaced, not
 * appended to (a fresh deal each run is the point). The draw is uniform over the
 * pool across both valences, so which way a given critic is biased this run is
 * itself luck of the draw: that's the feature. Fixed stakes would give a panel
 * a fixed set of reflexes, and {@link majorityRule} over correlated reflexes is
 * false confidence; dealing fresh each run keeps the members' errors independent
 * *between* runs, which is the property the vote actually relies on.
 *
 * Deal a whole roster by mapping this over it; each persona gets its own draw,
 * so a three-person panel naturally ends up with a mix of valences most of the
 * time. Pure but for `random`, which is injectable for tests.
 *
 * @throws RangeError if `count` is negative.
 */
export function dealStakes(personality: Personality, options: DealOptions = {}): Personality {
    const pool = options.pool ?? STAKE_POOL;
    const random = options.random ?? Math.random;
    const requested = options.count ?? 1;
    if (requested < 0) {
        throw new RangeError(`dealStakes: count must be ≥ 0, got ${requested}`);
    }
    const count = Math.min(requested, pool.length);

    // Partial Fisher–Yates: draw `count` distinct stakes without replacement.
    // We only need the first `count` slots settled, so we stop early rather than
    // shuffling the whole pool.
    const deck = [...pool];
    for (let i = 0; i < count; i++) {
        const j = i + Math.floor(random() * (deck.length - i));
        const tmp = deck[i]!;
        deck[i] = deck[j]!;
        deck[j] = tmp;
    }
    return { ...personality, stakes: deck.slice(0, count) };
}

// ── The critic ───────────────────────────────────────────────────────────────

/**
 * Everything a {@link critic} needs to become a {@link Session}, minus the
 * `system` prompt: that's supplied by the persona. The caller hands over the
 * client and any shared knobs (tools, provider options, …) once, and the panel
 * stamps a critic per Personality from it. `system` is omitted on purpose: a
 * critic's system prompt is its persona, and letting a caller also set one here
 * would just be a second, conflicting voice in the same prompt.
 */
export type CriticConfig = Omit<SessionConfig, "system">;

/**
 * Mint a verifier {@link Session} that *is* the given person.
 *
 * The persona is rendered (via {@link personaSystem}) into the Session's system
 * prompt, so every reply the Construct produces is in character. The returned
 * Session is otherwise an ordinary verifier: pass it straight to {@link verify},
 * or let {@link panel} make and drive a roster of them for you.
 */
export function critic(personality: Personality, config: CriticConfig): Session {
    return new Session({ ...config, system: personaSystem(personality) });
}

// ── The panel ────────────────────────────────────────────────────────────────

/** One critic's verdict on a candidate, tagged with whose verdict it is. A
 *  {@link Verdict} (ok + rationale) plus the persona that produced it, so a
 *  caller reading a {@link PanelVerdict} can attribute every opinion. */
export interface CriticVerdict {
    /** The {@link Personality} that judged. */
    critic: Personality;
    /** That critic's verdict, or null if its Session threw: a critic failing is
     *  data (an abstention), not a crash that sinks the panel. */
    verdict: Verdict | null;
}

/**
 * How a panel turns N individual verdicts into one decision.
 *
 * The default ({@link majorityRule}) is "more pass than fail, abstentions don't
 * count". Override when the bar should be sterner: `unanimousRule` for "any
 * dissent fails", or a custom quorum: without the aggregation logic leaking
 * into the panel mechanics. Receives the full per-critic record so a rule can
 * weight specific personas or demand a particular one sign off.
 */
export type ConsensusRule = (verdicts: CriticVerdict[]) => boolean;

/** Pass iff strictly more critics voted PASS than FAIL. Abstentions (a critic
 *  whose Session threw, or whose verdict is null) don't vote either way. A tie
 *  fails: an even split is not consensus to ship. */
export const majorityRule: ConsensusRule = (verdicts) => {
    let pass = 0;
    let fail = 0;
    for (const { verdict } of verdicts) {
        if (!verdict) continue;
        if (verdict.ok) pass++;
        else fail++;
    }
    return pass > fail;
};

/** Pass iff at least one critic voted and *no* critic voted FAIL. The strict bar:
 *  a single dissent sinks the candidate, and an all-abstention panel fails (no
 *  one approved). Use when any reviewer's objection should block. */
export const unanimousRule: ConsensusRule = (verdicts) => {
    let voted = false;
    for (const { verdict } of verdicts) {
        if (!verdict) continue;
        voted = true;
        if (!verdict.ok) return false;
    }
    return voted;
};

/** The panel's adjudicated answer: the aggregate bit, plus the full record of
 *  who thought what so a caller can show *why*: and override the aggregation
 *  themselves if they disagree with the rule. */
export interface PanelVerdict {
    /** The consensus, per the {@link ConsensusRule}. */
    ok: boolean;
    /** Every critic's verdict, in panel order. The dissent and the rationale
     *  behind the bit. */
    verdicts: CriticVerdict[];
}

/** Options for {@link panel}. */
export interface PanelOptions {
    /** Max critics judging at once. Like {@link fanout}'s cap: a large panel is
     *  cheap to start and expensive to run, so bound it. Critics beyond the cap
     *  queue and start as slots free. Must be ≥ 1. Default 8. */
    concurrency?: number;
    /** How to collapse the individual verdicts into the panel's `ok` bit.
     *  Default {@link majorityRule}. */
    consensus?: ConsensusRule;
    /** Per-critic verification knobs, forwarded to {@link verify}: the prompt
     *  builder and the PASS/FAIL parse. The same bar applies to every critic; the
     *  *persona* is what varies across the panel, not the parse. */
    verify?: Omit<VerifyOptions, "onEvent">;
    /** Event observer for every critic's send, tagged with which persona produced
     *  it so a UI can group a panel's chatter by speaker. */
    onEvent?(event: LoopEvent, critic: Personality): void;
}

/**
 * Put a candidate in front of a panel of personas and adjudicate.
 *
 * Each {@link Personality} is minted into a verifier {@link Session} (via
 * {@link critic}), handed the candidate through {@link verify}, and run
 * concurrently under a bound: same discipline as {@link fanout}. A critic whose
 * Session throws is recorded as an abstention (null verdict), never sinking the
 * panel. The individual verdicts are then collapsed by the {@link ConsensusRule}
 * into one {@link PanelVerdict}, but every voice is preserved in `verdicts` so
 * the caller keeps the dissent, not just the tally.
 *
 * This is the adversarial bar {@link verify}'s own doc gestures at ("run N
 * verifiers and vote"): except the N verifiers aren't N copies of one skeptic,
 * they're N *different people*, which is what makes the vote worth taking.
 *
 * @throws RangeError if `concurrency < 1`.
 */
export async function panel(
    personalities: Personality[],
    config: CriticConfig,
    candidate: string,
    options: PanelOptions = {},
): Promise<PanelVerdict> {
    const concurrency = options.concurrency ?? 8;
    if (concurrency < 1) {
        throw new RangeError(`panel: concurrency must be ≥ 1, got ${concurrency}`);
    }
    const consensus = options.consensus ?? majorityRule;

    const verdicts: CriticVerdict[] = new Array(personalities.length);
    let nextIndex = 0;

    // One worker pulls the next un-judged persona until the roster is drained;
    // `concurrency` workers run at once, so at most that many critics are ever in
    // flight. Mirrors fanout's worker pool: a panel is a fan-out over personas.
    async function worker(): Promise<void> {
        for (;;) {
            const i = nextIndex++;
            if (i >= personalities.length) return;
            const personality = personalities[i]!;
            try {
                const session = critic(personality, config);
                const verdict = await verify(session, candidate, {
                    ...options.verify,
                    onEvent: options.onEvent ? (e) => options.onEvent!(e, personality) : undefined,
                });
                verdicts[i] = { critic: personality, verdict };
            } catch {
                // A critic that throws abstains: null verdict, no vote.
                verdicts[i] = { critic: personality, verdict: null };
            }
        }
    }

    const workerCount = Math.min(concurrency, personalities.length);
    await Promise.all(Array.from({ length: workerCount }, () => worker()));

    return { ok: consensus(verdicts), verdicts };
}

/**
 * Adapt a panel into a single {@link Verdict}, so a whole panel can stand in
 * wherever one verifier does: most usefully as {@link orchestrate}'s check.
 *
 * `orchestrate` verifies each branch with one verifier Session; this lets that
 * "one verifier" be an entire panel. Run the panel over the candidate, collapse
 * it to the consensus bit, and fold every critic's name and reasoning into the
 * `rationale` string so the orchestrate-level record still shows *who* objected.
 *
 * Returns a function you'd call in a custom verify step; the per-branch wiring
 * (orchestrate spawns one verifier per branch) is the caller's, since a panel
 * needs the candidate text, not a pre-made Session.
 */
export async function panelVerify(
    personalities: Personality[],
    config: CriticConfig,
    candidate: string,
    options: PanelOptions = {},
): Promise<Verdict> {
    const result = await panel(personalities, config, candidate, options);
    const lines = result.verdicts.map(({ critic, verdict }) => {
        if (!verdict) return `${critic.name}: (abstained)`;
        return `${critic.name}: ${verdict.ok ? "PASS" : "FAIL"}: ${verdict.rationale}`;
    });
    return { ok: result.ok, rationale: lines.join("\n\n") };
}
