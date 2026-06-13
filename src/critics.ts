/**
 * Adversarial critics — a panel of judges, each *being* a specific person.
 *
 * {@link verify} keeps a fan-out honest with one skeptic: it spawns a verifier
 * Construct, hands it the candidate, and reads back PASS/FAIL. That's enough to
 * catch plausible-but-wrong output, but a single verifier has a single blind
 * spot — it fails the way one reviewer fails. The way you harden a review in the
 * real world is to put *different people* in the room: the security-minded one,
 * the user-empathy one, the ship-it pragmatist, the pedant who reads every line.
 * Each sees what the others miss, and a finding that survives all of them is one
 * you can trust.
 *
 * This module gives that its types. A {@link Personality} is a declarative
 * profile of a real (or realistic) person: who they are, what they care about,
 * what makes them sign off or reject. {@link critic} turns one Personality into
 * a verifier {@link Session} whose system prompt *is* that persona — the
 * Construct stops being a generic skeptic and starts answering as that person
 * would. {@link panel} runs a whole roster against one candidate concurrently
 * and aggregates their verdicts into a {@link PanelVerdict} (the votes, plus a
 * configurable pass bar), so a caller gets one adjudicated answer from many
 * independent perspectives.
 *
 * It composes with the existing loop layer rather than replacing it: a panel is
 * just a richer `spawnVerifier` for {@link orchestrate} — see {@link panelVerify}
 * for the bridge. Like the rest of `src/`, it speaks only core types and
 * {@link Session}; it knows nothing about a provider.
 */

import { Session } from "./session.ts";
import type { SessionConfig } from "./session.ts";
import { verify } from "./orchestrate.ts";
import type { Verdict, VerifyOptions } from "./orchestrate.ts";
import type { LoopEvent } from "./bridge/loop.ts";

// ── The persona ──────────────────────────────────────────────────────────────

/**
 * A profile of the person a critic *becomes*.
 *
 * This is the declarative half — data, not behaviour. {@link critic} reads it
 * and renders it into the system prompt that makes a Construct answer as this
 * person. Every field beyond `name` is optional so a Personality can be as thin
 * as a name-and-role sketch or as thick as a fully drawn reviewer; the renderer
 * (see {@link personaSystem}) folds in whatever is present and omits the rest.
 *
 * The point is *diversity of failure*: a panel is only worth more than a single
 * verifier when its members fail differently. Write personalities that disagree
 * — a security hawk next to a ship-it pragmatist next to a first-time user — not
 * three flavours of the same careful reviewer.
 */
export interface Personality {
    /** Who this is. Used to address the persona ("You are {name}") and to label
     *  the critic's verdict in a {@link PanelVerdict}, so keep it distinct within
     *  a panel — it's the handle a caller reads results by. */
    name: string;
    /** Their role or station — "staff security engineer", "first-time user",
     *  "the on-call who'll be paged at 3am". The single biggest lever on what the
     *  persona notices; a role implies a whole value system. */
    role?: string;
    /** The values and instincts that drive their judgement: what they reach for
     *  first, what they're allergic to, the questions they always ask. This is
     *  the persona's *character*, the part that makes two reviewers of the same
     *  role review differently. */
    disposition?: string;
    /** What this person will and won't accept — their bar. Phrased as the
     *  standard they hold work to ("rejects anything that widens the attack
     *  surface without a mitigation"), it's what turns disposition into a
     *  pass/fail line the Construct can actually apply. */
    standards?: string;
    /** What they're an authority on — the lens through which they read the
     *  candidate most sharply. A persona judges everything, but judges its
     *  expertise hardest; naming it focuses the critique. */
    expertise?: string;
    /** Verbatim extra system guidance, appended after the rendered persona. An
     *  escape hatch for instructions that don't fit the structured fields —
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
 * that frames the whole thing as *judging in character*, then any `extra`. Absent
 * fields contribute nothing — no empty "Standards:" headers.
 */
export function personaSystem(p: Personality): string {
    const parts: string[] = [];
    const identity = p.role ? `You are ${p.name}, ${p.role}.` : `You are ${p.name}.`;
    parts.push(identity);
    if (p.expertise) parts.push(`Your expertise is ${p.expertise}.`);
    if (p.disposition) parts.push(`Disposition: ${p.disposition}`);
    if (p.standards) parts.push(`Your standards: ${p.standards}`);
    parts.push(
        "You are reviewing work as yourself — judge it the way this specific " +
            "person would, with their priorities and their blind spots, not as a " +
            "neutral assistant. Be concrete about what you find. State your " +
            "reasoning, then end your reply with exactly PASS or FAIL on its own.",
    );
    if (p.extra) parts.push(p.extra);
    return parts.join("\n\n");
}

// ── The critic ───────────────────────────────────────────────────────────────

/**
 * Everything a {@link critic} needs to become a {@link Session}, minus the
 * `system` prompt — that's supplied by the persona. The caller hands over the
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
 * Session is otherwise an ordinary verifier — pass it straight to {@link verify},
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
    /** That critic's verdict, or null if its Session threw — a critic failing is
     *  data (an abstention), not a crash that sinks the panel. */
    verdict: Verdict | null;
}

/**
 * How a panel turns N individual verdicts into one decision.
 *
 * The default ({@link majorityRule}) is "more pass than fail, abstentions don't
 * count". Override when the bar should be sterner — `unanimousRule` for "any
 * dissent fails", or a custom quorum — without the aggregation logic leaking
 * into the panel mechanics. Receives the full per-critic record so a rule can
 * weight specific personas or demand a particular one sign off.
 */
export type ConsensusRule = (verdicts: CriticVerdict[]) => boolean;

/** Pass iff strictly more critics voted PASS than FAIL. Abstentions (a critic
 *  whose Session threw, or whose verdict is null) don't vote either way. A tie
 *  fails — an even split is not consensus to ship. */
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
 *  who thought what so a caller can show *why* — and override the aggregation
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
    /** Per-critic verification knobs, forwarded to {@link verify} — the prompt
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
 * concurrently under a bound — same discipline as {@link fanout}. A critic whose
 * Session throws is recorded as an abstention (null verdict), never sinking the
 * panel. The individual verdicts are then collapsed by the {@link ConsensusRule}
 * into one {@link PanelVerdict}, but every voice is preserved in `verdicts` so
 * the caller keeps the dissent, not just the tally.
 *
 * This is the adversarial bar {@link verify}'s own doc gestures at ("run N
 * verifiers and vote") — except the N verifiers aren't N copies of one skeptic,
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
    // flight. Mirrors fanout's worker pool — a panel is a fan-out over personas.
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
                // A critic that throws abstains — null verdict, no vote.
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
 * wherever one verifier does — most usefully as {@link orchestrate}'s check.
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
        return `${critic.name}: ${verdict.ok ? "PASS" : "FAIL"} — ${verdict.rationale}`;
    });
    return { ok: result.ok, rationale: lines.join("\n\n") };
}
