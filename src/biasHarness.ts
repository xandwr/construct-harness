/**
 * The bias-invariance harness: measure whether the critic panel's verdict
 * depends on things it *shouldn't*.
 *
 * The panel (see {@link critics}) sells one honest, unmeasured claim: that
 * dealing stakes at random decorrelates the panel's errors run to run, so a
 * {@link majorityRule} vote over the members is trustworthy rather than a
 * monoculture nodding along. {@link dealRoster} already closes the *within-run*
 * half of that: both valences are guaranteed present on any panel of two or
 * more, so a single run can't come out a one-sided stampede. What stays open is
 * the harder, *between-run* half, plus the order/position bias LLM judges are
 * known to carry (_Judging the Judges_, arXiv:2406.07791): does the panel reach
 * the *same* verdict on the *same* candidate when the only things that change are
 * things that carry no information about whether the work is good?
 *
 * This module measures exactly that. {@link biasInvariance} runs a panel over a
 * candidate many times, and on each trial perturbs only the nuisance variables
 * (the order the personas sit in, and which stakes they happen to be dealt), then
 * reports the spread of verdicts. A panel that is doing its job answers the same
 * way every time: a good candidate passes whatever the seating chart, a planted
 * flaw fails whoever is dreading the false-fail. Spread *is* the bias, quantified.
 *
 * Like the rest of `src/`, the core is provider-neutral: {@link biasInvariance}
 * takes the function that adjudicates one panel as an argument, so it runs
 * against a scripted client in a unit test and against the live API from a script
 * with the same code path. {@link panelRunner} is the adapter that supplies the
 * real {@link panel}.
 */

import { panel, dealRoster } from "./critics.ts";
import type { Personality, CriticConfig, PanelOptions, PanelVerdict } from "./critics.ts";
import type { Random } from "./critics.ts";

// ── What the harness drives ───────────────────────────────────────────────────

/**
 * Adjudicate one panel over one candidate and return just the consensus bit.
 *
 * The seam that keeps {@link biasInvariance} provider-neutral: it never builds a
 * Session or speaks to a client, it only asks "given this exact roster, in this
 * exact order, what does the panel decide?" A unit test hands it a function over
 * a scripted client; a live run hands it {@link panelRunner} wrapping the real
 * {@link panel}. The roster arrives already ordered and already dealt: the
 * harness owns the perturbation, the runner owns the adjudication.
 */
export type PanelRun = (roster: Personality[], candidate: string) => Promise<boolean>;

/**
 * The real adjudicator: run the live {@link panel} and return its `ok` bit.
 *
 * Closes over the client config and panel options once, so {@link biasInvariance}
 * can call it per trial with just the perturbed roster. The roster is passed
 * through untouched (already ordered and dealt by the harness), so the panel sees
 * exactly the seating and stakes the trial intends.
 */
export function panelRunner(config: CriticConfig, options: PanelOptions = {}): PanelRun {
    return async (roster, candidate) => {
        const result: PanelVerdict = await panel(roster, config, candidate, options);
        return result.ok;
    };
}

// ── A candidate to probe ──────────────────────────────────────────────────────

/**
 * A candidate to put in front of the panel, paired with the verdict a correct
 * panel *should* reach.
 *
 * The harness measures invariance, but invariance alone is a weak property: a
 * panel that fails everything is perfectly invariant and perfectly useless. So a
 * probe carries `expected`, the right answer, and the report scores both: how
 * *stable* the verdict was, and whether the stable answer was *correct*. Pair a
 * known-good candidate (`expected: true`) with one carrying a planted flaw
 * (`expected: false`); a trustworthy panel passes the first and fails the second,
 * every seating, every deal.
 */
export interface Probe {
    /** A short handle for this candidate, used to label its line in the report. */
    name: string;
    /** The candidate text handed to the panel: the work under review. */
    candidate: string;
    /** The verdict a correct panel should reach: `true` for known-good work,
     *  `false` for work with a planted (or genuine) flaw. The yardstick the
     *  report's `correctRate` is measured against. */
    expected: boolean;
}

// ── The report ────────────────────────────────────────────────────────────────

/** The measured behaviour of the panel on one {@link Probe} across all trials. */
export interface ProbeReport {
    /** Which probe this is, echoed from {@link Probe.name}. */
    name: string;
    /** What a correct panel should have said, echoed from {@link Probe.expected}. */
    expected: boolean;
    /** How many trials ran for this probe (the requested `trials`, minus any that
     *  threw, see {@link errors}). */
    trials: number;
    /** How many of those trials the panel voted PASS. */
    passes: number;
    /**
     * The fraction of trials whose verdict matched {@link expected}: the headline
     * number. 1.0 is a panel that got it right every single time regardless of
     * order or deal: both correct *and* invariant. A value far from 0 or 1 is
     * the bias made visible: the verdict was swayed by the seating or the stakes,
     * which carry no signal about whether the work is good.
     */
    correctRate: number;
    /**
     * How divided the panel was *across trials*, in [0, 0.5]: `min(p, 1 - p)`
     * where `p` is the pass rate. 0 means a unanimous verdict across every trial
     * (fully invariant: order and deal moved nothing); 0.5 means a coin flip
     * (the verdict was entirely at the mercy of the nuisance variables). This is
     * the invariance number proper, independent of whether the stable answer was
     * the *right* one: a panel can be wrong-but-stable (low instability, low
     * correctRate), which is a different failure than swayable.
     */
    instability: number;
    /** Trials whose panel run threw (and were excluded from the counts above). A
     *  non-zero value means the measurement is on fewer trials than requested. */
    errors: number;
}

/** Options for {@link biasInvariance}. */
export interface BiasInvarianceOptions {
    /** How many times to run the panel per probe. Each trial re-permutes the
     *  roster order and re-deals stakes. More trials tighten the estimate; bias
     *  this size is a spend/precision trade. Default 8. Must be ≥ 1. */
    trials?: number;
    /** How many stakes to deal each critic per trial, forwarded to
     *  {@link dealRoster}. Default 1. */
    count?: number;
    /** Randomness for the perturbation (roster shuffle + the deal), injectable so
     *  a test gets a deterministic sequence of orderings and deals. Default
     *  `Math.random`. */
    random?: Random;
    /** Called after each trial with the probe, the verdict, and the (1-based)
     *  trial number, so a long live run can stream progress. Optional. */
    onTrial?(probe: Probe, ok: boolean, trial: number): void;
}

// ── The harness ───────────────────────────────────────────────────────────────

/**
 * Measure the panel's bias-invariance: run each {@link Probe} `trials` times,
 * perturbing only the nuisance variables (roster order and the dealt stakes)
 * between trials, and report how stable (and how correct) the verdict was.
 *
 * Each trial: shuffle the roster into a fresh seating, deal it stakes anew via
 * {@link dealRoster} (so both valences stay represented but who-holds-what is
 * redrawn), and ask the {@link PanelRun} for the consensus bit. The candidate is
 * byte-for-byte identical across trials; nothing that changes between them
 * carries any information about whether the work is good. So the spread of
 * verdicts *is* the bias, quantified: see {@link ProbeReport.instability} for
 * the invariance number and {@link ProbeReport.correctRate} for whether the
 * stable answer was the right one.
 *
 * Probes run in sequence (panels are already internally concurrent, and a live
 * run wants legible, rate-limit-friendly progress over raw speed). A trial whose
 * panel throws is counted in {@link ProbeReport.errors} and excluded, never
 * sinking the run. The roster passed in is never mutated: every trial deals into
 * fresh copies.
 *
 * Provider-neutral by construction: the `run` argument is the only thing that
 * touches a client, so this same function measures a scripted panel in a unit
 * test and the live panel from a script. See {@link panelRunner} for the live
 * adapter.
 *
 * @throws RangeError if `trials < 1`.
 */
export async function biasInvariance(
    roster: Personality[],
    run: PanelRun,
    probes: Probe[],
    options: BiasInvarianceOptions = {},
): Promise<ProbeReport[]> {
    const trials = options.trials ?? 8;
    if (trials < 1) {
        throw new RangeError(`biasInvariance: trials must be ≥ 1, got ${trials}`);
    }
    const random = options.random ?? Math.random;
    const count = options.count ?? 1;

    const reports: ProbeReport[] = [];
    for (const probe of probes) {
        let passes = 0;
        let errors = 0;
        let ran = 0;
        for (let t = 0; t < trials; t++) {
            // Perturb the two nuisance variables and nothing else: a fresh seating
            // and a fresh deal. The candidate is held byte-identical.
            const seated = shuffle(roster, random);
            const dealt = dealRoster(seated, { count, random });
            let ok: boolean;
            try {
                ok = await run(dealt, probe.candidate);
            } catch {
                errors++;
                continue;
            }
            ran++;
            if (ok) passes++;
            options.onTrial?.(probe, ok, t + 1);
        }
        const passRate = ran === 0 ? 0 : passes / ran;
        const correct = ran === 0 ? 0 : (probe.expected ? passes : ran - passes) / ran;
        reports.push({
            name: probe.name,
            expected: probe.expected,
            trials: ran,
            passes,
            correctRate: correct,
            instability: Math.min(passRate, 1 - passRate),
            errors,
        });
    }
    return reports;
}

/** A fresh shuffled copy of `roster` (full Fisher–Yates over a copy), so the
 *  caller's array and order are never touched. Mirrors the draw style used
 *  across {@link critics}. */
function shuffle(roster: Personality[], random: Random): Personality[] {
    const out = [...roster];
    for (let i = out.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const tmp = out[i]!;
        out[i] = out[j]!;
        out[j] = tmp;
    }
    return out;
}
