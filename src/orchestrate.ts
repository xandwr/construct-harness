/**
 * Loops: driving Constructs without a human in the seat.
 *
 * Where {@link Session} is the thing a *person* talks to (a REPL feeds it one
 * user turn at a time), a Loop is the thing that talks to a Session *for* you:
 * it supplies the turns programmatically, decides from each result whether the
 * goal is met, and re-prompts until it is. The implementation is handled by the
 * Construct; the Loop is the control flow you'd otherwise be running by hand.
 *
 * Two layers, smallest first:
 *
 *  - {@link loop} drives ONE Construct to a goal. You give it an opening task, a
 *    `done` predicate over each turn, and a `next` prompt to send when it isn't
 *    done yet. It iterates until `done` or a `maxIterations` backstop.
 *
 *  - {@link fanout}, {@link verify} and {@link orchestrate} sit on top: run many
 *    independent Constructs in parallel (bounded concurrency), check each one's
 *    output with a verifier Construct, and merge the survivors. The single-
 *    Construct case is just `orchestrate` with one task: the layers compose.
 *
 * Like the rest of `src/`, this speaks only core types and {@link Session}; it
 * knows nothing about a provider. A Session with no store is a fine Construct to
 * loop: memory is orthogonal.
 */

import { Session } from "./session.ts";
import type { SessionConfig, TurnResult } from "./session.ts";
import type { LoopEvent } from "./bridge/loop.ts";

// ── Layer 1: iterate one Construct to a goal ─────────────────────────────────

/**
 * One step of a {@link loop}: the turn the Construct just produced, plus which
 * iteration it was (1-based) so a `done`/`next` callback can reason about depth
 * without threading its own counter.
 */
export interface LoopStep {
    /** The Construct's result for this iteration. */
    turn: TurnResult;
    /** 1-based iteration number this turn corresponds to. */
    iteration: number;
}

/** How a {@link loop} run ended. */
export type LoopStopReason =
    /** The `done` predicate returned true: the goal was reached. */
    | "done"
    /** Hit {@link LoopOptions.maxIterations} without `done` ever holding. */
    | "maxIterations";

/** Configuration for {@link loop}. */
export interface LoopOptions {
    /**
     * Goal predicate, evaluated after every iteration's turn. Return true to
     * stop with reason `"done"`. This is the Loop's whole judgement: it is the
     * `while` condition you'd otherwise be applying in your head each time you
     * read the Construct's reply. May be async (e.g. to run a check command).
     */
    done(step: LoopStep): boolean | Promise<boolean>;
    /**
     * The prompt to send for the *next* iteration, given the turn that didn't
     * satisfy `done`. This is where you steer: "address the failing test", "you
     * still have TODOs, continue". Not called once `done` holds. May be async.
     */
    next(step: LoopStep): string | Promise<string>;
    /**
     * Hard cap on iterations, so a `done` that never fires can't run forever.
     * The Loop's only cost bound (we deliberately do no token accounting here:
     * iterations are the knob). Must be ≥ 1. Default 10.
     */
    maxIterations?: number;
    /**
     * Optional observer for every {@link LoopEvent} the underlying Session
     * emits, across all iterations: so a REPL or TUI can render the Loop's work
     * live exactly as it renders an interactive turn. The Loop itself needs only
     * the per-turn {@link TurnResult}, so this is purely for display.
     */
    onEvent?(event: LoopEvent, iteration: number): void;
}

/** The outcome of a {@link loop} run. */
export interface LoopResult {
    /** Why the loop stopped. */
    stopReason: LoopStopReason;
    /** The final turn the Construct produced (the one `done` was last checked
     *  against). Always set: a loop runs at least one iteration. */
    final: TurnResult;
    /** Every turn produced, in order: `turns.length` is the iteration count. */
    turns: TurnResult[];
    /** Convenience: turns produced (1-based max iteration reached). */
    iterations: number;
    /** Token totals summed across every iteration of the run, so a caller can
     *  see what the whole Loop cost without re-summing `turns`. */
    usage: { inputTokens: number; outputTokens: number; cacheReadTokens: number };
}

/** Drain one `Session.send`, forwarding events to `onEvent` and returning the
 *  turn. Factored out because both the opening task and each follow-up are the
 *  same "run a send to completion" shape. */
async function drive(
    session: Session,
    prompt: string,
    iteration: number,
    onEvent: LoopOptions["onEvent"],
): Promise<TurnResult> {
    const gen = session.send(prompt);
    let next = await gen.next();
    while (!next.done) {
        onEvent?.(next.value, iteration);
        next = await gen.next();
    }
    return next.value;
}

/**
 * Drive a single Construct toward a goal.
 *
 * Sends `task`, then evaluates `done` against the result. If the goal isn't met
 * and iterations remain, it asks `next` for the follow-up prompt and sends that
 * into the *same* Session: so the Construct keeps its full conversation and
 * memory across iterations, exactly as if a person had typed the follow-ups.
 * Stops the moment `done` holds, or when `maxIterations` is reached.
 *
 * The Session is the caller's: `loop` neither constructs nor disposes it, so the
 * same Construct can be looped more than once, or talked to interactively after.
 *
 * @throws RangeError if `maxIterations < 1`: a loop must run at least once, and
 *   a zero/negative cap is a caller bug, not a no-op we should swallow.
 */
export async function loop(
    session: Session,
    task: string,
    options: LoopOptions,
): Promise<LoopResult> {
    const maxIterations = options.maxIterations ?? 10;
    if (maxIterations < 1) {
        throw new RangeError(`loop: maxIterations must be ≥ 1, got ${maxIterations}`);
    }

    const turns: TurnResult[] = [];
    const usage = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0 };

    let prompt = task;
    let stopReason: LoopStopReason = "maxIterations";

    for (let iteration = 1; iteration <= maxIterations; iteration++) {
        const turn = await drive(session, prompt, iteration, options.onEvent);
        turns.push(turn);
        usage.inputTokens += turn.usage.inputTokens;
        usage.outputTokens += turn.usage.outputTokens;
        usage.cacheReadTokens += turn.usage.cacheReadTokens;

        const step: LoopStep = { turn, iteration };
        if (await options.done(step)) {
            stopReason = "done";
            break;
        }
        // Not done. If iterations remain, compute the follow-up; if this was the
        // last allowed iteration, we fall out of the loop with "maxIterations".
        if (iteration < maxIterations) {
            prompt = await options.next(step);
        }
    }

    return {
        stopReason,
        final: turns[turns.length - 1]!,
        turns,
        iterations: turns.length,
        usage,
    };
}

// ── Layer 2: fan out, verify, merge ──────────────────────────────────────────

/**
 * A factory that mints a *fresh* Construct for one fan-out branch.
 *
 * Each branch must get its own {@link Session} so the branches are independent:
 * sharing one Session would entangle their conversations and serialize them
 * through one history. The factory receives the branch's task and index, so it
 * can specialize the Construct per branch (different system prompt, tools, …) if
 * it wants; the trivial factory ignores both and returns `new Session(cfg)`.
 */
export type Spawn = (task: string, index: number) => Session;

/** Build a {@link Spawn} that stamps out Sessions from one shared config. The
 *  common case: N identical Constructs differing only in their task. */
export function spawnFrom(config: SessionConfig): Spawn {
    return () => new Session(config);
}

/** One branch's outcome. A branch that threw is captured here as `error` with a
 *  null `result`, rather than rejecting the whole batch: same discipline as the
 *  loop's tool handling: an independent failure is data, not a crash. */
export interface BranchResult {
    /** The task this branch was given. */
    task: string;
    /** Position in the input list, so results stay correlatable after settling. */
    index: number;
    /** The branch's final turn, or null if it threw. */
    result: TurnResult | null;
    /** The error, if this branch threw. Mutually exclusive with a non-null
     *  `result`. */
    error?: Error;
}

/** Options for {@link fanout}. */
export interface FanoutOptions {
    /**
     * Max branches running at once. Fan-out is token-cheap to *start* and
     * expensive to run; a cap keeps a 100-task fan-out from opening 100
     * concurrent model streams. Branches beyond the cap queue and start as slots
     * free. Must be ≥ 1. Default 8.
     */
    concurrency?: number;
    /** Forwarded to each branch's {@link loop} as its event observer; the branch
     *  index is the loop iteration's sibling, so a UI can group by branch. */
    onEvent?(event: LoopEvent, branch: number): void;
}

/**
 * Run many Constructs in parallel, one per task, with bounded concurrency.
 *
 * Each task gets a fresh Construct from `spawn` and is driven by a single send
 * (the fan-out unit is "one Construct, one task": loop *within* a branch by
 * having `spawn` hand back a Session you then drive, or compose with
 * {@link loop} in the caller). Branches are independent: one throwing is caught
 * and surfaced as a {@link BranchResult} with an `error`, never sinking its
 * siblings. Results come back in input order regardless of finish order.
 */
export async function fanout(
    tasks: string[],
    spawn: Spawn,
    options: FanoutOptions = {},
): Promise<BranchResult[]> {
    const concurrency = options.concurrency ?? 8;
    if (concurrency < 1) {
        throw new RangeError(`fanout: concurrency must be ≥ 1, got ${concurrency}`);
    }

    const results: BranchResult[] = new Array(tasks.length);
    let nextIndex = 0;

    // A worker pulls the next un-started task index until the queue is drained.
    // `concurrency` workers run concurrently; each awaits its branch fully before
    // taking another, so at most `concurrency` branches are ever in flight.
    async function worker(): Promise<void> {
        for (;;) {
            const index = nextIndex++;
            if (index >= tasks.length) return;
            const task = tasks[index]!;
            try {
                const session = spawn(task, index);
                const turn = await drive(
                    session,
                    task,
                    index,
                    options.onEvent && ((e, branch) => options.onEvent!(e, branch)),
                );
                results[index] = { task, index, result: turn };
            } catch (err) {
                results[index] = {
                    task,
                    index,
                    result: null,
                    error: err instanceof Error ? err : new Error(String(err)),
                };
            }
        }
    }

    const workerCount = Math.min(concurrency, tasks.length);
    await Promise.all(Array.from({ length: workerCount }, () => worker()));
    return results;
}

/** A verifier's judgement on a candidate. */
export interface Verdict {
    /** Whether the candidate passed the verifier's bar. */
    ok: boolean;
    /** The verifier Construct's full reply, so a caller can show *why*: the
     *  rationale, not just the bit. */
    rationale: string;
}

/** Options for {@link verify}. */
export interface VerifyOptions {
    /**
     * Decide pass/fail from the verifier Construct's reply text. The default
     * treats a reply that contains "PASS" (and not "FAIL") as ok: pair it with
     * a verifier system prompt that ends its verdict with PASS or FAIL. Override
     * to parse whatever convention your verifier uses.
     */
    decide?(replyText: string): boolean;
    /** How to phrase the verification request to the verifier Construct, given
     *  the candidate. Default wraps it in a terse "verify this / answer PASS or
     *  FAIL" instruction. */
    prompt?(candidate: string): string;
    /** Event observer, forwarded to the verifier's single send. */
    onEvent?(event: LoopEvent): void;
}

/** Default verdict parse: PASS present and FAIL absent. Case-sensitive so prose
 *  that merely discusses "fail" doesn't trip it: the verifier emits the tokens
 *  deliberately. */
function defaultDecide(replyText: string): boolean {
    return replyText.includes("PASS") && !replyText.includes("FAIL");
}

function defaultVerifyPrompt(candidate: string): string {
    return (
        "Verify the following work. Judge it strictly and explain your reasoning, " +
        "then end your reply with exactly PASS or FAIL on its own.\n\n" +
        candidate
    );
}

/**
 * Check a candidate with a verifier Construct.
 *
 * Spawns nothing itself: the caller passes a `verifier` Session so the verifier
 * can carry its own system prompt (the skeptic instructions) and, if desired,
 * accumulate context across several `verify` calls. Runs one send and maps the
 * reply to a {@link Verdict} via `decide`.
 *
 * This is the unit that keeps a fan-out honest: a Loop that generates work
 * without an independent check just produces plausible output at scale. Run N
 * verifiers and vote in the caller for an adversarial bar.
 */
export async function verify(
    verifier: Session,
    candidate: string,
    options: VerifyOptions = {},
): Promise<Verdict> {
    const decide = options.decide ?? defaultDecide;
    const buildPrompt = options.prompt ?? defaultVerifyPrompt;

    const turn = await drive(
        verifier,
        buildPrompt(candidate),
        1,
        options.onEvent && ((e) => options.onEvent!(e)),
    );
    return { ok: decide(turn.text), rationale: turn.text };
}

/** A fanned-out branch paired with its verifier's verdict. */
export interface VerifiedBranch extends BranchResult {
    /** The verdict, or null if the branch itself failed (nothing to verify). */
    verdict: Verdict | null;
}

/** Options for {@link orchestrate}. */
export interface OrchestrateOptions {
    /** Mints a worker Construct per task. */
    spawn: Spawn;
    /** Mints a verifier Construct per *successful* branch. A fresh verifier per
     *  branch keeps verdicts independent; reuse one by closing over it if you
     *  want a verifier that sees every candidate. */
    spawnVerifier: Spawn;
    /** Concurrency for the worker fan-out (verification reuses it). Default 8. */
    concurrency?: number;
    /** Per-candidate verification knobs, forwarded to {@link verify}. */
    verify?: Omit<VerifyOptions, "onEvent">;
    /** Event observer for both the worker and verifier sends. The `phase` tells
     *  a UI which stage a given branch's events belong to. */
    onEvent?(event: LoopEvent, branch: number, phase: "work" | "verify"): void;
}

/** The result of an {@link orchestrate} run. */
export interface OrchestrateResult {
    /** Every branch with its verdict, in input order: the full record. */
    branches: VerifiedBranch[];
    /** Just the branches that produced work AND passed verification: the
     *  trustworthy output. */
    confirmed: VerifiedBranch[];
}

/**
 * The full Loop: fan out tasks to Constructs, verify each result, return the
 * survivors.
 *
 * This is the "I write Loops and the Loops do the implementation" shape end to
 * end: you supply *what* to do (the tasks) and *how to trust it* (the verifier
 * and its bar), and the orchestration runs the Constructs and the checks without
 * you adjudicating each one. The single-Construct case (`tasks.length === 1`) is
 * just this with one branch, so there's one primitive, not two.
 *
 * Verification runs as each branch finishes its work: a branch that threw is
 * carried through with a null verdict (nothing to check), never verified, and
 * excluded from `confirmed`.
 */
export async function orchestrate(
    tasks: string[],
    options: OrchestrateOptions,
): Promise<OrchestrateResult> {
    const concurrency = options.concurrency ?? 8;

    // Stage 1: fan out the work.
    const worked = await fanout(tasks, options.spawn, {
        concurrency,
        onEvent: options.onEvent && ((e, branch) => options.onEvent!(e, branch, "work")),
    });

    // Stage 2: verify the branches that produced something, with the same
    // concurrency bound. A failed branch has nothing to verify: pass it through
    // with a null verdict so the input-order record stays complete.
    const branches: VerifiedBranch[] = new Array(worked.length);
    let nextIndex = 0;

    async function verifyWorker(): Promise<void> {
        for (;;) {
            const i = nextIndex++;
            if (i >= worked.length) return;
            const branch = worked[i]!;
            if (!branch.result) {
                branches[i] = { ...branch, verdict: null };
                continue;
            }
            const verifier = options.spawnVerifier(branch.task, branch.index);
            const verdict = await verify(verifier, branch.result.text, {
                ...options.verify,
                onEvent: options.onEvent
                    ? (e) => options.onEvent!(e, branch.index, "verify")
                    : undefined,
            });
            branches[i] = { ...branch, verdict };
        }
    }

    const workerCount = Math.min(concurrency, worked.length);
    await Promise.all(Array.from({ length: workerCount }, () => verifyWorker()));

    const confirmed = branches.filter((b) => b.verdict?.ok === true);
    return { branches, confirmed };
}
