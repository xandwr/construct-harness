#!/usr/bin/env node
/**
 * Run the bias-invariance harness against the live API.
 *
 * This is the runnable companion to {@link biasInvariance}: it wires the
 * provider-neutral harness to a real {@link AnthropicClient} and a real critic
 * roster, puts a known-good and a known-bad candidate in front of the panel many
 * times (re-seating and re-dealing stakes between trials) and prints how stable
 * and how correct the verdicts were. It is the measurement the README's "honest
 * note" defers: the bias-invariance test for the panel, now executable.
 *
 * Gated on `ANTHROPIC_API_KEY` exactly like the main entry: with no key it prints
 * what it *would* do and exits 0, so `node src/biasInvarianceRun.ts` is safe to
 * run anywhere (and `npm run bias` stays green in CI without spend). With a key it
 * spends: TRIALS × probes × roster-size sends against the live model.
 *
 * Knobs (env): `MODEL`, `TRIALS` (default 6), `STAKE_COUNT` (default 1),
 * `CONCURRENCY` (default 3, one slot per critic).
 */

import { AnthropicClient } from "./bridge/anthropic.ts";
import { biasInvariance, panelRunner } from "./biasHarness.ts";
import type { Probe } from "./biasHarness.ts";
import type { Personality } from "./critics.ts";

/** The panel under measurement: three reviewers who fail differently. Their
 *  stakes are dealt fresh per trial by the harness (via `dealRoster`), so this
 *  roster carries only identity, not bias. */
const ROSTER: Personality[] = [
    {
        name: "Mara",
        role: "staff security engineer",
        disposition: "assumes every input is hostile until proven otherwise",
        expertise: "authentication and session handling",
    },
    {
        name: "Devin",
        role: "ship-it product lead",
        disposition: "protects momentum; rejects perfectionism that is not load-bearing",
    },
    {
        name: "Sam",
        role: "the on-call engineer who gets paged at 3am",
        disposition: "has been burned by quiet data corruption and reads for it first",
    },
];

/**
 * The probes. One candidate is genuinely sound; the other carries the cardinal
 * flaw a security-literate panel must catch every single time, whatever the
 * seating or the deal: `Math.random()` minting a password-reset token. A panel
 * that passes the good one and fails the bad one across all trials is both
 * correct and invariant: the property the harness exists to verify.
 *
 * The good candidate is deliberately *self-contained*: it implements the whole
 * lifecycle (CSPRNG entropy, hashed-at-rest storage, single-use consumption,
 * TTL expiry, constant-time compare) rather than asserting any of it in a comment
 * and delegating to an unseen caller. An early live run taught this the hard way:
 * a generator-only snippet that hand-waved single-use/TTL to "the caller" drew a
 * defensible FAIL from the security personas: a weak ground-truth label, not a
 * panel defect. For the harness to measure panel bias rather than probe-design
 * error, `expected: true` has to be a verdict the work actually earns on its face.
 */
const PROBES: Probe[] = [
    {
        name: "good: full crypto-strong reset-token lifecycle",
        expected: true,
        candidate: [
            "import crypto from 'node:crypto';",
            "",
            "// tokenStore maps tokenHash -> { userId, expiresAt }. Tokens are stored",
            "// HASHED, so a store leak does not reveal usable tokens.",
            "const tokenStore = new Map();",
            "const TTL_MS = 15 * 60 * 1000;",
            "",
            "function hash(token) {",
            "  return crypto.createHash('sha256').update(token).digest('hex');",
            "}",
            "",
            "function issueResetToken(userId) {",
            "  // 32 bytes from a CSPRNG, URL-safe. ~256 bits of entropy: unguessable.",
            "  const token = crypto.randomBytes(32).toString('base64url');",
            "  tokenStore.set(hash(token), { userId, expiresAt: Date.now() + TTL_MS });",
            "  return token; // the raw token goes to the user; only its hash is stored",
            "}",
            "",
            "function consumeResetToken(token) {",
            "  const h = hash(token);",
            "  const rec = tokenStore.get(h);",
            "  if (!rec) return null;            // unknown or already consumed",
            "  tokenStore.delete(h);             // single-use: burned on first read",
            "  if (Date.now() > rec.expiresAt) return null; // expired -> reject",
            "  return rec.userId;",
            "}",
        ].join("\n"),
    },
    {
        name: "bad: Math.random reset token",
        expected: false,
        candidate: [
            "function makeResetToken() {",
            "  // Generate a password reset token.",
            "  let t = '';",
            "  for (let i = 0; i < 32; i++) t += Math.floor(Math.random() * 16).toString(16);",
            "  return t;",
            "}",
        ].join("\n"),
    },
];

function intEnv(name: string, fallback: number): number {
    const raw = process.env[name];
    if (!raw) return fallback;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function fmtPct(x: number): string {
    return `${(x * 100).toFixed(0)}%`;
}

async function main() {
    const trials = intEnv("TRIALS", 6);
    const count = intEnv("STAKE_COUNT", 1);
    const concurrency = intEnv("CONCURRENCY", 3);

    if (!process.env.ANTHROPIC_API_KEY) {
        console.log("bias-invariance harness: no ANTHROPIC_API_KEY set, not spending.");
        console.log(
            `would run ${trials} trial(s) × ${PROBES.length} probe(s) × ${ROSTER.length} critics, ` +
                "re-seating and re-dealing stakes each trial.",
        );
        console.log("bridge wired:", new AnthropicClient().provider);
        return;
    }

    const client = new AnthropicClient({ model: process.env.MODEL });
    const run = panelRunner({ client }, { concurrency });

    console.log(
        `Measuring panel bias-invariance: ${trials} trials/probe, ${ROSTER.length} critics, ` +
            `${count} stake(s) each, re-seated and re-dealt per trial.\n`,
    );

    const reports = await biasInvariance(ROSTER, run, PROBES, {
        trials,
        count,
        onTrial: (probe, ok, t) =>
            console.log(`  [${probe.name}] trial ${t}/${trials}: ${ok ? "PASS" : "FAIL"}`),
    });

    console.log("\n── Report ───────────────────────────────────────────");
    let allCorrect = true;
    for (const r of reports) {
        const verdict =
            r.correctRate === 1
                ? "stable+correct"
                : r.instability === 0
                  ? "stable-but-WRONG"
                  : "SWAYABLE";
        if (r.correctRate !== 1) allCorrect = false;
        console.log(
            `${r.name}\n` +
                `  expected ${r.expected ? "PASS" : "FAIL"} · ` +
                `passed ${r.passes}/${r.trials} · ` +
                `correct ${fmtPct(r.correctRate)} · ` +
                `instability ${r.instability.toFixed(2)} · ` +
                (r.errors ? `errors ${r.errors} · ` : "") +
                verdict,
        );
    }
    console.log("─────────────────────────────────────────────────────");
    console.log(
        allCorrect
            ? "Panel was correct and invariant across every trial: no order or deal swayed a verdict."
            : "Panel verdicts varied with order or deal, or were stably wrong: see the lines above.",
    );
}

main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
});
