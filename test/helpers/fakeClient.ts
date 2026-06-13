/**
 * The scripted {@link ModelClient} used across the test suite.
 *
 * The implementation now lives in `src/testing.ts` so it ships as part of the
 * public surface (consumers can write the same zero-spend dry-runs against their
 * own Constructs). This file is a thin re-export so the existing test imports
 * (`./helpers/fakeClient.ts`) keep working unchanged.
 */

export { FakeClient, callTurn, textTurn } from "../../src/testing.ts";
export type { ScriptedTurn } from "../../src/testing.ts";
