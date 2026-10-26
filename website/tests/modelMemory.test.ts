import assert from "node:assert/strict";
import { test } from "node:test";
import { assertReplacementFits, estimateReplacement, ModelMemoryBudgetError } from "../src/lib/modelMemory.ts";
import type { ModelDef } from "../src/lib/models.ts";

function model(key: string, size: number): ModelDef {
  return { key, label: key, hfRepo: `fixture/${key}`, approxSizeMb: size, description: "" };
}
const small = model("small", 282);
const base = model("base", 738);
const multi = model("multilingual", 1096);
const decide = model("decide", 1856);

test("conservative 32-bit guard allows ordinary small and base cold loads", () => {
  assert.equal(estimateReplacement(null, small).isOversized, false);
  assert.equal(estimateReplacement(null, base).isOversized, false);
  assert.equal(estimateReplacement(null, multi).isOversized, false);
});

test("preflight permits bounded small-to-base replacement but blocks unsafe dual residency", () => {
  assert.equal(estimateReplacement(small, base).isOversized, false);
  for (const [from, to] of [[small, multi], [base, multi], [multi, small]] as const) {
    assert.throws(() => assertReplacementFits(from, to), (error: unknown) => {
      assert.ok(error instanceof ModelMemoryBudgetError);
      assert.match(error.message, /unload active model/);
      assert.match(error.message, /refresh the tab/);
      assert.ok(error.estimate.estimatedPeakMiB > error.estimate.budgetMiB);
      return true;
    });
  }
});

test("conservative deviceMemory cap prevents 1-GiB model on a 2-GiB device", () => {
  assert.equal(estimateReplacement(null, multi, { deviceMemoryGiB: 2 }).isOversized, true);
  assert.equal(estimateReplacement(null, small, { deviceMemoryGiB: 2 }).isOversized, false);
});

test("monotonically grown WebAssembly memory is counted after explicit unload", () => {
  const estimate = estimateReplacement(null, small, { wasmLinearMiB: 3200 });
  assert.equal(estimate.isOversized, true);
});

test("invalid estimates cannot bypass the guard", () => {
  assert.throws(() => assertReplacementFits(null, model("invalid", Number.NaN)), /Invalid replacement/);
  assert.throws(() => assertReplacementFits(model("invalid", 0), small), /Invalid active/);
});

test("oversized new checkpoint is rejected before any download", () => {
  assert.throws(() => assertReplacementFits(null, decide), ModelMemoryBudgetError);
});

test("allocation failures carry actionable recovery and non-memory errors keep their original cause", async () => {
  const { explainModelLoadFailure } = await import("../src/lib/modelMemory.ts");
  assert.throws(
    () => explainModelLoadFailure(new RangeError("memory access out of bounds"), small),
    /refresh the tab to reset WebAssembly linear memory/,
  );
  const invalid = new Error("invalid checkpoint config");
  assert.throws(() => explainModelLoadFailure(invalid, small), (error) => error === invalid);
});
