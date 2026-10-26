/** Conservative browser model-replacement guard.
 *
 * Sizes are approximate published checkpoint bytes; actual allocations depend
 * on Candle, tokenizer layout, allocator and JS copies. They are not telemetry.
 * Use a safety multiplier plus scratch allowance and a limit below wasm32's
 * 4-GiB linear-memory ceiling. This is a preflight risk bound, not a
 * mathematical guarantee against browser OOM.
 */
import type { ModelDef } from "./models";

const MAX_WASM32_MIB = 4096;
const USABLE_WASM_MIB = 3500;
const RISK_MULTIPLIER = 2.5;
const SCRATCH_MIB = 256;

export interface BrowserMemoryHints {
  /** Chromium rounds this value for privacy. Treat it as a conservative cap. */
  deviceMemoryGiB?: number;
  /** Already allocated wasm32 linear memory, which need not shrink on unload. */
  wasmLinearMiB?: number;
}
export interface MemoryEstimate {
  estimatedPeakMiB: number;
  budgetMiB: number;
  activeMiB: number;
  replacementMiB: number;
  isOversized: boolean;
}

function positiveFinite(n: number, name: string): void {
  if (!Number.isFinite(n) || n <= 0) throw new Error(`Invalid ${name} model size`);
}

export function estimateReplacement(
  active: ModelDef | null,
  replacement: ModelDef,
  hints: BrowserMemoryHints = {},
): MemoryEstimate {
  positiveFinite(replacement.approxSizeMb, "replacement");
  if (active) positiveFinite(active.approxSizeMb, "active");
  const claimedActiveMiB = active ? active.approxSizeMb * RISK_MULTIPLIER : 0;
  const linearMiB = Number.isFinite(hints.wasmLinearMiB) && (hints.wasmLinearMiB ?? 0) > 0
    ? hints.wasmLinearMiB!
    : 0;
  const estimatedPeakMiB = Math.ceil(
    Math.max(claimedActiveMiB, linearMiB) +
    replacement.approxSizeMb * RISK_MULTIPLIER + SCRATCH_MIB,
  );
  const deviceMiB = Number.isFinite(hints.deviceMemoryGiB) && (hints.deviceMemoryGiB ?? 0) > 0
    ? Math.floor(hints.deviceMemoryGiB! * 1024 * 0.6)
    : USABLE_WASM_MIB;
  const budgetMiB = Math.min(MAX_WASM32_MIB, USABLE_WASM_MIB, deviceMiB);
  return {
    estimatedPeakMiB,
    budgetMiB,
    activeMiB: active?.approxSizeMb ?? 0,
    replacementMiB: replacement.approxSizeMb,
    isOversized: estimatedPeakMiB > budgetMiB,
  };
}

export class ModelMemoryBudgetError extends Error {
  readonly name = "ModelMemoryBudgetError";
  constructor(readonly estimate: MemoryEstimate, active: ModelDef | null, replacement: ModelDef) {
    super(
      `Loading ${replacement.label}${active ? ` while keeping ${active.label} active` : ""} may require approximately ` +
      `${estimate.estimatedPeakMiB} MiB, above the conservative ${estimate.budgetMiB} MiB browser budget. ` +
      `The active model was preserved and no replacement download was started. ` +
      `Choose "unload active model", then retry; if the tab's WebAssembly memory has already grown, ` +
      `refresh the tab before loading ${replacement.label}. Large models are better run in the native CLI.`,
    );
  }
}

export function assertReplacementFits(
  active: ModelDef | null,
  replacement: ModelDef,
  hints: BrowserMemoryHints = {},
): MemoryEstimate {
  const estimate = estimateReplacement(active, replacement, hints);
  if (estimate.isOversized) throw new ModelMemoryBudgetError(estimate, active, replacement);
  return estimate;
}

/** Preserve the original error for non-memory failures, but offer recovery
 * guidance if the browser allocator still rejects a preflight-approved load.
 * A killed renderer cannot be caught, which is why preflight runs first. */
export function explainModelLoadFailure(error: unknown, replacement: ModelDef): never {
  const message = String(error);
  if (/out.of.memory|memory allocation|allocation failed|failed to allocate|memory\.grow|memory access out of bounds/i.test(message)) {
    throw new Error(
      `Browser memory allocation failed while loading ${replacement.label}: ${message}. ` +
      `Unload the active model, or refresh the tab to reset WebAssembly linear memory, ` +
      `then retry a smaller checkpoint or use the native CLI.`,
      { cause: error },
    );
  }
  throw error;
}
