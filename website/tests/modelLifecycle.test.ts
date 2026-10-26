import assert from "node:assert/strict";
import { test } from "node:test";
import { ModelLifecycle } from "../src/lib/modelLifecycle.ts";

function specimen(id: string) {
  let frees = 0;
  return { id, model: { free: () => { frees++; } }, get frees() { return frees; } };
}

test("failed load preserves usable current model and does not commit preferences", async () => {
  const original = specimen("original");
  let current: ReturnType<typeof specimen> | null = original;
  let preferences = "";
  const slot = new ModelLifecycle(() => current, (m) => { current = m; });
  await assert.rejects(
    slot.replace(async () => { throw new Error("checkpoint SHA-256 mismatch"); }, () => { preferences = "new"; }),
    /SHA-256 mismatch/,
  );
  assert.equal(current, original);
  assert.equal(original.frees, 0);
  assert.equal(preferences, "");
});

test("successful replacement publishes once and frees only the previous instance", async () => {
  const original = specimen("original");
  const next = specimen("next");
  let current: ReturnType<typeof specimen> | null = original;
  const events: string[] = [];
  const slot = new ModelLifecycle(() => current, (m) => { current = m; events.push("publish"); });
  const value = await slot.replace(async () => next, () => events.push("commit"));
  assert.equal(value, next);
  assert.equal(current, next);
  assert.equal(original.frees, 1);
  assert.equal(next.frees, 0);
  assert.deepEqual(events, ["publish", "commit"]);
});

test("a competing selection disposes a stale completed model, never publishing it", async () => {
  const original = specimen("original");
  const stale = specimen("stale");
  const latest = specimen("latest");
  let current: ReturnType<typeof specimen> | null = original;
  let resolveStale!: () => void;
  const slow = new Promise<void>((resolve) => { resolveStale = resolve; });
  const slot = new ModelLifecycle(() => current, (m) => { current = m; });
  const pending = slot.replace(async () => { await slow; return stale; });
  const winner = await slot.replace(async () => latest);
  resolveStale();
  await assert.rejects(pending, (e: unknown) => e instanceof DOMException && e.name === "AbortError");
  assert.equal(winner, latest);
  assert.equal(current, latest);
  assert.equal(original.frees, 1);
  assert.equal(stale.frees, 1);
  assert.equal(latest.frees, 0);
});

test("unload invalidates an in-flight selection and does not publish its late result", async () => {
  const original = specimen("original");
  const late = specimen("late");
  let current: ReturnType<typeof specimen> | null = original;
  let resolve!: () => void;
  const waiting = new Promise<void>((ready) => { resolve = ready; });
  const slot = new ModelLifecycle(() => current, (m) => { current = m; });
  const pending = slot.replace(async () => { await waiting; return late; });
  slot.unload();
  resolve();
  await assert.rejects(pending, (e: unknown) => e instanceof DOMException && e.name === "AbortError");
  assert.equal(current, null);
  assert.equal(original.frees, 1);
  assert.equal(late.frees, 1);
});
