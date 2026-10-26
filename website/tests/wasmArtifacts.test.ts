import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, test } from "node:test";
import { artifacts, packageDifferences } from "../scripts/wasm-artifacts.ts";

const dirs: string[] = [];
function fixtures(): { built: string; tracked: string } {
  const dir = mkdtempSync(join(tmpdir(), "gliner-wasm-drift-test-"));
  dirs.push(dir);
  const built = join(dir, "built");
  const tracked = join(dir, "tracked");
  mkdirSync(built);
  mkdirSync(tracked);
  for (const file of artifacts) {
    const bytes = file.endsWith(".wasm") ? Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]) : Buffer.from(file);
    writeFileSync(join(built, file), bytes);
    writeFileSync(join(tracked, file), bytes);
  }
  return { built, tracked };
}
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }); });

test("identical generated artifacts pass", () => {
  const { built, tracked } = fixtures();
  assert.deepEqual(packageDifferences(built, tracked), []);
});

test("changed WebAssembly binary fails even with identical package metadata", () => {
  const { built, tracked } = fixtures();
  writeFileSync(join(built, "gliner_rs_bg.wasm"), Buffer.from([0, 97, 115, 109, 1, 0, 0, 1]));
  assert.match(packageDifferences(built, tracked).join("\n"), /gliner_rs_bg\.wasm: generated SHA-256/);
  assert.equal(readFileSync(join(built, "package.json"), "utf8"), readFileSync(join(tracked, "package.json"), "utf8"));
});

test("changed JavaScript glue and declarations fail", () => {
  const { built, tracked } = fixtures();
  writeFileSync(join(built, "gliner_rs.js"), "different glue");
  writeFileSync(join(built, "gliner_rs.d.ts"), "different typings");
  const differences = packageDifferences(built, tracked);
  assert.equal(differences.length, 2);
});

test("missing committed binary and missing generated metadata fail", () => {
  const { built, tracked } = fixtures();
  unlinkSync(join(tracked, "gliner_rs_bg.wasm"));
  unlinkSync(join(built, "package.json"));
  assert.deepEqual(packageDifferences(built, tracked).map((d) => d.split(":")[0]), [
    "committed package is missing gliner_rs_bg.wasm",
    "generator did not emit package.json",
  ]);
});
