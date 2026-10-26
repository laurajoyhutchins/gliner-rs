/**
 * Regenerate or independently compare the committed browser WebAssembly package.
 * Never compare files against a build directory that was reused from a prior run.
 * The Rust and wasm-pack versions are pinned in root rust-toolchain.toml and
 * website/wasm-pack-version respectively.
 */
import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const committed = join(root, "website/src/wasm-pkg");
export const artifacts = [
  "gliner_rs_bg.wasm",
  "gliner_rs.js",
  "gliner_rs.d.ts",
  "gliner_rs_bg.wasm.d.ts",
  "package.json",
] as const;

function digest(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/** Every generated browser-facing byte is compared, not only package version. */
export function packageDifferences(generated: string, checkedIn: string): string[] {
  const differences: string[] = [];
  for (const file of artifacts) {
    const actual = join(generated, file);
    const expected = join(checkedIn, file);
    if (!existsSync(actual)) {
      differences.push(`generator did not emit ${file}`);
    } else if (!existsSync(expected)) {
      differences.push(`committed package is missing ${file}`);
    } else if (digest(actual) !== digest(expected)) {
      differences.push(`${file}: generated SHA-256 ${digest(actual)} != committed ${digest(expected)}`);
    }
  }
  return differences;
}

function commandOutput(command: string, args: string[]): string {
  return execFileSync(command, args, { cwd: root, encoding: "utf8" }).trim();
}

function checkTools(): void {
  const toolchain = readFileSync(join(root, "rust-toolchain.toml"), "utf8");
  const rustVersion = /^channel\s*=\s*"([^"]+)"\s*$/m.exec(toolchain)?.[1];
  if (!rustVersion || !/^\d+\.\d+\.\d+$/.test(rustVersion)) {
    throw new Error("rust-toolchain.toml must pin an exact Rust release");
  }
  const wasmPackVersion = readFileSync(join(root, "website/wasm-pack-version"), "utf8").trim();
  if (!/^\d+\.\d+\.\d+$/.test(wasmPackVersion)) {
    throw new Error("website/wasm-pack-version must pin an exact wasm-pack release");
  }
  const rustc = commandOutput("rustc", ["--version"]);
  const wasmPack = commandOutput("wasm-pack", ["--version"]);
  if (!rustc.startsWith(`rustc ${rustVersion} `)) {
    throw new Error(`expected rustc ${rustVersion}, found ${rustc}`);
  }
  if (wasmPack !== `wasm-pack ${wasmPackVersion}`) {
    throw new Error(`expected wasm-pack ${wasmPackVersion}, found ${wasmPack}`);
  }
  console.log(`${rustc}; ${wasmPack}`);
}

function build(generated: string): void {
  checkTools();
  const result = spawnSync(
    "wasm-pack",
    [
      "build", "--target", "web", "--release", "--no-opt",
      "--out-dir", generated, "--", "--locked", "--no-default-features",
    ],
    {
      cwd: root,
      env: { ...process.env, CARGO_INCREMENTAL: "0", SOURCE_DATE_EPOCH: "0", TZ: "UTC" },
      stdio: "inherit",
    },
  );
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`wasm-pack build failed with status ${result.status}`);
  for (const file of artifacts) {
    if (!existsSync(join(generated, file))) throw new Error(`build omitted ${file}`);
  }
}

export function main(args: string[]): void {
  const mode = args[0];
  if (args.length !== 1 || (mode !== "--check" && mode !== "--write")) {
    throw new Error("usage: node --experimental-strip-types website/scripts/wasm-artifacts.ts --check|--write");
  }
  const scratch = mkdtempSync(join(tmpdir(), "gliner-wasm-"));
  try {
    const generated = resolve(scratch, "pkg");
    build(generated);
    if (mode === "--check") {
      const differences = packageDifferences(generated, committed);
      if (differences.length > 0) {
        throw new Error(
          "Committed WebAssembly is stale relative to current Rust source:\n" +
          differences.map((line) => `  - ${line}`).join("\n") +
          "\nRegenerate with: pnpm --dir website build:wasm, and commit website/src/wasm-pkg/.",
        );
      }
      console.log("Committed WebAssembly is byte-for-byte identical to a fresh pinned build.");
    } else {
      mkdirSync(committed, { recursive: true });
      for (const file of artifacts) copyFileSync(join(generated, file), join(committed, file));
      console.log("Regenerated browser WebAssembly package; commit all five generated files.");
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  }
}
