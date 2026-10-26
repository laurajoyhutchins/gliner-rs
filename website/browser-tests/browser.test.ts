/**
 * Chromium execution tests for the committed browser runtime, not a Node WASM
 * substitute. The HTTP server exposes exactly the files the test requires.
 * No mock of WebAssembly, wasm-bindgen glue, or the Rust inference methods.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { once } from "node:events";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { finished } from "node:stream/promises";
import { after, before, test } from "node:test";
import { chromium, type Browser } from "playwright";

const website = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const generated = join(website, "src/wasm-pkg");
const tmp = mkdtempSync(join(tmpdir(), "gliner-browser-smoke-"));
const files = ["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"] as const;
type FileName = typeof files[number];
interface Lock {
  version: number;
  repo: string;
  revision: string;
  sha256: Record<FileName, string>;
}
const locks = JSON.parse(readFileSync(join(website, "src/lib/checkpoint-locks.json"), "utf8")) as Lock[];
const pinned = locks.find((entry) => entry.repo === "fastino/gliner2.5-small-v1");
if (!pinned || pinned.version !== 1 || !/^[a-f0-9]{40}$/.test(pinned.revision)) {
  throw new Error("Missing immutable small-checkpoint lock in website manifest");
}
const checkpoint = pinned;

let server: Server;
let browser: Browser;
let baseUrl = "";
const fixture = new Map<string, string>();
const contentType = (path: string): string => path.endsWith(".wasm") ? "application/wasm" : path.endsWith(".js") ? "text/javascript" : path.endsWith(".json") ? "application/json" : "application/octet-stream";

async function fixtureFile(file: FileName, lock: Lock = checkpoint): Promise<string> {
  const output = join(tmp, lock.repo.split("/").pop() + "_" + file.replaceAll("/", "_"));
  const expected = lock.sha256[file];
  if (!/^[a-f0-9]{64}$/.test(expected)) throw new Error(`Unpinned or invalid SHA-256 for ${file}`);
  const url = `https://huggingface.co/${lock.repo}/resolve/${lock.revision}/${file}`;
  const response = await fetch(url, { cache: "no-store", signal: AbortSignal.timeout(300_000) });
  if (!response.ok || !response.body) throw new Error(`Pinned download ${file}: HTTP ${response.status}`);
  const stream = createWriteStream(output, { flags: "wx" });
  const digest = createHash("sha256");
  let length = 0;
  try {
    for await (const value of response.body) {
      const bytes = Buffer.from(value);
      length += bytes.length;
      digest.update(bytes);
      if (!stream.write(bytes)) await once(stream, "drain");
    }
    stream.end();
    await finished(stream);
    const actual = digest.digest("hex");
    if (actual !== expected) {
      throw new Error(`Pinned ${file} SHA-256 mismatch: ${actual} != ${expected}`);
    }
    console.log(`verified fixture ${file}: ${length} bytes, SHA-256 ${actual}`);
    return output;
  } catch (error) {
    stream.destroy();
    rmSync(output, { force: true });
    throw error;
  }
}

function serve(requestUrl: string | undefined, response: import("node:http").ServerResponse): void {
  const pathname = new URL(requestUrl || "/", "http://localhost").pathname;
  if (pathname === "/") {
    response.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
    response.end("<!doctype html><html><head><meta charset='utf-8'></head><body>WebAssembly runtime smoke</body></html>");
    return;
  }
  const paths: Readonly<Record<string, string>> = {
    "/gliner_rs.js": join(generated, "gliner_rs.js"),
    "/gliner_rs_bg.wasm": join(generated, "gliner_rs_bg.wasm"),
  };
  const file = paths[pathname] ?? fixture.get(pathname);
  if (!file || !existsSync(file)) {
    response.writeHead(404);
    response.end("No such approved test resource");
    return;
  }
  const stats = statSync(file);
  response.writeHead(200, {
    "content-type": contentType(file),
    "content-length": String(stats.size),
    "cache-control": "no-store",
    "access-control-allow-origin": "http://127.0.0.1",
  });
  createReadStream(file).pipe(response);
}

before(async () => {
  assert.ok(existsSync(join(generated, "gliner_rs_bg.wasm")), "Committed .wasm must exist");
  server = createServer((req, res) => serve(req.url, res));
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Expected ephemeral localhost HTTP port");
  baseUrl = `http://127.0.0.1:${address.port}`;
  browser = await chromium.launch({ headless: true });
});

after(async () => {
  await browser?.close();
  if (server?.listening) await new Promise<void>((resolveClose, reject) => server.close((error) => error ? reject(error) : resolveClose()));
  rmSync(tmp, { recursive: true, force: true });
});

test("generated Rust WebAssembly initializes in Chromium and rejects malformed checkpoints", { timeout: 90_000 }, async () => {
  const page = await browser.newPage();
  try {
    await page.goto(baseUrl);
    const got = await page.evaluate(async () => {
      const wasm = await import("/gliner_rs.js");
      const instance = await wasm.default();
      wasm.init_panic_hook();
      let message = "";
      try {
        wasm.WasmModel.load(new Uint8Array([0, 1, 2, 3]), new Uint8Array([123]), "{not-json", "{}");
      } catch (error) {
        message = String(error);
      }
      return {
        actualModule: instance.memory instanceof WebAssembly.Memory,
        exposedLoader: typeof wasm.WasmModel.load === "function",
        malformedMessage: message,
      };
    });
    assert.equal(got.actualModule, true, "Must instantiate compiled .wasm, not a fake JS implementation");
    assert.equal(got.exposedLoader, true);
    assert.ok(got.malformedMessage.length > 0, "Malformed checkpoint must fail without crashing the tab");
    assert.ok(await page.evaluate(() => !!document.body), "Browser remains usable after rejected checkpoint");
  } finally {
    await page.close();
  }
});

test("manifest-pinned real GLiNER2.5-small checkpoint runs classification in Chromium", {
  timeout: 900_000,
  skip: process.env.GLINER_REAL_CHECKPOINT !== "1" ? "Enable with GLINER_REAL_CHECKPOINT=1; exact-head CI must set this" : false,
}, async () => {
  // Download and digest-check all checkpoint files outside the browser, then
  // serve their EXACT bytes over localhost. Nothing is mocked in WasmModel.
  for (const file of files) fixture.set(`/checkpoint/${file}`, await fixtureFile(file));
  const page = await browser.newPage();
  try {
    await page.goto(baseUrl);
    const started = Date.now();
    const result = await page.evaluate(async () => {
      const wasm = await import("/gliner_rs.js");
      await wasm.default();
      wasm.init_panic_hook();
      const get = async (name: string): Promise<Uint8Array> => {
        const response = await fetch(`/checkpoint/${name}`);
        if (!response.ok) throw new Error(`Checkpoint HTTP ${response.status}: ${name}`);
        return new Uint8Array(await response.arrayBuffer());
      };
      const [config, encoder] = await Promise.all([
        fetch("/checkpoint/config.json").then((r) => r.text()),
        fetch("/checkpoint/encoder_config/config.json").then((r) => r.text()),
      ]);
      const tokenizer = await get("tokenizer.json");
      const weights = await get("model.safetensors");
      const model = wasm.WasmModel.load(weights, tokenizer, config, encoder);
      try {
        const tasks = JSON.stringify([{
          task: "topic",
          labels: [{ label: "technology" }, { label: "sports" }, { label: "business" }],
        }]);
        return JSON.parse(model.classify("Apple released a new iPhone in Cupertino.", tasks));
      } finally {
        model.free();
      }
    });
    assert.ok(Array.isArray(result) && result.length === 1, "Classification returns one task");
    assert.equal(result[0].task, "topic");
    assert.equal(result[0].labels.length, 3);
    const names = new Set(result[0].labels.map((x: { label: string }) => x.label));
    assert.deepEqual(names, new Set(["technology", "sports", "business"]));
    for (const label of result[0].labels) {
      assert.ok(Number.isFinite(label.prob) && label.prob >= 0 && label.prob <= 1,
        `Real WebAssembly inference must return a finite probability: ${JSON.stringify(label)}`);
    }
    assert.ok(result[0].labels.some((x: { prob: number }, index: number, values: { prob: number }[]) =>
      index > 0 && x.prob !== values[0].prob), "Inference should distinguish inputs/labels, not return a constant stub");
    console.log(JSON.stringify({
      proof: "browser-real-inference",
      checkpoint: checkpoint.repo,
      revision: checkpoint.revision,
      elapsedMs: Date.now() - started,
      predictions: result,
    }));
  } finally {
    await page.close();
  }
});

/** Opt-in real checkpoint memory bench: every model starts in a fresh page.
 * Chromium process VmRSS is sampled during synchronous WASM initialization.
 * VmRSS sums include shared mappings and are not physical peak PSS; preserve
 * this limitation in every report. Network fixture acquisition is excluded
 * from reported cold start time. */
test("measure real browser cold-load and bounded switching footprints", {
  timeout: 2_700_000,
  skip: process.env.GLINER_MEMORY_BENCHMARK !== "1" ? "Set GLINER_MEMORY_BENCHMARK=1" : false,
}, async () => {
  const keys = [
    "fastino/gliner2.5-small-v1",
    "fastino/gliner2.5-base-v1",
    "fastino/gliner2.5-multi-v1",
  ];
  const input = "Apple released a new iPhone in Cupertino.";
  const task = JSON.stringify([{ task: "topic", labels: [{ label: "technology" }, { label: "sports" }, { label: "business" }] }]);
  const reports: Record<string, unknown>[] = [];
  const CDP = await browser.newBrowserCDPSession();
  const readRssMiB = async (): Promise<number> => {
    const value = await CDP.send("SystemInfo.getProcessInfo") as { processInfo: { id: number }[] };
    let total = 0;
    for (const process of value.processInfo) {
      try {
        const status = readFileSync(`/proc/${process.id}/status`, "utf8");
        const match = /^VmRSS:\s+(\d+)\s+kB/m.exec(status);
        if (match) total += Number(match[1]) / 1024;
      } catch {
        // A renderer may exit between the CDP snapshot and /proc read.
      }
    }
    return total;
  };
  const measure = async <T>(action: () => Promise<T>): Promise<{ value: T; baselineRssMiB: number; peakRssMiB: number; elapsedMs: number }> => {
    const baselineRssMiB = await readRssMiB();
    if (baselineRssMiB <= 0) throw new Error("Chromium RSS process telemetry unavailable; cannot claim a memory benchmark");
    let peakRssMiB = baselineRssMiB;
    let busy = false;
    const sample = async () => {
      if (busy) return;
      busy = true;
      try { peakRssMiB = Math.max(peakRssMiB, await readRssMiB()); }
      finally { busy = false; }
    };
    const timer = setInterval(() => { void sample().catch(() => {}); }, 75);
    const began = performance.now();
    try {
      const value = await action();
      await sample();
      return { value, baselineRssMiB, peakRssMiB, elapsedMs: Math.round(performance.now() - began) };
    } finally {
      clearInterval(timer);
    }
  };
  try {
    for (const repo of keys) {
      const lock = locks.find((value) => value.repo === repo);
      if (!lock) throw new Error(`Missing immutable lock for ${repo}`);
      const root = `/bench/${repo.split("/").pop()}`;
      for (const file of files) fixture.set(`${root}/${file}`, await fixtureFile(file, lock));
      const page = await browser.newPage();
      try {
        await page.goto(baseUrl);
        const measured = await measure(() => page.evaluate(async ({ root, input, task }) => {
          const wasm = await import("/gliner_rs.js");
          const exports = await wasm.default();
          const fetchText = async (path: string) => (await fetch(path)).text();
          const fetchBytes = async (path: string) => new Uint8Array(await (await fetch(path)).arrayBuffer());
          const config = await fetchText(`${root}/config.json`);
          const encoder = await fetchText(`${root}/encoder_config/config.json`);
          const tokenizer = await fetchBytes(`${root}/tokenizer.json`);
          const weights = await fetchBytes(`${root}/model.safetensors`);
          const t0 = performance.now();
          const model = wasm.WasmModel.load(weights, tokenizer, config, encoder);
          const initMs = Math.round(performance.now() - t0);
          try {
            const result = JSON.parse(model.classify(input, task));
            if (!Array.isArray(result) || !result[0]?.labels?.length) throw new Error("Inference did not return task labels");
            return { initMs, wasmLinearMiB: exports.memory.buffer.byteLength / 1048576 };
          } finally {
            model.free();
          }
        }, { root, input, task }));
        reports.push({ repo, status: "passed", ...measured, checkpointBytes: statSync(fixture.get(`${root}/model.safetensors`)!).size });
      } catch (error) {
        reports.push({ repo, status: "failed", error: String(error) });
      } finally {
        await page.close().catch(() => {});
      }
    }
    // An actual co-resident switch, keeping small alive while base initializes.
    const page = await browser.newPage();
    try {
      await page.goto(baseUrl);
      const switching = await measure(() => page.evaluate(async ({ task, input }) => {
        const wasm = await import("/gliner_rs.js");
        const exports = await wasm.default();
        const load = async (root: string) => {
          const config = await (await fetch(`${root}/config.json`)).text();
          const encoder = await (await fetch(`${root}/encoder_config/config.json`)).text();
          const tokenizer = new Uint8Array(await (await fetch(`${root}/tokenizer.json`)).arrayBuffer());
          const weights = new Uint8Array(await (await fetch(`${root}/model.safetensors`)).arrayBuffer());
          return wasm.WasmModel.load(weights, tokenizer, config, encoder);
        };
        const old = await load("/bench/gliner2.5-small-v1");
        try {
          const t0 = performance.now();
          const replacement = await load("/bench/gliner2.5-base-v1");
          const switchMs = Math.round(performance.now() - t0);
          try {
            const out = JSON.parse(replacement.classify(input, task));
            if (!out[0]?.labels?.length) throw new Error("Switched model did not perform inference");
            return { switchMs, wasmLinearMiB: exports.memory.buffer.byteLength / 1048576 };
          } finally { replacement.free(); }
        } finally { old.free(); }
      }, { task, input }));
      reports.push({ from: "small", to: "base", status: "passed", ...switching });
    } catch (error) {
      reports.push({ from: "small", to: "base", status: "failed", error: String(error) });
    } finally {
      await page.close().catch(() => {});
    }
    for (const report of reports) console.log(JSON.stringify({ proof: "browser-model-memory", ...report }));
    const failed = reports.filter((entry) => entry.status !== "passed");
    if (failed.length) throw new Error(`${failed.length} real browser model memory measurements failed; inspect report lines`);
  } finally {
    await CDP.detach();
  }
});
