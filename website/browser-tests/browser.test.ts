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

async function fixtureFile(file: FileName): Promise<string> {
  const output = join(tmp, file.replaceAll("/", "_"));
  const expected = checkpoint.sha256[file];
  if (!/^[a-f0-9]{64}$/.test(expected)) throw new Error(`Unpinned or invalid SHA-256 for ${file}`);
  const url = `https://huggingface.co/${checkpoint.repo}/resolve/${checkpoint.revision}/${file}`;
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
