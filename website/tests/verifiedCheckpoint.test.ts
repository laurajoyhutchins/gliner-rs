import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";
import { fetchVerifiedCheckpoint, pinnedCheckpointUrl } from "../src/lib/verifiedCheckpoint.ts";
import type { CheckpointLock } from "../src/lib/checkpointLocks.ts";

const good = new TextEncoder().encode("hello");
const corrupted = new TextEncoder().encode("other");
const sha = createHash("sha256").update(good).digest("hex");
const lock: CheckpointLock = {
  version: 1,
  repo: "example/test",
  revision: "0123456789abcdef0123456789abcdef01234567",
  sha256: {
    "config.json": sha,
    "encoder_config/config.json": sha,
    "tokenizer.json": sha,
    "model.safetensors": sha,
  },
};
const file = "config.json";
const url = pinnedCheckpointUrl(lock, file);

class MemoryCache {
  readonly entries = new Map<string, Uint8Array>();
  puts = 0;
  deletes = 0;
  async match(key: RequestInfo | URL): Promise<Response | undefined> {
    const bytes = this.entries.get(String(key));
    return bytes ? new Response(bytes) : undefined;
  }
  async put(key: RequestInfo | URL, response: Response): Promise<void> {
    this.puts++;
    this.entries.set(String(key), new Uint8Array(await response.arrayBuffer()));
  }
  async delete(key: RequestInfo | URL): Promise<boolean> {
    this.deletes++;
    return this.entries.delete(String(key));
  }
}

function transport(content: Uint8Array, announced?: number): typeof fetch {
  return (async () => new Response(content, {
    status: 200,
    headers: announced === undefined ? {} : { "content-length": String(announced) },
  })) as typeof fetch;
}
const progress = () => {};

test("cache key uses the 40-character immutable revision, never main", () => {
  assert.ok(url.includes("/resolve/" + lock.revision + "/"));
  assert.ok(!url.includes("/resolve/main/"));
});

test("verifies and publishes a downloaded file only after digest check", async () => {
  const cache = new MemoryCache();
  const got = await fetchVerifiedCheckpoint(lock, file, progress, undefined, {
    cache, fetcher: transport(good, good.length),
  });
  assert.deepEqual(got, good);
  assert.equal(cache.puts, 1);
  assert.deepEqual(cache.entries.get(url), good);
});

test("a corrupt cached entry is evicted, not admitted, then repaired from pinned source", async () => {
  const cache = new MemoryCache();
  cache.entries.set(url, corrupted);
  const got = await fetchVerifiedCheckpoint(lock, file, progress, undefined, {
    cache, fetcher: transport(good, good.length),
  });
  assert.deepEqual(got, good);
  assert.equal(cache.deletes, 1);
  assert.deepEqual(cache.entries.get(url), good);
});

test("wrong downloaded digest fails closed without cache admission", async () => {
  const cache = new MemoryCache();
  await assert.rejects(
    fetchVerifiedCheckpoint(lock, file, progress, undefined, { cache, fetcher: transport(corrupted, corrupted.length) }),
    /SHA-256 mismatch/,
  );
  assert.equal(cache.puts, 0);
});

test("truncated download is rejected even when the bytes themselves hash correctly", async () => {
  const cache = new MemoryCache();
  await assert.rejects(
    fetchVerifiedCheckpoint(lock, file, progress, undefined, { cache, fetcher: transport(good, good.length + 1) }),
    /Incomplete checkpoint/,
  );
  assert.equal(cache.puts, 0);
});

test("interrupted stream never publishes partial bytes", async () => {
  const cache = new MemoryCache();
  const fetcher = (async () => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(good.subarray(0, 2));
      controller.error(new Error("connection interrupted"));
    },
  }), { status: 200 })) as typeof fetch;
  await assert.rejects(
    fetchVerifiedCheckpoint(lock, file, progress, undefined, { cache, fetcher }),
    /connection interrupted/,
  );
  assert.equal(cache.puts, 0);
});

test("abort before network completion prevents cache admission", async () => {
  const cache = new MemoryCache();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    fetchVerifiedCheckpoint(lock, file, progress, controller.signal, { cache, fetcher: transport(good, good.length) }),
    (e: unknown) => e instanceof DOMException && e.name === "AbortError",
  );
  assert.equal(cache.puts, 0);
});

test("browser cache miss works without Content-Length and reports the right bytes", async () => {
  const cache = new MemoryCache();
  const events: { loaded: number; total: number | null }[] = [];
  const got = await fetchVerifiedCheckpoint(lock, file, (p) => { events.push(p); }, undefined, {
    cache, fetcher: transport(good),
  });
  assert.deepEqual(got, good);
  assert.equal(events.at(-1)?.loaded, good.length);
  assert.equal(cache.puts, 1);
});
