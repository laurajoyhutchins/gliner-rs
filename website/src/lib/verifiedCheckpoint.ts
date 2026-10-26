/**
 * Stream a pinned model file into one output buffer while incrementally
 * hashing received bytes. Corrupt cache entries are evicted and fetched again.
 * No unverified bytes are published to the Cache API.
 */
import { sha256 } from "@noble/hashes/sha2.js";
import type { CheckpointFile, CheckpointLock } from "./checkpointLocks";

/** An immutable HF resolve URL: no mutable main references in the downloader. */
export function pinnedCheckpointUrl(lock: CheckpointLock, file: CheckpointFile): string {
  return `https://huggingface.co/${lock.repo}/resolve/${lock.revision}/${file}`;
}

const CACHE_NAME = "gliner-rs-verified-checkpoints-v2";
const MAX_BYTES = 2_500_000_000; // all supported browser model files are below this bound

type CheckpointCache = Pick<Cache, "match" | "put" | "delete">;
export interface DownloadDependencies {
  fetcher?: typeof fetch;
  cache?: CheckpointCache | null;
}
export interface ByteProgress {
  loaded: number;
  total: number | null;
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function contentLength(response: Response): number | null {
  const text = response.headers.get("content-length");
  if (text === null) return null;
  const len = Number(text);
  if (!Number.isSafeInteger(len) || len < 0 || len > MAX_BYTES) throw new Error("Invalid checkpoint Content-Length");
  return len;
}

async function verifiedBytes(
  response: Response,
  expected: string,
  progress: (p: ByteProgress) => void,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const length = contentLength(response);
  if (!response.body) throw new Error("Checkpoint response has no readable body");
  const reader = response.body.getReader();
  const digest = sha256.create();
  let output: Uint8Array | null = length === null ? null : new Uint8Array(length);
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  progress({ loaded, total: length });

  try {
    for (;;) {
      signal?.throwIfAborted();
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value.byteLength;
      if (loaded > MAX_BYTES || (length !== null && loaded > length)) {
        throw new Error("Checkpoint response exceeds its announced size or browser limit");
      }
      digest.update(value);
      if (output) output.set(value, loaded - value.byteLength);
      else chunks.push(value);
      progress({ loaded, total: length });
    }
  } finally {
    reader.releaseLock();
  }

  signal?.throwIfAborted();
  if (length !== null && loaded !== length) {
    throw new Error(`Incomplete checkpoint: received ${loaded} bytes, expected ${length}`);
  }
  const actual = hex(digest.digest());
  if (actual.toLowerCase() !== expected.toLowerCase()) {
    throw new Error(`Checkpoint SHA-256 mismatch: expected ${expected}, received ${actual}`);
  }
  if (!output) {
    // No Content-Length: exactly one final contiguous allocation. Never
    // allocate a second whole-file buffer during the streaming loop.
    output = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      output.set(chunk, offset);
      offset += chunk.byteLength;
    }
  }
  return output;
}

async function defaultCache(): Promise<CheckpointCache | null> {
  if (typeof caches === "undefined") return null;
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // private browsing and quota restrictions cannot block inference
  }
}

export async function fetchVerifiedCheckpoint(
  lock: CheckpointLock,
  file: CheckpointFile,
  onProgress: (p: ByteProgress) => void,
  signal?: AbortSignal,
  deps: DownloadDependencies = {},
): Promise<Uint8Array> {
  const url = pinnedCheckpointUrl(lock, file);
  const cache = deps.cache === undefined ? await defaultCache() : deps.cache;
  signal?.throwIfAborted();
  if (cache) {
    try {
      const cached = await cache.match(url);
      if (cached) return await verifiedBytes(cached, lock.sha256[file], onProgress, signal);
    } catch (error) {
      if (signal?.aborted) throw error;
      // A cache read failure or digest mismatch must never be admitted.
      try { await cache.delete(url); } catch { /* Cache API failures cannot block a clean retry. */ }
    }
  }

  const response = await (deps.fetcher ?? fetch)(url, { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`Checkpoint download ${file}: HTTP ${response.status}`);
  const bytes = await verifiedBytes(response, lock.sha256[file], onProgress, signal);
  signal?.throwIfAborted();
  if (cache) {
    try {
      await cache.put(url, new Response(bytes, {
        headers: { "content-type": "application/octet-stream", "content-length": String(bytes.byteLength) },
      }));
    } catch {
      // The model is verified in memory; storage quota is not a model error.
    }
  }
  return bytes;
}

export async function clearVerifiedCheckpointCache(): Promise<void> {
  if (typeof caches !== "undefined") await caches.delete(CACHE_NAME);
}

export async function verifiedCheckpointCacheKeys(): Promise<string[]> {
  if (typeof caches === "undefined") return [];
  const cache = await defaultCache();
  if (!cache) return [];
  // keys() is available on a real Cache; tests inject just the three used
  // operations above.
  return (await (cache as Cache).keys()).map((request) => request.url);
}
