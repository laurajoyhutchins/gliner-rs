/** In-browser GLiNER model loading with pinned Hugging Face checkpoints.
 * Network models are content-verified before use or cache admission. The
 * active model is replaced only after a new model finishes initializing.
 */

import { createSignal } from "solid-js";
import type { ModelDef } from "./models";
import { CHECKPOINT_FILES, checkpointLock } from "./checkpointLocks";
import { fetchVerifiedCheckpoint, clearVerifiedCheckpointCache, verifiedCheckpointCacheKeys } from "./verifiedCheckpoint";
import { ModelLifecycle } from "./modelLifecycle";
import {
  fsAccessSupported,
  getSubdirectory,
  listSubdirectories,
  loadRememberedDirectory,
  pickDirectory,
  queryReadPermission,
  readRelativeFile,
  rememberDirectory,
  requestReadPermission,
} from "./localFs";

export { fsAccessSupported };

export interface WasmModel {
  setCharSplit(on: boolean): void;
  extractCli(
    text: string,
    entities: string[],
    relations: string[],
    structures: string[],
    classify: string[],
    legacyStructures: boolean,
    optsJson: string,
  ): string;
  classify(text: string, tasksJson: string): string;
  free(): void;
}

type WasmExports = {
  default: (module_or_path?: unknown) => Promise<unknown>;
  WasmModel: { load(weights: Uint8Array, tokenizer: Uint8Array, configJson: string, encoderConfigJson: string): WasmModel };
  init_panic_hook: () => void;
};

let wasmReady: Promise<WasmExports> | null = null;

function loadWasm(): Promise<WasmExports> {
  if (!wasmReady) {
    wasmReady = (async () => {
      const mod = (await import("../wasm-pkg/gliner_rs.js")) as unknown as WasmExports;
      await mod.default();
      mod.init_panic_hook();
      return mod;
    })();
  }
  return wasmReady;
}

export interface DownloadProgress {
  file: string;
  fileIndex: number;
  fileCount: number;
  loaded: number;
  total: number | null;
}

export interface LoadedModel {
  model: WasmModel;
  def: ModelDef;
  source: "hf" | "local";
  revision?: string;
  libraryHandle?: FileSystemDirectoryHandle;
}

/** A local mirror of several checkpoints, picked once as a single parent
 * directory (e.g. the gliner-rs cache dir, or a folder someone rsynced from
 * one): `<root>/<hf-repo-basename>/config.json`, `.../model.safetensors`,
 * etc — one subdirectory per checkpoint, named after the last segment of
 * its Hugging Face repo id (`fastino/gliner2.5-small-v1` -> `gliner2.5-small-v1`),
 * the same layout the gliner-rs cache directory and `gliner --model <dir>`
 * both use.
 *
 * Held as a [File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API)
 * directory handle rather than an uploaded file list: picking it never
 * enumerates the whole tree (a checkpoint mirror can hold thousands of HF
 * cache blob files), and the handle itself is what's remembered across a
 * reload — see [`ensureLocalLibraryResumeAttempted`]. */
export interface LocalLibrary {
  rootName: string;
  handle: FileSystemDirectoryHandle;
  checkpointNames: string[];
}

function hfRepoBasename(hfRepo: string): string {
  return hfRepo.split("/").pop() ?? hfRepo;
}

/** Which of `models` (by Hugging Face repo basename) are present in a
 * local library, in `models` order. */
export function availableLocally(library: LocalLibrary, models: ModelDef[]): ModelDef[] {
  const names = new Set(library.checkpointNames);
  return models.filter((m) => names.has(hfRepoBasename(m.hfRepo)));
}

/** Loads one checkpoint from a local library directory — no network
 * involved. Expects the same four files a Hugging Face checkpoint has:
 * `config.json`, `encoder_config/config.json`, `tokenizer.json`,
 * `model.safetensors`. */
export async function loadModelFromHandle(
  def: ModelDef,
  library: LocalLibrary,
  onProgress?: (p: DownloadProgress) => void,
): Promise<LoadedModel> {
  const wasm = await loadWasm();
  const root = await getSubdirectory(library.handle, hfRepoBasename(def.hfRepo));
  // The CLI's pinned cache nests files under a commit SHA. Continue to
  // support explicit older/manual directories at the model root.
  const checkpointDir = await root.getDirectoryHandle(checkpointLock(def.hfRepo).revision).catch(() => root);
  const bytes: Record<string, Uint8Array> = {};
  for (let i = 0; i < CHECKPOINT_FILES.length; i++) {
    const name = CHECKPOINT_FILES[i];
    const file = await readRelativeFile(checkpointDir, name);
    onProgress?.({ file: name, fileIndex: i, fileCount: CHECKPOINT_FILES.length, loaded: 0, total: file.size });
    bytes[name] = new Uint8Array(await file.arrayBuffer());
    onProgress?.({ file: name, fileIndex: i, fileCount: CHECKPOINT_FILES.length, loaded: file.size, total: file.size });
  }
  const configJson = new TextDecoder().decode(bytes["config.json"]);
  const encoderConfigJson = new TextDecoder().decode(bytes["encoder_config/config.json"]);
  const model = wasm.WasmModel.load(bytes["model.safetensors"], bytes["tokenizer.json"], configJson, encoderConfigJson);
  return { model, def, source: "local", libraryHandle: library.handle };
}

/** Downloads (or reuses cached) checkpoint files for `def` and loads them
 * into a fresh `WasmModel`. Reports per-file byte progress via `onProgress`. */
export async function loadModel(
  def: ModelDef,
  onProgress?: (p: DownloadProgress) => void,
  signal?: AbortSignal,
): Promise<LoadedModel> {
  const wasm = await loadWasm();
  const lock = checkpointLock(def.hfRepo);
  const files = {} as Record<(typeof CHECKPOINT_FILES)[number], Uint8Array>;
  for (let i = 0; i < CHECKPOINT_FILES.length; i++) {
    signal?.throwIfAborted();
    const file = CHECKPOINT_FILES[i];
    files[file] = await fetchVerifiedCheckpoint(
      lock, file, ({ loaded, total }) =>
        onProgress?.({ file, fileIndex: i, fileCount: CHECKPOINT_FILES.length, loaded, total }),
      signal,
    );
  }
  signal?.throwIfAborted();
  const configJson = new TextDecoder().decode(files["config.json"]);
  const encoderConfigJson = new TextDecoder().decode(files["encoder_config/config.json"]);
  const model = wasm.WasmModel.load(files["model.safetensors"], files["tokenizer.json"], configJson, encoderConfigJson);
  return { model, def, source: "hf", revision: lock.revision };
}

/** Clears only verified cached bytes; an already loaded model keeps running. */
export async function clearModelCache(): Promise<void> {
  await clearVerifiedCheckpointCache();
}

export async function cachedModelKeys(): Promise<string[]> {
  return verifiedCheckpointCacheKeys();
}

/** Shared "which model is currently loaded" state, so every demo tab can
 * reuse one loaded model instead of re-downloading per tab. */
const [current, setCurrent] = createSignal<LoadedModel | null>(null);
export const currentModel = current;
const lifecycle = new ModelLifecycle(current, setCurrent);

export type ModelSource = "hf" | "local";

/** What's remembered across reloads: which source, which Hugging Face
 * model, and (since a local directory's file handles can't survive a
 * reload) which local root/checkpoint name to look for once the user picks
 * that folder again. */
interface PersistedState {
  source: ModelSource;
  hfModelKey?: string;
  localRootName?: string;
  localModelKey?: string;
}

const LS_KEY = "gliner-rs:demo-model-v1";

function readPersisted(): PersistedState {
  if (typeof localStorage === "undefined") return { source: "hf" };
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { source: "hf" };
    const parsed = JSON.parse(raw) as Partial<PersistedState>;
    return { source: parsed.source === "local" ? "local" : "hf", ...parsed };
  } catch {
    return { source: "hf" };
  }
}

function writePersisted(patch: Partial<PersistedState>) {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({ ...readPersisted(), ...patch }));
  } catch {
    /* quota / private-mode — resuming after reload just won't work this time */
  }
}

/** What was loaded last session, for the "resume on refresh" prompt — read
 * once per `ModelPicker` mount (not reactive: it only matters right after a
 * page load, before anything's picked again this session). */
export const persistedState = readPersisted;

/** The source (Hugging Face vs. local directory) the last successful load
 * used, shared across every `ModelPicker` instance so switching demo tabs
 * keeps whichever one the user picked instead of resetting to "Hugging
 * Face" each time — and restored from localStorage on page load. */
const [preferredSource, setPreferredSource] = createSignal<ModelSource>(readPersisted().source);
export { preferredSource };

/** The local library directory the user picked, shared across every
 * `ModelPicker` instance — pick it once, then every demo tab's "local
 * directory" source lists the checkpoints found inside it, no re-picking
 * per tab or per model. */
const [localLibrary, setLocalLibrary] = createSignal<LocalLibrary | null>(null);
export { localLibrary };

/** Set once a remembered directory handle is found but its permission
 * wasn't silently granted — the user needs one click (a `requestPermission`
 * call needs a user gesture; the browser dialog itself is skipped since the
 * directory was already picked before). See [`grantPendingLibraryAccess`]. */
const [pendingLibrary, setPendingLibrary] = createSignal<LocalLibrary | null>(null);
export { pendingLibrary };

async function toLibrary(handle: FileSystemDirectoryHandle): Promise<LocalLibrary> {
  return { rootName: handle.name, handle, checkpointNames: await listSubdirectories(handle) };
}

export async function selectLocalLibrary(): Promise<LocalLibrary | null> {
  const handle = await pickDirectory();
  if (!handle) return null; // user dismissed the picker
  await rememberDirectory(handle);
  const library = await toLibrary(handle);
  setLocalLibrary(library);
  setPendingLibrary(null);
  writePersisted({ localRootName: library.rootName });
  return library;
}

/** Tries to reopen the directory handle remembered from a previous session,
 * entirely without a file picker. If the browser silently still grants read
 * permission (common — Chromium persists it), the library becomes active
 * right away; otherwise it's parked in `pendingLibrary` for one
 * [`grantPendingLibraryAccess`] click. Safe to call repeatedly — only the
 * first call in a session does anything. */
let resumeAttempted = false;
export function ensureLocalLibraryResumeAttempted(): void {
  if (resumeAttempted || localLibrary()) return;
  resumeAttempted = true;
  void (async () => {
    const handle = await loadRememberedDirectory();
    if (!handle) return;
    const granted = (await queryReadPermission(handle)) === "granted";
    const library = granted
      ? await toLibrary(handle)
      : { rootName: handle.name, handle, checkpointNames: [] as string[] };
    if (granted) {
      setLocalLibrary(library);
    } else {
      setPendingLibrary(library);
    }
  })();
}

/** Grants (or denies) access to the directory found by
 * [`ensureLocalLibraryResumeAttempted`] — call from a click handler, since
 * `requestPermission` needs a user gesture. No file picker dialog appears;
 * it's a lightweight browser permission prompt for the already-known
 * directory. */
export async function grantPendingLibraryAccess(): Promise<LocalLibrary | null> {
  const pending = pendingLibrary();
  if (!pending) return null;
  const granted = (await requestReadPermission(pending.handle)) === "granted";
  if (!granted) return null;
  const library = await toLibrary(pending.handle);
  setLocalLibrary(library);
  setPendingLibrary(null);
  return library;
}

export async function selectModel(def: ModelDef, onProgress?: (p: DownloadProgress) => void): Promise<LoadedModel> {
  const existing = current();
  const revision = checkpointLock(def.hfRepo).revision;
  if (existing?.source === "hf" && existing.def.hfRepo === def.hfRepo && existing.revision === revision) {
    lifecycle.cancelPending();
    setPreferredSource("hf");
    writePersisted({ source: "hf", hfModelKey: def.key });
    return existing;
  }
  return lifecycle.replace(
    (signal) => loadModel(def, onProgress, signal),
    () => {
      setPreferredSource("hf");
      writePersisted({ source: "hf", hfModelKey: def.key });
    },
  );
}

export async function selectLocalModel(def: ModelDef, onProgress?: (p: DownloadProgress) => void): Promise<LoadedModel> {
  const library = localLibrary();
  if (!library) throw new Error("no local library directory selected");
  if (!library.checkpointNames.includes(hfRepoBasename(def.hfRepo))) {
    throw new Error(`${def.label} not found in ${library.rootName}`);
  }
  const existing = current();
  if (existing?.source === "local" && existing.def.hfRepo === def.hfRepo && existing.libraryHandle === library.handle) {
    lifecycle.cancelPending();
    setPreferredSource("local");
    writePersisted({ source: "local", localRootName: library.rootName, localModelKey: def.key });
    return existing;
  }
  return lifecycle.replace(
    async (signal) => {
      signal.throwIfAborted();
      const loaded = await loadModelFromHandle(def, library, onProgress);
      // Filesystem reads are not cancellable on all supported browsers, so
      // stale load completions are disposed by ModelLifecycle.replace.
      return loaded;
    },
    () => {
      setPreferredSource("local");
      writePersisted({ source: "local", localRootName: library.rootName, localModelKey: def.key });
    },
  );
}

export function unloadModel(): void {
  lifecycle.unload();
}
