/** Browser checkpoint locks use the same version-1 JSON Schema as the CLI. */
import manifest from "./checkpoint-locks.json";

export const CHECKPOINT_FILES = ["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"] as const;
export type CheckpointFile = (typeof CHECKPOINT_FILES)[number];

export interface CheckpointLock {
  readonly version: 1;
  readonly repo: string;
  readonly revision: string;
  readonly sha256: Readonly<Record<CheckpointFile, string>>;
}

const byRepo = new Map<string, CheckpointLock>(
  manifest.map((entry) => [entry.repo, entry as CheckpointLock]),
);

/** Never silently fall back to a mutable revision for downloadable models. */
export function checkpointLock(repo: string): CheckpointLock {
  const lock = byRepo.get(repo);
  if (!lock) throw new Error(`No pinned checkpoint lock for ${repo}; download blocked`);
  return lock;
}
