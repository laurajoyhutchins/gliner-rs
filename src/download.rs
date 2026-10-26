//! Downloads checkpoints from Hugging Face into the shared model cache.
//!
//! By default, legacy CLI downloads resolve a mutable `main` revision and
//! reuse existing files without independent provenance checks. For
//! reproducible, fail-closed loading set `GLINER_MODEL_LOCK` to a JSON lock
//! containing an immutable commit revision and SHA-256 for every required file.
//! Locked checkpoints occupy a revision-specific cache directory.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::model_path::VariantDef;

/// Files every checkpoint needs in order to load.
const REQUIRED_FILES: &[&str] = &["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"];
static NEXT_DOWNLOAD: AtomicU64 = AtomicU64::new(0);

/// Independently maintained, reviewable checkpoint provenance.
/// SHA-256 digests must originate from trusted model artifacts, not an
/// unverified download made by this tool.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointLock {
    version: u32,
    repo: String,
    revision: String,
    sha256: HashMap<String, String>,
}

fn parse_lock(raw: &str, variant: &VariantDef) -> Result<CheckpointLock> {
    let lock: CheckpointLock = serde_json::from_str(raw).context("parsing checkpoint lock JSON")?;
    if lock.version != 1 {
        bail!("unsupported checkpoint lock version {}; expected 1", lock.version);
    }
    if lock.repo != variant.hf_repo {
        bail!("checkpoint lock repo {:?} does not match variant repo {:?}", lock.repo, variant.hf_repo);
    }
    if lock.revision.len() != 40 || !lock.revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("checkpoint lock revision must be a 40-character Git commit SHA");
    }
    if lock.sha256.len() != REQUIRED_FILES.len() {
        bail!("checkpoint lock must contain exactly the four required file hashes");
    }
    for path in REQUIRED_FILES {
        let hash = lock.sha256.get(*path).with_context(|| format!("checkpoint lock is missing SHA-256 for {path}"))?;
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("checkpoint lock SHA-256 for {path} must be 64 hexadecimal characters");
        }
    }
    Ok(lock)
}

fn configured_lock(variant: &VariantDef) -> Result<Option<CheckpointLock>> {
    let Some(path) = std::env::var_os("GLINER_MODEL_LOCK") else { return Ok(None) };
    if path.is_empty() {
        bail!("GLINER_MODEL_LOCK is empty; provide a JSON lock file path or unset it");
    }
    let path = PathBuf::from(path);
    let raw = std::fs::read_to_string(&path).with_context(|| format!("reading checkpoint lock {}", path.display()))?;
    Ok(Some(parse_lock(&raw, variant)?))
}

fn lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    out
}

fn sha256_reader(reader: &mut impl Read) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buffer).context("reading checkpoint bytes for SHA-256")?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(lower_hex(&hasher.finalize()))
}

fn verify_cached_file(path: &Path, expected: &str) -> Result<()> {
    let mut file = File::open(path).with_context(|| format!("opening cached checkpoint {}", path.display()))?;
    let actual = sha256_reader(&mut file)?;
    if !actual.eq_ignore_ascii_case(expected) {
        bail!(
            "cached checkpoint {} does not match its pinned SHA-256 (expected {expected}, got {actual}); remove or quarantine this file before retrying",
            path.display()
        );
    }
    Ok(())
}

/// `$XDG_CACHE_HOME/gliner-rs`, defaulting to `~/.cache/gliner-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("gliner-rs"))
}

/// The checkpoint directory name for a variant: the repo name portion of
/// its `hf_repo` (e.g. `fastino/gliner2.5-multi-v1` -> `gliner2.5-multi-v1`).
fn checkpoint_dir_name(variant: &VariantDef) -> &str {
    variant.hf_repo.rsplit('/').next().unwrap_or(variant.hf_repo)
}

/// Legacy cache root for a variant. Pinned revisions live in subdirectories
/// of this root, named by their 40-character commit SHA.
pub fn variant_cache_dir(variant: &VariantDef) -> Option<PathBuf> {
    Some(cache_dir()?.join(checkpoint_dir_name(variant)))
}

fn fetch_to_file(url: &str, dest: &Path, expected_sha256: Option<&str>) -> Result<()> {
    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());

    eprint!("  {} ...", dest.file_name().unwrap_or_default().to_string_lossy());
    std::io::stderr().flush().ok();
    let written = stage_verified_file(resp.into_reader(), dest, len, expected_sha256)?;
    eprintln!(" done ({written} bytes)");
    Ok(())
}

/// Deterministic I/O boundary: only publishes fully received, verified bytes.
/// Separated from HTTP so cache lifecycle can be tested with in-memory input.
fn stage_verified_file(
    mut reader: impl Read,
    dest: &Path,
    expected_len: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<u64> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    // An exclusive, process-specific temporary file avoids collisions among
    // simultaneous downloaders, unlike a single fixed `.part` sibling.
    let temporary_name = format!(
        "{}.{}.{}.part",
        dest.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        NEXT_DOWNLOAD.fetch_add(1, Ordering::Relaxed)
    );
    let part = dest.with_file_name(temporary_name);
    let mut file = OpenOptions::new().write(true).create_new(true).open(&part)
        .with_context(|| format!("creating exclusive temporary file {}", part.display()))?;

    let written = (|| -> Result<u64> {
        let mut buf = [0u8; 1 << 16];
        let mut written = 0u64;
        let mut hasher = expected_sha256.map(|_| Sha256::new());
        loop {
            let n = reader.read(&mut buf).with_context(|| format!("reading download for {}", dest.display()))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
            if let Some(hasher) = &mut hasher {
                hasher.update(&buf[..n]);
            }
            written += n as u64;
        }
        if let Some(expected_len) = expected_len {
            if written != expected_len {
                bail!(
                    "short download for {}: got {written} bytes, expected {expected_len}",
                    dest.display()
                );
            }
        }
        if let (Some(expected), Some(hasher)) = (expected_sha256, hasher) {
            let actual = lower_hex(&hasher.finalize());
            if !actual.eq_ignore_ascii_case(expected) {
                bail!("downloaded checkpoint {} SHA-256 mismatch: expected {expected}, got {actual}", dest.display());
            }
        }
        file.sync_all().with_context(|| format!("syncing {}", part.display()))?;
        Ok(written)
    })();
    drop(file);

    // Never expose incomplete or unverified bytes as the final cache entry.
    let written = match written {
        Ok(written) => written,
        Err(error) => {
            let _ = std::fs::remove_file(&part);
            return Err(error);
        }
    };
    if let Err(error) = std::fs::rename(&part, dest) {
        let _ = std::fs::remove_file(&part);
        return Err(error).with_context(|| format!("renaming {} to {}", part.display(), dest.display()));
    }
    Ok(written)
}

/// Return a variant's checkpoint directory, downloading missing files.
///
/// With `GLINER_MODEL_LOCK`, the revision is immutable, each cached file is
/// checked before use, and each download must match a trusted SHA-256 before
/// atomically replacing a missing cache entry. Corrupted cached files are
/// reported, not silently accepted or overwritten.
///
/// Without a lock, compatibility mode retains the original mutable `main`
/// behavior and should not be mistaken for provenance-verified loading.
pub fn download_variant(variant: &VariantDef) -> Result<PathBuf> {
    let lock = configured_lock(variant)?;
    let mut dir = variant_cache_dir(variant).context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    if let Some(lock) = &lock {
        dir.push(&lock.revision);
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let mut announced = false;
    for file in REQUIRED_FILES {
        let dest = dir.join(file);
        if dest.is_file() {
            if let Some(lock) = &lock {
                verify_cached_file(&dest, &lock.sha256[*file])?;
            }
            continue;
        }
        if !announced {
            eprintln!("Downloading {} from https://huggingface.co/{} into {}", variant.key, variant.hf_repo, dir.display());
            announced = true;
        }
        if std::env::var_os("GLINER_OFFLINE").is_some() {
            bail!(
                "{} is missing from {} and GLINER_OFFLINE is set; unset it to allow downloading, or pass --model",
                file,
                dir.display()
            );
        }
        let revision = lock.as_ref().map_or("main", |lock| lock.revision.as_str());
        let expected = lock.as_ref().map(|lock| lock.sha256[*file].as_str());
        let url = format!("https://huggingface.co/{}/resolve/{revision}/{file}", variant.hf_repo);
        fetch_to_file(&url, &dest, expected)?;
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const TEST_HASH: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn variant() -> VariantDef {
        VariantDef { key: "test", hf_repo: "example/checkpoint" }
    }

    fn lock_json() -> String {
        let hashes: HashMap<&str, &str> = REQUIRED_FILES.iter().map(|path| (*path, TEST_HASH)).collect();
        serde_json::json!({
            "version": 1,
            "repo": "example/checkpoint",
            "revision": REVISION,
            "sha256": hashes
        }).to_string()
    }

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(sha256_reader(&mut Cursor::new(b"hello")).unwrap(), TEST_HASH);
    }

    #[test]
    fn staged_bytes_publish_only_after_length_and_digest_agree() {
        let dir = std::env::temp_dir().join(format!(
            "gliner-rs-stage-test-{}-{}",
            std::process::id(),
            NEXT_DOWNLOAD.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let good = dir.join("good.json");
        let corrupt = dir.join("corrupt.json");
        let short = dir.join("short.json");

        let accepted = stage_verified_file(Cursor::new(b"hello"), &good, Some(5), Some(TEST_HASH));
        let rejected_hash = stage_verified_file(Cursor::new(b"other"), &corrupt, Some(5), Some(TEST_HASH));
        let rejected_len = stage_verified_file(Cursor::new(b"hello"), &short, Some(6), Some(TEST_HASH));
        let staged_leftovers: Vec<_> = std::fs::read_dir(&dir).unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "part"))
            .collect();

        assert_eq!(accepted.unwrap(), 5);
        assert_eq!(std::fs::read(&good).unwrap(), b"hello");
        assert!(rejected_hash.unwrap_err().to_string().contains("SHA-256 mismatch"));
        assert!(rejected_len.unwrap_err().to_string().contains("short download"));
        assert!(!corrupt.exists());
        assert!(!short.exists());
        assert!(staged_leftovers.is_empty(), "stale download staging files: {staged_leftovers:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cached_bytes_must_match_the_independent_digest() {
        let dir = std::env::temp_dir().join(format!(
            "gliner-rs-lock-test-{}-{}",
            std::process::id(),
            NEXT_DOWNLOAD.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, b"hello").unwrap();
        let valid = verify_cached_file(&path, TEST_HASH);
        std::fs::write(&path, b"corrupted").unwrap();
        let invalid = verify_cached_file(&path, TEST_HASH);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(valid.is_ok());
        assert!(invalid.unwrap_err().to_string().contains("does not match its pinned SHA-256"));
    }

    #[test]
    fn accepts_complete_pinned_manifest() {
        let lock = parse_lock(&lock_json(), &variant()).unwrap();
        assert_eq!(lock.revision, REVISION);
        assert_eq!(lock.sha256["model.safetensors"], TEST_HASH);
    }

    #[test]
    fn rejects_wrong_repository_or_unpinned_revision() {
        let mut value: serde_json::Value = serde_json::from_str(&lock_json()).unwrap();
        value["repo"] = "other/checkpoint".into();
        assert!(parse_lock(&value.to_string(), &variant()).is_err());
        value["repo"] = "example/checkpoint".into();
        value["revision"] = "main".into();
        assert!(parse_lock(&value.to_string(), &variant()).is_err());
    }

    #[test]
    fn rejects_incomplete_or_malformed_hashes() {
        let mut value: serde_json::Value = serde_json::from_str(&lock_json()).unwrap();
        value["sha256"].as_object_mut().unwrap().remove("model.safetensors");
        assert!(parse_lock(&value.to_string(), &variant()).is_err());
        value["sha256"]["model.safetensors"] = "not-a-digest".into();
        assert!(parse_lock(&value.to_string(), &variant()).is_err());
    }

    #[test]
    fn rejects_wrong_lock_version_and_extra_fields() {
        let mut value: serde_json::Value = serde_json::from_str(&lock_json()).unwrap();
        value["version"] = 2.into();
        assert!(parse_lock(&value.to_string(), &variant()).is_err());
        value["version"] = 1.into();
        value["extra"] = true.into();
        assert!(parse_lock(&value.to_string(), &variant()).is_err());
    }
}
