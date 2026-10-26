//! Opt-in native performance observations only. No model/state mutations.
//! Each event is a completed duration measured on the same process and host.
//! CPU execution is synchronous; do not use these timings as GPU kernel timings.
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("GLINER_PROFILE_STAGES").is_some())
}

pub(crate) fn begin() -> Option<Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    if enabled() { return Some(Instant::now()); }
    None
}

pub(crate) fn end(label: &'static str, start: Option<Instant>) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(start) = start {
        // ASCII-only tab framing avoids ambiguity with unrelated server logs.
        // The scraper only recognizes events with this exact prefix.
        eprintln!("GLINER_STAGE\t{}\t{}", label, start.elapsed().as_micros());
    }
    #[cfg(target_arch = "wasm32")]
    { let _ = (label, start); }
}
