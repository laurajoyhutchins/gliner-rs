# Real Chromium model memory benchmarks

Run with `GLINER_REAL_CHECKPOINT=1 GLINER_MEMORY_BENCHMARK=1 npm test`
after installing the pinned browser harness and Chromium.

The benchmark downloads the exact immutable SHA-256-verified original
`small`, `base`, and `multilingual` checkpoints. Each starts in a fresh
Chromium tab. It records WASM model initialization latency, from-localhost
fetch plus init/inference wall time, final WebAssembly linear-memory size,
and **sampled** process-family VmRSS during execution. It additionally
measures a real co-resident small → base swap while preserving the old
model until the new one has initialized. Unsafe high-risk swaps are denied
at the UI preflight and are **not** attempted as intentional browser OOMs.

VmRSS summed over Chromium processes includes shared pages, is not
unique/private resident memory, and 75ms polling can miss shorter spikes.
Measurements are benchmark evidence, not hard admission thresholds or
cross-device memory safety guarantees. Network checkpoint acquisition
happens before timing; it is recorded by size/hash but not counted in
cold-load duration. The exact-head CI logs print one JSON report per
model and switch for independent readback.
