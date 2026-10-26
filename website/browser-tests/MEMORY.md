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

## Observed runner results, 2026-10-26

[Passing real Chromium benchmark](https://github.com/laurajoyhutchins/gliner-rs/actions/runs/40386020046)
at `d8cc5534140be956bb59b731862be3d72c5f16e8`.

| Model / operation | Checkpoint weight bytes | wasm-bindgen model initialization | Localhost transfer + load + inference | Sampled peak browser process VmRSS | Final WASM linear memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small, cold | 295,567,700 | 628 ms | 1,362 ms | 1,359 MiB | 703 MiB |
| Base, cold | 774,366,564 | 1,123 ms | 3,033 ms | 2,809 MiB | 1,697 MiB |
| Multilingual, cold | 1,149,461,028 | 1,607 ms | 4,177 ms | 3,967 MiB | 2,503 MiB |
| Small → Base co-resident replacement | Previous plus 774,366,564 | 2,486 ms replacement | 4,330 ms | 3,198 MiB | 2,071 MiB |

Approximate baseline Chromium process-family VmRSS was 307 MiB in each
scenario. The report includes shared memory in every process' VmRSS and
can overcount physical pages shared across processes. The 75ms sampling
period can also miss shorter allocation spikes. These values are
environment-specific evidence, not browser-wide maxima or a promise of
success on a mobile device.

The conservative `2.5× checkpoint bytes + 256 MiB scratch` memory
preflight is intended to bound **WASM linear-memory risk**, not predict
total process RSS. In the observed cold tests, final WASM linear memory
was approximately 2.3–2.5× the stored weight bytes. The deliberately
high-risk base → multilingual co-resident transition remains refused
before downloading; it was not triggered just to induce a renderer OOM.
