# Real-model inference-stage profiling

The native server uses `GLINER_PROFILE_STAGES=1` to emit ASCII-framed
`GLINER_STAGE<TAB>name<TAB>duration_microseconds` lines on stderr.
It does **not** enable profiling by default, and the WASM target compiles
the native-only observability feature away.

`python benchmarks/profile_stages.py` drives the real Rust HTTP server
against a SHA-256-verified immutable checkpoint. It groups events by completed
`extract_total` record, discards warmups, and emits per-case raw timings,
median and min/max for each stage. The instrumentation itself introduces
a small overhead: measurements are a *relative attribution guide*, not
the source of absolute uninstrumented speed claims.

Stage hierarchy:

- `extract_total`: end-to-end Rust-side processing (not HTTP transport)
- `prepare` / `encoder` / `post_encoder`: disjoint top-level spans
- `deberta_position_setup`, `deberta_attention`, `deberta_ffn`: subsets
  of `encoder`; repeated attention and FFN events are summed per extraction
- `boundary_total`: subset of `post_encoder`
- `boundary_encode`, `boundary_marginals`, `boundary_pool`,
  `boundary_shared_score`: major components of `boundary_total`
- `explicit_shared_score`: repeated explicit-span scoring during decoding,
  counted in `post_encoder` and not in the initial `boundary_total`

Never add nested stages together as though they were disjoint. The
independent whole-model benchmark remains the admission boundary for
performance changes, and Python/Rust full parity remains the correctness gate.
