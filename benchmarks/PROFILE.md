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

## First measured baseline: pinned GLiNER2.5-small

Source: [CI 40394924950](https://github.com/laurajoyhutchins/gliner-rs/actions/runs/40394924950).
Source was compiled from exact commit `ad83553f6be42a6f78911ae5fc6e5fcc10f1411d`.
Weights were verified by SHA-256 `4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b`.
Ubuntu 22.04 GitHub runner, two inference threads, six samples per case after two warmups.
The canonical stage JSON and raw logs are available as the workflow artifact.

On the ten-case general suite, median encoder-time share across cases was
**94.4%** (range 85.1–98.9%). The median initial boundary-head share was
**4.5%**. Approximate median time shares from repeated measured DeBERTa
layers were **56.8% feed-forward** and **37.3% attention**. Each percentage
is a median of per-case fractions, not a decomposition that must sum to 100%.

Per-case boundary candidate-pool selection took approximately **0.07–0.22 ms**
on the general suite, reaching **0.30 ms** in the longest record case;
the encoder took tens of milliseconds. Further work should prioritize the
DeBERTa CPU linear/activation backend, not pooled candidate top-k.

This is *attribution*, not evidence of a production speedup. Candidate
changes still require a controlled A/B benchmark with complete output parity,
the pinned Python reference, and independent browser checks where applicable.
