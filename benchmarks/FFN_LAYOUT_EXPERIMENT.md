# DeBERTa FFN weight layout experiment

Candle 0.11 already routes dense linear layers through its optimized
matrix-multiplication backend. This trial does **not** replace GEMM or GELU.

The opt-in `prepacked-ffn` feature materializes the transpose of DeBERTa's
intermediate and output feed-forward weights at checkpoint load time and
stores each as a transposed *view* of contiguous GEMM-facing data.
The default configuration and weights remain unchanged.

Goal: test whether paying a one-time layout cost saves meaningful recurring
CPU runtime. The storage reorganization may consume transient load-time
memory, so never enable it by default without measured benefit.

Validation: real checksum-pinned model, the exact same Rust source with feature
on vs off, identical full JSON inference outputs (score differences must be
at most 1e-4), AB/BA two-order comparison with repeated warm requests,
and stage profiler readback. Reports include RSS and CPU provenance.

A negligible gain, confidence mismatch, or material memory penalty rejects
the experiment. No performance-only tests are permitted to weaken independent
Python parity or reproducible browser artifact checks.
