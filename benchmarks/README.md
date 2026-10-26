# GLiNER2.5 reference parity and CPU benchmark

`compare.py` runs the release/profile-selected Rust `gliner` HTTP server and the
original Python `gliner2` extractor against the **same** checkpoint and case
corpus. It refuses weights whose `model.safetensors` SHA-256 differs from
the supplied expected digest. Its JSON report includes complete predictions,
element-level differences, repeated wall-clock timings, and separate-process
peak resident memory.

## Example

Install Python, CPU PyTorch, GLiNER2 and a compatible Transformers version in a
virtual environment. Use an independently fetched, pinned checkpoint and a
locally compiled `gliner` binary. Then:

```sh
python benchmarks/compare.py \
  --rust-bin target/release/gliner \
  --model-dir /path/to/gliner2.5-small-v1 \
  --checkpoint-sha256 4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b \
  --python-source /path/to/pinned/GLiNER2-source \
  --threads 2 --warmups 2 --rounds 8 \
  --out gliner-benchmark.json
```

The public `gliner2.5-small-v1` weights above were measured on October 25,
2026. Pin the Python revision too: the reference used upstream GLiNER2 2.1.0,
PyTorch 2.10.0+cpu and Transformers 5.8.0. The Rust baseline was
`apiplant/gliner-rs` 0.3.1 at `c5c65948478d687190f9c83fcc51d3afa15b5f2a`.
The checkpoint SHA is an explicit input so additional audited models can run.

**Measurement boundary:** Rust latency includes localhost HTTP framing.
Python latency excludes transport. Model loading is excluded from warm
measurements. They are not intrinsic kernel timings. Run each benchmark
multiple times on an otherwise idle machine and compare distributions rather
than treating minor per-case differences as meaningful.

Agreement with Python proves implementation parity, not factual correctness
of model predictions. The ten-case corpus is intentionally small. Extend it
with labeled multilingual, regulatory, records, attributes, long-document and
batch inputs before making broad compatibility claims. `--require-full-parity` now passes 10/10 on the pinned source and checksum-verified small checkpoint after [PR #18](https://github.com/laurajoyhutchins/gliner-rs/pull/18), independently verified by [CI run 40390839329](https://github.com/laurajoyhutchins/gliner-rs/actions/runs/40390839329). This is limited to the ten reviewed cases; it is not a claim about all models or inputs.

## Focused candidate-pool microbenchmark

The Rust source includes an ignored unit benchmark comparing the original
candidate-deduplication implementation against the single-sort fast path,
using identical 1,024-candidate inputs. Run on an otherwise idle CPU:

```sh
cargo test --release --lib benchmark_pool_selection_reference_vs_single_sort \
  -- --ignored --nocapture
```

The microbenchmark prints microseconds per call. It does **not** establish
end-to-end speedup, which must be measured separately through `compare.py`.
A non-ignored property test checks 1,000 generated candidate sets including
invalid candidates, ties and masking sentinels for exactly equal selection.

## End-to-end Rust optimization comparison

`compare_rust_revisions.py` checks that two compiled Rust engines produce
the same complete inference outputs, while measuring warm end-to-end HTTP
latency and peak resident memory. It alternates baseline/candidate process
order to reduce systematic drift from host load. It rejects a mismatching
model-weight SHA before either inference process starts.

```sh
python benchmarks/compare_rust_revisions.py \
  --baseline /path/to/untouched/gliner \
  --candidate target/release/gliner \
  --model-dir /path/to/gliner2.5-small-v1 \
  --checkpoint-sha256 4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b \
  --threads 2 --pairs 2 --samples 8 \
  --out rust-revision-benchmark.json
```

**Measured in the ChatGPT Debian 13 VM, October 25, 2026:** comparing the
fork's pre-fusion head `521d0f63` (with candidate-pool optimization) against
the same source plus fused DeBERTa query/key/value projections, on 5 vCPU,
2 Rayon threads, two AB/BA measurement pairs, 8 warm samples per case:
**1.025x geometric-mean speedup** across ten scenarios (about 2.5% lower
latency in aggregate), with **10/10 exact output matches** (tolerance 1e-4)
and approximately 713 MiB peak RSS per process. A separate single AB pair
measured 1.064x. The performance gain is modest and noisy, and some
individual workloads regress. These figures are local evidence, not an
independently reproduced CI performance claim or a universal speedup.

A related Python parity run on the fused model retained **10/10 semantic matches** and **7/10 full confidence-metadata matches** versus GLiNER2 2.1.0 at that historical pre-fix revision. The three differences were subsequently corrected in [PR #18](https://github.com/laurajoyhutchins/gliner-rs/pull/18).

The fused DeBERTa QKV implementation preserves weight tensors and computes
three attention projections through one matrix multiplication per layer.
It adds a unit regression covering independently calculated projections for
both biased and bias-free batched inputs. It does not affect ModernBERT or
the legacy span model. There is no expectation of a GPU speedup without GPU
benchmark evidence.

Keep `cargo test --release --locked --lib` and the Python parity harness
as correctness checks. Do not use a timing threshold as a mandatory CI gate
on shared runners.

## Length-bucketed inference (no retraining)

The boundary architecture offers an **opt-in** `GLiNER2::extract_batch_bucketed`
method for document batches whose encoded token lengths vary substantially.
It sorts inputs by prepared token count, groups at most 8 similarly sized
documents, requires no more than 25% length spread, and caps
`batch_size * max_tokens^2` at 1,000,000 for multi-input groups. Extraction
results return in the caller's original order. Other model architectures
retain their existing batch path.

This avoids padding every document to the longest sequence, which wastes
bidirectional attention computation and intermediate memory. The weights,
tokenizer, decoder and public output format do not change. This is not
an autoregressive KV-cache change or a retrained sparse-attention model.

**Reproduce against a pinned checkpoint** (with the same SHA-256 validation
used by `compare.py` before executing model inference):

```sh
cargo test --release --locked --lib
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  cargo run --release --locked --example bench_bucketed -- /path/to/gliner2.5-small-v1
```

On October 25, 2026, the ChatGPT Debian 13 CPU VM (5 vCPUs,
2 inference threads, GLiNER2.5-small FP32 weights with SHA-256
`4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b`)
measured a 16-document mixed-length electrical-text example over five
alternating A/B passes:

| Run | Existing batch | Length-bucketed | Speedup |
|---|---:|---:|---:|
| Initial | 3029.9 ms | 1221.5 ms | 2.48× |
| Clean fused-encoder reproduction | 3089.9 ms | 1172.2 ms | 2.64× |

All five output comparisons in each run matched **exactly** including
confidence fields and offsets. These are end-to-end batch-throughput timings
excluding checkpoint loading, not single-document latencies. Smaller,
uniform-length batches may benefit little or not at all; hardware, model,
and batch composition can materially change the outcome.

Other no-retraining experiments were explicitly rejected: the bounded
relative-position cache plus unpadded fast path measured **0.96×** the
reference speed (4% regression) on ten short cases; a flattened CPU linear
projection rewrite produced noisy, inconsistent results. Neither is part of
this optimization. Specialized GEMM kernels or quantization require
separate per-hardware benchmarking and conformance checks.

## Record enum-score reuse (post-parity candidate)

The shared-pool explicit-span scorer evaluates schema-prefix enum choices
against the current encoded document/query. That score is the same for each
record instance in the **same** extraction. The record-specific choice
ownership, source spans, and formatting must still run separately.

`src/model.rs` uses a request-local cache keyed by query ID and exact
explicit-token spans. Cache misses execute the original scorer; errors and
nonfinite probabilities cannot enter the cache. Historical `master` after
PR #18 is the compatible output baseline.

`rust-revision-record-perf.yml` runs an AB/BA whole-model comparison with
identical SHA-256-verified weights and an additional record-rich input corpus.
It verifies **full JSON** equality before presenting timing summaries.
Results must be evaluated as noisy host-specific distributions, not
intrinsic model performance guarantees. The improvement is expected to be
concentrated in record/enum workloads, not general NER.
