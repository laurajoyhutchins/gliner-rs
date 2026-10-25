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
batch inputs before making broad compatibility claims. `--require-full-parity`
fails on the confidence/metadata differences still known in relationship
and record outputs.

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
