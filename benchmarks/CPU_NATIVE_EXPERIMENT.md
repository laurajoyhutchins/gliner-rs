# CPU-specific DeBERTa code generation: controlled trial

The independently pinned stage profile in GLiNER2.5-small found that DeBERTa
dominates inference, with feed-forward layers consuming the majority of the
total request time. This experiment asks whether CPU-native Rust code
generation benefits real workloads without changing any tensor math.

Run the workflow `encoder-native-cpu-experiment.yml` on a consistent
x86-64 CI host. It compiles portable and `-C target-cpu=native` binaries
from the **same exact source**, validates one immutable model-weight SHA-256,
then alternates AB/BA whole-model runs for ordinary and record-heavy cases.

Admission boundaries:

1. Full output equivalence within existing `compare_rust_revisions.py`
   numeric tolerance (`1e-4`) is mandatory. That is not byte-for-byte
   floating point equality.
2. Report the geometric-mean speedup and per-case regressions, not a selected
   microbenchmark.
3. Deployable binaries using `target-cpu=native` may require CPU instructions
   absent on other hardware. They must never replace the portable published
   binary or browser WASM package by default.
4. If the performance gain is not robust and material, do not adopt the flag.
   The workflow itself is experimental evidence, not permission to change
   production build configuration.
