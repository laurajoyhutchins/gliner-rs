# Adaptive default for heterogeneous document batching

The boundary checkpoint's original `extract_batch` API always pads
the entire input to its longest token sequence. The existing optional
`extract_batch_bucketed` method demonstrated >2x lower execution
time for a mixed-length electrical document batch on a pinned checkpoint.
The adaptive default reuses that implementation only when its estimated
quadratic padded-attention work is at least one-third smaller.

This is a deterministic routing rule based solely on prepared input token
lengths. It does not change weights, schema interpretation, field scores,
model math, output ordering or the separate span-model implementation.
The original single batch path remains the default for uniform/similar
lengths and small batches. Explicit bucketing is still supported.

The exact full-output comparison intentionally compiles an identical
`bench_adaptive_policy.rs` harness into both pre-change and changed
revisions, exercises 16 mixed-length and 16 equal-length documents,
alternates direct/forced-bucket order, asserts full JSON equivalence
within each revision, and compares SHA-256 across revisions. Any change
fails closed. Latencies are reported, never a hard CI performance threshold
on shared hosts.

Reproduce with `adaptive-batch-proof.yml` using the immutable checkpoint
revision `df5910e44bc4ffdb0d95399a83b0ca4516349aa5`, verified weights
SHA `4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b`.
The demonstrated benefit is batch throughput, not single-document latency.
