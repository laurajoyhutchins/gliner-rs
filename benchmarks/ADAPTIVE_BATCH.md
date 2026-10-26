# Adaptive default for heterogeneous document batching

The boundary checkpoint's original `extract_batch` API always pads
the entire input to its longest token sequence. The existing optional
`extract_batch_bucketed` method demonstrated >2x lower execution
time for a mixed-length electrical document batch on a pinned checkpoint.
The adaptive default reuses that implementation when its estimated
quadratic padded-attention work is at least one-third smaller, and for
16+ document batches whose bounded 8-row execution was independently
measured to avoid a slower 16-row default. Smaller uniform batches retain
the original single padded pass.

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

## Evidence and refinement

First real-checkpoint A/B: [run 40406095002](https://github.com/laurajoyhutchins/gliner-rs/actions/runs/40406095002) showed **2.932×** faster mixed-length default with exact SHA-256-equivalent full JSON, but the 16-row near-uniform candidate default ran ~14% slower than old default. [Independent repeat 40406203181](https://github.com/laurajoyhutchins/gliner-rs/actions/runs/40406203181) showed **2.850×** mixed gain and reproduced that uniform slowdown. In both cases, the candidate's already-existing explicit 8-row bucket route matched or closely approached the old uniform throughput (approximately 537–541 ms vs 528–535 ms), and output hashes were unchanged. We therefore choose bounded groups for 16+ inputs even when estimated quadratic padding savings are small. This is a narrow observed CPU throughput policy, not a general claim that every 16-row batch is faster. Reverify the revised default at exact head, preserving full output identity.
