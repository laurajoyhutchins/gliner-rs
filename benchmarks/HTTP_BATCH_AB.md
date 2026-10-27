# HTTP batch prompt preparation experiment

At the GLiNER HTTP batch endpoint, all texts share a schema. Prior
`Processor::prepare` builds and tokenizes the *same* schema prefix for
every document even when all other request options match.

The candidate introduces `Processor::prepare_batch`, computing marker
offsets and tokenized prefix IDs once per batch. Text splitting and per-word
tokenization remain independent. No global mutable cache, new inference
kernel or changed output contract is involved. This code is not a blanket
claim of HTTP speedup: the pinned comparison must show one.

Run `http-batch-ab.yml` to compile the original merged HTTP API as the
reference, and the candidate on the exact PR head. The **same test program**,
CPU threads, pinned checkpoint SHA-256 and multi-case corpus are used for
both. All full output SHA-256 hashes must agree, both with serial request
results and across candidate/reference revisions. Performance measurements
are AB/BA repeated with per-case latency distributions.

Do not enable this approach for production if runtime gains do not exceed
shared-runner noise or if any full output difference appears. The existing
real HTTP conformance and upstream Python parity checks remain mandatory.
