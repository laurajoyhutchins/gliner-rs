# Private regulatory extraction performance benchmark

**Purpose:** Compare the production Rust Candle extractor to the pinned Python
GLiNER2.5 reference on real regulatory-document text. Separate latency,
memory, semantic equivalence, and independently reviewed accuracy. **The
scripts deliberately publish no code-book passages or extracted surface
strings.** The input corpus and review annotations stay in an ignored local
folder. They must not be uploaded to a public GitHub Actions artifact.

## Preparing actual Docling output

Use an installed Docling or docling.rs converter *locally* on a lawfully
held PDF. Export DoclingDocument JSON with text elements and `prov.page_no`.
For example, with a docling.rs CLI that supports this output format:

```sh
docling-rs private-nec.pdf --to json > benchmarks/regulatory/private-codebook/docling.json
python3 benchmarks/regulatory/prepare_corpus.py \
  --docling-json benchmarks/regulatory/private-codebook/docling.json \
  --pages 40,80,100,230,350,450,550,650 \
  --max-chars 550 \
  --out benchmarks/regulatory/private-codebook/corpus.json
```

Docling JSON page numbers must be inspected against the source PDF. The
implementation currently accepts the Docling text items with provenance;
table-cell reconstruction is not yet supported. The private PDF is never
required by the repository-owned CI gate.

Where Docling is unavailable, a distinct **PyMuPDF text-layer fallback**
supports the same corpus contract:

```sh
python3 benchmarks/regulatory/prepare_corpus.py \
  --pdf private-nec.pdf --pages 40,80,100,230,350,450,550,650 \
  --out benchmarks/regulatory/private-codebook/corpus.json
```

**Do not describe fallback output as Docling-derived.** The report records
`extraction=pymupdf_text_blocks_not_docling`, the SHA-256 of the original
source, each physical page number, and SHA-256 per private chunk.

## Benchmarking both inference backends

Install the upstream `gliner2` 2.1.0 Python source and a PyTorch CPU build.
Build the fork's Rust `gliner` executable from its exact revision. Supply
the same pinned model directory and verify its Safetensors hash:

```sh
python3 benchmarks/regulatory/benchmark_regulatory.py \
  --corpus benchmarks/regulatory/private-codebook/corpus.json \
  --rust-binary target/release/gliner \
  --model-dir /secure/gliner2.5-small-v1 \
  --python-source /secure/gliner2-reference \
  --checkpoint-sha256 4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b \
  --cases 16 --threads 2 --rounds 4 \
  --out benchmarks/regulatory/private-codebook/summary.json
```

Warm Rust timing includes loopback HTTP overhead; Python timing is
in-process. Neither includes model load. Peak resident memory is measured
separately. A `semantic_equal` result is comparison to the Python
implementation, *not* verification that a statement satisfies a code clause.

For the existing MKL-enabled PyTorch CPU backend, compare CPU threads and
oneDNN operator dispatch:

```sh
python3 benchmarks/regulatory/benchmark_torch_backend.py \
  --corpus benchmarks/regulatory/private-codebook/corpus.json \
  --checkpoint /secure/gliner2.5-small-v1 \
  --checkpoint-sha256 4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b \
  --python-source /secure/gliner2-reference \
  --cases 8 --repeats 2 \
  --out benchmarks/regulatory/private-codebook/backend-matrix.json
```

The oneDNN-off experiment **does not disable PyTorch's MKL GEMM**. This is
a CPU configuration experiment, not an ONNX Runtime or Candle+MKL result.
Do not deploy or add another runtime without a repeatable gain and an
assurance plan. Benchmarks run on shared compute are sensitive to contention.

## Independently reviewed accuracy

Prepare a *private* `gold.json` after reviewing entity and offset spans
against the same PDF and edition. It must include the source SHA and the exact
chunk hashes. Example shape (illustrative offsets, not code-book data):

```json
{
  "format_version": 1,
  "source_sha256": "FULL_SHA256_OF_PRIVATE_SOURCE",
  "cases": [
    {
      "case_id": "p0040-c0000",
      "chunk_sha256": "SHA256_OF_CHUNK_TEXT",
      "complete": true,
      "labels": [
        {"label": "electrical equipment", "start": 12, "end": 25}
      ]
    }
  ]
}
```

Pass `--gold-json /secure/gold.json` to the primary benchmark to report
exact-offset micro precision, recall, and F1 **only on reviewed cases**.
An annotation must explicitly be marked `complete`, meaning all
occurrences of *all six configured label types* were checked in that chunk.
An incorrect source or chunk hash fails the run. With no reviewed
annotations, accuracy is `NOT_ASSESSED_NO_INDEPENDENT_GOLD_ANNOTATIONS`.
Never substitute agreement between two model implementations for human
review or external engineering confirmation.

## October 25, 2026 exploratory evidence (local VM)

Source: owner's private 2017 NEC PDF (881 physical pages), SHA-256
`603ef5c461247bacd716e3953222bfb227f1ddc780fffdbfcb90756b02c237c7`.
Input extraction **PyMuPDF, not Docling**. Sixteen text chunks sampled evenly
over physical pages 40, 80, 100, 230, 350, 450, 550, and 650; roughly
540–550 Unicode characters per chunk. Two CPU threads, four timed warm
samples per case on an AMD EPYC Linux VM, identical small model checkpoint
SHA-256 `4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b`:

| Measurement | Candle Rust FP32 | Python/PyTorch CPU with MKL |
|---|---:|---:|
| Sum of per-case median latencies | 2,275 ms | 1,754 ms |
| Peak resident process memory | 713 MiB | 1,134 MiB |
| Reference semantic parity | 16 / 16 | reference |
| Full numeric parity (1e-4) | 16 / 16 | reference |

Geometric mean Python/Rust per-case speed ratio: **0.767**, meaning Python
is approximately 1.30x as fast on this chosen CPU workload. The total
above is a sum of individual medians, not a measured 16-document
parallel/batched runtime. This benchmark does not isolate encoder kernels.

A first four-case PyTorch configuration probe measured 633 ms at one
thread, 446 ms at two threads, 419 ms at four threads, and 500 ms at two
threads with oneDNN off. Those are **single-run exploratory numbers**, not
repeatable evidence of an optimal setting; a longer interleaved run was
inconclusive under host load. No accelerated Rust backend or quantization
variant has yet been verified. There are no human-reviewed gold labels.

**Decision:** keep the current Rust engine and its lower memory footprint;
do not introduce an ONNX runtime or another production backend solely based
on these early results. Next gate: use actual Docling JSON and labeled
regulatory excerpts, then profile encoder GEMM before considering backend
migration.
