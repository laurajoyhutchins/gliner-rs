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

**Decision:** retain the Rust inference engine and current memory footprint;
this benchmark does not justify a second inference runtime. Independently
reviewed labeled excerpts remain necessary before making accuracy claims.

## Direct Docling conversion and GLiNER benchmark

The new **run_docling.py** runner executes a pinned docling.rs CLI against the
original private PDF, one selected physical page per conversion. It verifies
that DoclingDocument JSON retained the original physical `page_no`, validates
the cross-repository JSON Schema, converts both text blocks and indexed table
rows into GLiNER inputs, validates the resulting corpus schema, and invokes
the existing Rust-versus-Python benchmark. It emits a separate source-safe
receipt. **No private pages, raw predictions or tables are published.**

Run from the GLiNER repository root, where the isolated Docling executable
and the model already exist:

```sh
python3 -m pip install 'jsonschema==4.26.0'
python3 benchmarks/regulatory/run_docling.py \
  --pdf /secure/nec-2017.pdf \
  --docling-bin /secure/docling-rs \
  --docling-revision e91d7fe5e756aabae322bba10d48f4d8cfce8ce0 \
  --mode text-layer \
  --pages 40,80,100,230,350,450,550,650 \
  --private-dir benchmarks/regulatory/private-codebook/nec2017 \
  --rust-binary target/release/gliner \
  --model-dir /secure/gliner2.5-small-v1 \
  --python-source /secure/GLiNER2 \
  --checkpoint-sha256 4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b \
  --rounds 4 --threads 2
```

**Do not use the user's fork commit argument with an upstream release binary.**
Pass its true release tag or actual source revision, and verify the binary
SHA in the safe receipt. Source provenance must be truthful.

`--mode standard` (the default) invokes Docling's **real structured PDF
layout and table pipeline**, with no silent text-only fallback. Install the
matching Docling ONNX layout, TableFormer and OCR model files through the
Docling fork's `scripts/install/download_dependencies.sh`. For a digital
text-bearing NEC PDF, `--mode text-layer` is substantially lighter, but
it **does not recognize tables as structured grid cells**. Any tables emitted
by the standard path are handled as provenance-bearing grid rows; malformed
tables cause the entire conversion to fail. Captions, bounding boxes and
schema versions remain available in the private per-page Docling JSON.

The JSON boundary is specified by both:
`contracts/docling-export.schema.json` (JSON Schema 2020-12) and
`contracts/docling-export.odcs.yaml` (ODCS 3.2).
The locally generated corpus is validated against
`contracts/docling-corpus.schema.json`. This preserves the repository's
contract rule without making GLiNER dependent on Docling's Rust runtime.

**Verified on the private 2017 NEC PDF (October 25, 2026)** using the
upstream docling.rs v2.51.3 release CLI (historical reference run):
8 original physical pages, 84 text-layer chunks, source SHA
`603ef5c461247bacd716e3953222bfb227f1ddc780fffdbfcb90756b02c237c7`.
On 16 sampled chunks, Rust/Python extraction agreed semantically and
numerically in **16/16** cases (1e-4 tolerance). Rust summed warm per-chunk
median latency **2403ms** versus Python/PyTorch **1772ms** (two timing samples
per chunk). This is implementation parity, not independently validated NEC
accuracy. Source passages stay only in the local private directory.

The older `prepare_corpus.py --docling-json` path now also requires
`--source-pdf` so that its `source_sha256` identifies the original code
book, not a transient JSON conversion artifact.

### Owner-built Docling confirmation

The owner's fork, `laurajoyhutchins/docling.rs`, was compiled on
Ubuntu 24.04 from exact commit
`e91d7fe5e756aabae322bba10d48f4d8cfce8ce0`, yielding
Docling CLI `2.51.0`, binary SHA-256
`633ff50da0a3de30faa6e43521e0292cc224a15ec0ec4fbc5cdbc807820b6b5e`.
Ubuntu 22.04 was incompatible with bundled ONNX Runtime C++ symbols.

The owner's binary reproduced **all 84 identical text-chunk SHA-256 values**
on the same eight 2017 NEC physical pages compared with the upstream
Docling v2.51.3 release. Running those chunks through both inference
engines yielded **16/16 semantic and 16/16 numerical matches** within
`1e-4`; the sum of per-case median timings from two samples was
Rust/Candle **2415 ms** and Python/PyTorch **2056 ms**. This is
end-to-end **text-layer** conversion, not ML layout or table accuracy
verification.

For future isolated builds, use the on-demand
`.github/workflows/docling-cli-transport.yml` workflow with an exact
40-character Docling fork commit SHA. It archives the compiled CLI with
both binary and archive checksums, a source-revision manifest, and
one-day artifact retention. It does not ship PDFs or extracted text.
The usual pull-request CI never rebuilds Docling.

### Verified full-layout, TableFormer and OCR pipeline (October 25, 2026)

The Docling executable compiled from **the owner's fork at**
`e91d7fe5e756aabae322bba10d48f4d8cfce8ce0`
(binary SHA-256 `633ff50da0a3de30faa6e43521e0292cc224a15ec0ec4fbc5cdbc807820b6b5e`)
executed **standard ML layout and TableFormer** with SHA-verified published
model assets from Docling's `models-v1` release:

| Source physical PDF page | Structured tables | Grid dimensions | Extracted anchored cells | Wall time | Peak RSS |
| --- | ---: | --- | ---: | ---: | ---: |
| NEC 2017 page 80 | 2 | 4×3, 5×3 | 26 | 3.24 s | 1.05 GiB |
| NEC 2017 page 650 | 1 | 26×9 | 206 | 9.89 s | 1.75 GiB |

The same owner executable extracted **33 page-provenanced text elements**
from an image-only scan created locally from original PDF page 40
(no embedded PDF text), in **6.84 seconds**, at **1.71 GiB peak RSS**.
An unordered token-multiset check against the original digital page found
approximately **98.5% precision** and **96.0% recall**. It does not test
reading order, correct table interpretation or engineering compliance.

The private Docling JSON preserves cell coordinates, merged-cell spans,
column/row header roles, original page references and table captions.
The private `docling-tables.json` is independently bound to the original
PDF hash and to its own content hash recorded in the corpus receipt.
The GLiNER textual projection uses explicit table and cell references;
**it is not the authoritative representation of the table grid**.
Furnishings such as page numbers and running headers are excluded from
regulatory GLiNER input without deleting their source provenance.

Run with the full models installed and match the exact owner binary:

```sh
python3 benchmarks/regulatory/run_docling.py \
  --pdf /secure/nec-2017.pdf \
  --docling-bin /secure/docling-rs \
  --docling-revision e91d7fe5e756aabae322bba10d48f4d8cfce8ce0 \
  --expected-docling-sha256 633ff50da0a3de30faa6e43521e0292cc224a15ec0ec4fbc5cdbc807820b6b5e \
  --mode standard --pages 80,650 \
  --private-dir benchmarks/regulatory/private-codebook/nec2017-layout
```

`--mode text-layer` remains a different and faster benchmark profile.
For an offline machine, the on-demand
`.github/workflows/docling-models-transport.yml` workflow packages two
short-lived artifacts containing the exact model assets pinned in
`benchmarks/regulatory/docling-models.sha256`. Verify the included archive
SHA-256, then extract both archives so that `.models/` is beside the
Docling CLI or in its recognized model search path. This downloads only
public model weights. Never upload a private PDF to GitHub Actions.

**Remaining assurance boundary:** model-backed table extraction is now
executed and structurally checked, not independently transcribed and
scored cell-by-cell. OCR token matching on one rasterized page is not OCR
quality assurance across all layouts or editions. GLiNER regulatory
accuracy still needs reviewed gold annotations. Do not claim that matching
two implementations demonstrates NEC rule interpretation accuracy.

### Exact-commit one-command end-to-end verification

Using owner's Docling binary SHA-256
`633ff50da0a3de30faa6e43521e0292cc224a15ec0ec4fbc5cdbc807820b6b5e`,
the GLiNER Rust executable compiled from verification commit
`9a09fc121cb647043f469d7dd5de16a8e2b0e885` (binary SHA-256
`b6a47cae4143c4c0dfbb5c38a118b2b0262ae9c94ad65b2a3f5c855adee46f39`),
and checkpoint SHA-256
`4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b`:

- Real private NEC 2017 PDF, physical pages 80 and 650, produced **3 tables
  with 232 anchored cells** and 20 text chunks. The private canonical
  table-evidence SHA-256 is
  `6162429deff529e6813282fd35b4402c2d83a1992e10980e3bf901a751790115`.
- Single `run_docling.py` invocation completed Docling conversion,
  provenance validation, table serialization, and Python/Rust inference
  comparison. **16/16 semantic and 16/16 full numerical matches** at 1e-4
  tolerance, with two threads. A one-sample-per-case run summed **2891 ms
  Rust** and **2061 ms Python**. A separate three-sample comparison also
  matched 16/16; the timing samples are exploratory.
- A generated *image-only* PDF from original NEC page 40 exercised real
  Docling OCR and the same adapter. It produced 33 recognized text items
  and 12 private chunks, without an embedded text layer. Against the
  digital source, unordered token precision was about **98.5%** and
  token recall **96.0%**. That is not an OCR layout or standards-accuracy
  measure.
- The three model-backed tables have no overlapping occupied grid cells.
  The 26×9 table on page 650 has two empty top-left header positions,
  retained as such, not invented.
- **19 local contract tests passed.** Exact-revision source and checksum
  evidence is local. The public repository contains no private source text,
  OCR transcriptions, or generated table grids.

This closes the executable structured-layout and OCR integration proof.
Engineering rule interpretation, table-cell transcription accuracy at
scale, and code-edition correctness remain separate independent-review
work.
