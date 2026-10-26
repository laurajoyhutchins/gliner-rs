# Chromium WebAssembly execution tests

This suite imports the **committed real WebAssembly module** through an
ephemeral loopback HTTP server in Chromium. JavaScript glue and Rust exports
are not stubbed or reimplemented. Source-to-artifact freshness is established
separately by the required browser regression build check.

Run locally with Node 22+, `npm ci`, and `npx playwright install chromium`:

```sh
cd website/browser-tests
npm ci
npx playwright install chromium
npm test                     # real Chromium init and invalid checkpoint rejection
GLINER_REAL_CHECKPOINT=1 npm test  # also verify and run pinned small checkpoint
```

The full CI job **sets GLINER_REAL_CHECKPOINT=1**, so it cannot succeed after
skipping real inference. It reads the existing versioned
`src/lib/checkpoint-locks.json` manifest, fetches only immutable-revision
URLs for four small-checkpoint files, verifies each original byte stream
against its SHA-256, and serves verified files through localhost. No model
weights are added to Git. A browser page loads those bytes via `WasmModel.load`
and calls the Rust `classify` path; outputs must satisfy the task label and
finite probability contracts. Network failure and memory exhaustion are
**hard failures**, not simulated successes.

The test records elapsed inference-path time but makes no memory-safety or
Python-parity claim. Those need separate measurements and references.
