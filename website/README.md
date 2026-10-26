# gliner-rs website

The marketing and documentation site for [gliner-rs](https://github.com/apiplant/gliner-rs).
Solid 2 RC + Tailwind v4 + Vite, static build, deployed to Cloudflare Pages.

```bash
pnpm install
pnpm dev       # http://127.0.0.1:5275
pnpm build     # → dist/
pnpm check     # types only
pnpm build:wasm        # rebuild committed browser package (pinned Rust + wasm-pack)
pnpm build:wasm:check  # fresh isolated build, byte-for-byte drift verification
```

## The wasm package is committed

The `/demo` pages run gliner-rs itself in the browser, so the site depends on
`src/wasm-pkg/` — the `wasm-pack` output for the crate one directory up. That
directory is **committed** rather than generated during the build: the deploy
environment has Node and nothing else, and asking it for a Rust toolchain would
mean installing rustup and compiling candle on every deploy.

So `pnpm build` never runs `wasm-pack`; it only checks the package is there.
A Rust change can no longer pass browser CI with a stale binary. The browser
regression job always rebuilds the crate using `../rust-toolchain.toml` and
`wasm-pack-version`, with `Cargo.lock` enforced, then compares all five
committed browser artifacts byte-for-byte against a fresh isolated build.
`--no-opt` disables the floating wasm-opt/Binaryen tool; Rust release
optimization still applies. Run `pnpm build:wasm` and commit the resulting
`src/wasm-pkg/` files whenever the generated output changes. The dedicated
`automation/wasm-artifact-sync` branch also has a narrowly scoped trusted
push workflow that commits rebuilt artifacts. Pull-request verification is
read-only and never updates baselines automatically.

This gate proves source-to-artifact consistency, **not** real browser execution
or checkpoint inference. Those require independent browser runtime tests.

## The version is not copied here

The install section names the version, and the binaries stamp the same string — so
`vite.config.ts` reads it from `../Cargo.toml` (`[package] version`) at build time and
injects it as `__VERSION__`. There is no copy in the site to keep in sync.

## Documentation pages

`src/components/docs/` holds one page per topic (`Overview`, `Library`, `Cli`, `Classify`,
`Pii`, `Guardrails`), a shared sidebar (`DocsLayout.tsx`), and shared prose primitives
(`Prose.tsx` — headings, paragraphs, code blocks, flag tables). There is no MDX pipeline:
each page is hand-written TSX kept in sync with `../README.md` by hand.

## Deploying

A static SPA: `dist/` is assets-only, and `wrangler.jsonc` sends unknown paths to
`index.html` (`not_found_handling: single-page-application`) so deep links resolve on a
cold load.

```bash
npx wrangler pages deploy dist --project-name gliner-rs-website
```

`index.html`, `public/robots.txt` and `public/sitemap.xml` name the default
`gliner-rs.apiplant.com` domain; update all three if the site moves.

## Verified browser checkpoint downloads

All Hugging Face models offered in the browser use `src/lib/checkpoint-locks.json`.
Each entry conforms to `../contracts/checkpoint-lock.schema.json` and
specifies the exact Hugging Face repository, immutable Git revision, and
SHA-256 hashes of all four required files. The initial locks were obtained
from pinned Hugging Face revisions: the published LFS SHA-256 hashes were used
for large files, and smaller files were downloaded at the exact commit,
checked against their published Git blob identities, then hashed with SHA-256.
These are source-bound observations, **not an independent attestation of model
trustworthiness**. Updating a model requires reviewing its new pinned lock.

The browser checks SHA-256 while downloading and **before** writing any
checkpoint to its versioned Cache API store. A corrupt cached file is evicted
and fetched again; an interrupted, incomplete, or mismatching network response
is never cached. The network fetch never resolves a mutable `main` ref.
Incremental hashing does not require a second full-file buffer. When the
server supplies Content-Length, bytes are written directly to one
preallocated Uint8Array; otherwise the downloader collects chunks and creates
one final contiguous buffer. WebAssembly itself may still copy these bytes.

Model replacement is transactional: if a download or WebAssembly
initialization fails, the prior model remains active. This can temporarily
increase memory use while two models coexist, especially for the large browser
checkpoints. Clearing cached files does **not** unload an already-running
model. Explicitly chosen local directories remain user-managed and are not
automatically claimed to match the published lock (including historical
unversioned local layouts).

```bash
pnpm install --frozen-lockfile
pnpm test        # transactional lifecycle and corrupted/short/aborted downloads
pnpm check       # TypeScript
pnpm build       # deployable website
```


## Model memory preflight

The browser cannot assume a second loaded model will fit alongside the first.
Before any replacement download, `src/lib/modelMemory.ts` applies a
conservative check to checkpoint sizes, existing WebAssembly linear memory,
and (when available) the browser's rounded `navigator.deviceMemory`.
It deliberately keeps headroom below wasm32's theoretical 4-GiB ceiling.
If the estimate exceeds this budget, selection fails **before downloading**
and preserves the active model. Use the new **unload active model** control
and retry, or refresh the tab if WebAssembly linear memory is already large.
Unloading is distinct from deleting cached checkpoint files and prevents
the automatic-resume effect from reloading the deliberately unloaded model.

These multipliers are conservative heuristics, not observed peak-memory
measurements or proof that every accepted load will succeed on all browsers.
