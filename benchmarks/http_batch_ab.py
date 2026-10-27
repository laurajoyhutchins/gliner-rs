#!/usr/bin/env python3
"""Source-bound AB/BA comparison of HTTP serial and adaptive-batch endpoints.

Every run compares all *full* batch outputs with independent serial calls.
Reference and candidate are built from pinned adjacent Git revisions with
identical Rust flags, corpus, machine, source-pinned model and CPU threads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.error
import urllib.request

SENTENCES = [
    "Panel LP-1 operates at 480 volts.",
    "Alice works for Acme in Paris.",
    "The distribution switchboard MSB-1 operates at 480/277 V and serves the electrical panels in Building A.",
    "Schneider Electric supplies the 1200 ampere switchgear in Building C. It has a 2000 kVA transformer with a 13.8 kV primary and 480/277 V secondary.",
    "Charlie visited Denver.",
    "Bob called Emily in Chicago.",
    "Industrial motor M-2 is rated 100 horsepower, 460 volts, with a nameplate full-load current of 124 amperes.",
    "Generator GEN-1 supplies emergency switchboard ESB via automatic transfer switch ATS-1.",
]
CASES = [
    ("mixed16", [(SENTENCES[(i * 7 + i // 5) % len(SENTENCES)] + " ") * (i % 3 + 1) for i in range(16)],
      ["company", "person", "location", "equipment", "electrical rating"]),
    ("uniform8", [f"Alice works for Acme in Denver office number {i}." for i in range(8)],
      ["company", "person", "location", "equipment", "electrical rating"]),
    ("dense_schema12", [f"Acme panel LP-{i} in Denver is rated 480 volts." for i in range(12)],
      ["company", "person", "location", "equipment", "electrical rating", "panel",
       "generator", "motor", "transformer", "voltage", "manufacturer", "circuit breaker"]),
]


def call(port: int, path: str, body: dict | None = None) -> dict:
    raw = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/{path}", data=raw,
        headers={"Content-Type": "application/json"} if raw is not None else {})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)


def sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def run(exe: Path, args, port: int) -> dict:
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads))
    log = args.out.with_suffix(f".{exe.parent.name}.{port}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        proc = subprocess.Popen([str(exe.resolve()), "--model", str(args.model_dir.resolve()),
            "--threads", str(args.threads), "serve", "--host", "127.0.0.1", "--port", str(port)],
            env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError(f"{exe} exited: {log.read_text()[-2000:]}")
                try:
                    call(port, "health")
                    break
                except (OSError, urllib.error.HTTPError):
                    time.sleep(0.1)
            else:
                raise TimeoutError(f"{exe} model startup timed out")
            result = {}
            for name, texts, entities in CASES:
                fields = {"entities": entities, "spans": True, "confidence": True, "threshold": 0.5}
                def serial():
                    start = time.perf_counter_ns()
                    values = [call(port, "v1/extract", dict(fields, text=text))["result"] for text in texts]
                    return (time.perf_counter_ns() - start) / 1e6, values
                def batch():
                    start = time.perf_counter_ns()
                    values = call(port, "v1/extract/batch", dict(fields, texts=texts))["results"]
                    return (time.perf_counter_ns() - start) / 1e6, values
                for _ in range(args.warmups):
                    _, a = serial()
                    _, b = batch()
                    if a != b:
                        raise AssertionError(f"{name}: warmup batch vs serial mismatch")
                samples_serial, samples_batch = [], []
                reference = None
                for sample in range(args.samples):
                    if sample % 2 == 0:
                        (t_a, a), (t_b, b) = serial(), batch()
                    else:
                        (t_b, b), (t_a, a) = batch(), serial()
                    if a != b:
                        raise AssertionError(f"{name}: batch vs serial mismatch, sample {sample}")
                    if reference is not None and a != reference:
                        raise AssertionError(f"{name}: nondeterministic full inference outputs")
                    reference = a
                    samples_serial.append(round(t_a, 3))
                    samples_batch.append(round(t_b, 3))
                result[name] = {"full_output_sha256": sha(reference),
                    "documents": len(texts), "serial_ms": samples_serial,
                    "batch_ms": samples_batch}
            return result
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--out", type=Path, default=Path("http-batch-ab.json"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--pairs", type=int, default=2)
    args = p.parse_args()
    if args.threads < 1 or args.samples < 1 or args.pairs < 1 or args.warmups < 0:
        p.error("threads, samples, pairs >= 1; warmups >= 0")
    with (args.model_dir / "model.safetensors").open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != args.checkpoint_sha256.lower():
        p.error(f"checkpoint digest mismatch: {actual}")
    readings = {"baseline": [], "candidate": []}
    for pair in range(args.pairs):
        sequence = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
        for index, name in enumerate(sequence):
            print(f"AB/BA pair {pair + 1}/{args.pairs}: {name}", flush=True)
            readings[name].append(run(getattr(args, name), args, 18810 + pair * 2 + index))
    report = {"weights_sha256": actual, "baseline": str(args.baseline),
        "candidate": str(args.candidate), "threads": args.threads, "pairs": args.pairs,
        "samples": args.samples, "warmups": args.warmups, "cases": []}
    for name, _, _ in CASES:
        expected = readings["baseline"][0][name]["full_output_sha256"]
        for side in ("baseline", "candidate"):
            for reading in readings[side]:
                if reading[name]["full_output_sha256"] != expected:
                    raise AssertionError(f"{name}: full output changed across revisions/runs")
        base = [v for r in readings["baseline"] for v in r[name]["batch_ms"]]
        candidate = [v for r in readings["candidate"] for v in r[name]["batch_ms"]]
        base_serial = [v for r in readings["baseline"] for v in r[name]["serial_ms"]]
        cand_serial = [v for r in readings["candidate"] for v in r[name]["serial_ms"]]
        ratio = statistics.median(base) / statistics.median(candidate)
        case = {"name": name, "documents": readings["baseline"][0][name]["documents"],
            "baseline_batch_ms": base, "candidate_batch_ms": candidate,
            "baseline_serial_ms": base_serial, "candidate_serial_ms": cand_serial,
            "batch_speedup": ratio, "full_output_sha256": expected}
        report["cases"].append(case)
        print(f"{name}: batch {statistics.median(base):.1f} -> "
              f"{statistics.median(candidate):.1f}ms ({ratio:.3f}x), "
              f"serial {statistics.median(base_serial):.1f} -> "
              f"{statistics.median(cand_serial):.1f}ms; identical full outputs", flush=True)
    report["geomean_batch_speedup"] = math.exp(statistics.mean(math.log(x["batch_speedup"]) for x in report["cases"]))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"geometric-mean batch speedup {report['geomean_batch_speedup']:.3f}x", flush=True)

if __name__ == "__main__":
    main()
