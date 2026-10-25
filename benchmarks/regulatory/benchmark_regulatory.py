#!/usr/bin/env python3
"""Warm CPU Rust-vs-PyTorch benchmark on a PRIVATE regulatory corpus.

The report contains hashes, page numbers, timings and numeric scores ONLY.
Neither licensed source excerpts nor predicted entity surface strings are
written to the public report. Compare semantic outputs and numerical values
separately; matching Python is not independent engineering accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from prepare_corpus import digest

LABELS = ["electrical equipment", "voltage rating", "current rating",
          "section reference", "exception", "location"]


def select_chunks(chunks, limit):
    by_page = {}
    for item in chunks:
        by_page.setdefault(item["page"], []).append(item)
    selected = []
    # Interleave all source pages, avoiding first-page bias.
    for fraction in (0.0, 0.5, 1.0):
        for _, items in sorted(by_page.items()):
            candidate = items[round((len(items) - 1) * fraction)]
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= limit:
                return selected
    return selected


def normalize(value, omit_confidence=False):
    if isinstance(value, dict):
        return {k: normalize(v, omit_confidence) for k, v in value.items()
                if not (omit_confidence and k == "confidence")}
    if isinstance(value, list):
        return [normalize(x, omit_confidence) for x in value]
    return value


def numerically_equal(a, b, tolerance=1e-4):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(numerically_equal(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(numerically_equal(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tolerance
    return a == b


def quant(samples):
    ordered = sorted(samples)
    return {"median_ms": round(statistics.median(ordered), 3),
            "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
            "samples_ms": [round(n, 3) for n in samples]}


def http_request(url, body=None):
    request = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as result:
        return json.load(result)


def rust_inference(args, cases):
    log = args.out.with_suffix(".rust-private.log")
    command = [str(args.rust_binary.resolve()), "--model", str(args.model_dir.resolve()),
               "--threads", str(args.threads), "serve", "--host", "127.0.0.1",
               "--port", str(args.port)]
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, RAYON_NUM_THREADS=str(args.threads),
                                            OMP_NUM_THREADS=str(args.threads)))
        try:
            for _ in range(300):
                if process.poll() is not None:
                    raise RuntimeError(f"Rust server exited {process.returncode}: {log}")
                try:
                    http_request(f"http://127.0.0.1:{args.port}/health")
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(0.1)
            else:
                raise TimeoutError("Rust model did not load in 30 seconds")
            results = {}
            for case in cases:
                body = {"text": case["text"], "entities": LABELS, "threshold": 0.5,
                        "spans": True, "confidence": True}
                def one():
                    return http_request(f"http://127.0.0.1:{args.port}/v1/extract", body)["result"]
                for _ in range(args.warmups):
                    one()
                times = []
                for _ in range(args.rounds):
                    start = time.perf_counter_ns()
                    answer = one()
                    times.append((time.perf_counter_ns() - start) / 1e6)
                results[case["id"]] = {"timing": quant(times), "prediction": answer}
            rss = None
            try:
                for line in Path(f"/proc/{process.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        rss = int(line.split()[1]) * 1024
            except OSError:
                pass
            return results, rss
        finally:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def python_inference(args, cases):
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    if args.python_source:
        sys.path.insert(0, str(args.python_source.resolve()))
    from gliner2 import AutoExtractor
    start = time.perf_counter()
    model = AutoExtractor.from_pretrained(str(args.model_dir.resolve()), map_location="cpu")
    model.eval()
    load_seconds = round(time.perf_counter() - start, 3)
    schema = model.create_schema().entities(LABELS)
    results = {}
    for case in cases:
        def one():
            with torch.inference_mode():
                return model.extract(case["text"], schema, threshold=0.5,
                                     include_confidence=True, include_spans=True)
        for _ in range(args.warmups):
            one()
        samples = []
        for _ in range(args.rounds):
            start = time.perf_counter_ns()
            prediction = one()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        results[case["id"]] = {"timing": quant(samples), "prediction": prediction}
    return results, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, load_seconds


def score_gold(cases, outputs, gold):
    by_id = {x["id"]: x for x in cases}
    stats = {"tp": 0, "fp": 0, "fn": 0, "reviewed_cases": 0}
    seen = set()
    for annotation in gold.get("cases", []):
        ident = annotation.get("case_id")
        if ident not in by_id or not annotation.get("complete", False):
            continue
        if ident in seen:
            raise ValueError(f"duplicate reviewed case: {ident}")
        seen.add(ident)
        if by_id[ident]["sha256"] != annotation.get("chunk_sha256"):
            raise ValueError(f"reviewed text changed: {ident}")
        expected = {(x["label"], x["start"], x["end"]) for x in annotation["labels"]}
        entities = outputs[ident]["prediction"].get("entities", {})
        found = {(label, x["start"], x["end"]) for label, values in entities.items()
                 for x in values if isinstance(x, dict) and "start" in x and "end" in x}
        stats["tp"] += len(expected & found)
        stats["fp"] += len(found - expected)
        stats["fn"] += len(expected - found)
        stats["reviewed_cases"] += 1
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    stats["precision"] = tp / (tp + fp) if tp + fp else None
    stats["recall"] = tp / (tp + fn) if tp + fn else None
    prec, recall = stats["precision"], stats["recall"]
    stats["f1"] = 2 * prec * recall / (prec + recall) if prec is not None and recall is not None and prec + recall else None
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--rust-binary", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--python-source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gold-json", type=Path)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--port", type=int, default=18780)
    args = parser.parse_args()
    if min(args.cases, args.threads, args.rounds) < 1 or args.warmups < 0:
        parser.error("cases/threads/rounds must be positive; warmups >= 0")
    model_hash = digest((args.model_dir / "model.safetensors").read_bytes())
    if model_hash.lower() != args.checkpoint_sha256.lower():
        parser.error("checkpoint SHA-256 mismatch")
    corpus = json.loads(args.corpus.read_text())
    if corpus.get("extraction") not in ("docling_document_json", "pymupdf_text_blocks_not_docling"):
        parser.error("unknown extraction provenance")
    cases = select_chunks(corpus["chunks"], args.cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rust, rust_rss = rust_inference(args, cases)
    python, python_rss, load_s = python_inference(args, cases)
    rows = []
    semantic_matches = full_matches = 0
    for case in cases:
        a, b = rust[case["id"]], python[case["id"]]
        semantic = numerically_equal(normalize(a["prediction"], True), normalize(b["prediction"], True))
        full = numerically_equal(a["prediction"], b["prediction"])
        semantic_matches += semantic
        full_matches += full
        rows.append({"case_id": case["id"], "page": case["page"], "chunk_sha256": case["sha256"],
                     "characters": len(case["text"]), "rust": a["timing"], "python": b["timing"],
                     "semantic_equal": semantic, "full_numeric_equal": full})
        print(f'{case["id"]}: Rust {a["timing"]["median_ms"]:.1f}ms, Python {b["timing"]["median_ms"]:.1f}ms, semantic={semantic}', flush=True)
    speedups = [x["python"]["median_ms"] / x["rust"]["median_ms"] for x in rows]
    report = {"source_sha256": corpus["source_sha256"], "extraction": corpus["extraction"],
              "model_sha256": model_hash, "cases": rows, "case_count": len(rows),
              "page_numbers": corpus["pages"], "threads": args.threads, "rounds": args.rounds,
              "warmups": args.warmups, "platform": platform.platform(),
              "rust_peak_rss_bytes": rust_rss, "python_peak_rss_bytes": python_rss,
              "python_load_seconds": load_s, "semantic_equal_count": semantic_matches,
              "full_numeric_equal_count": full_matches,
              "geomean_python_over_rust": round(math.exp(statistics.mean(math.log(n) for n in speedups)), 3),
              "total_rust_median_ms": round(sum(x["rust"]["median_ms"] for x in rows), 3),
              "total_python_median_ms": round(sum(x["python"]["median_ms"] for x in rows), 3)}
    if args.gold_json:
        gold = json.loads(args.gold_json.read_text())
        if gold.get("format_version") != 1 or gold.get("source_sha256") != corpus["source_sha256"]:
            parser.error("reviewed annotations do not match the PDF source")
        report["reviewed_accuracy_rust"] = score_gold(cases, rust, gold)
        report["reviewed_accuracy_python"] = score_gold(cases, python, gold)
    else:
        report["accuracy_status"] = "NOT_ASSESSED_NO_INDEPENDENT_GOLD_ANNOTATIONS"
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {args.out}; {semantic_matches}/{len(rows)} semantically equal")


if __name__ == "__main__":
    main()
