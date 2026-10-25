#!/usr/bin/env python3
"""Optional CPU configuration sweep of PyTorch (MKL + oneDNN).

Not a Candle+MKL implementation. Disabling oneDNN still leaves ATen's MKL
GEMM enabled. No licensed text or extracted entities are written to disk.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from benchmark_regulatory import LABELS, normalize, numerically_equal, select_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--python-source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.cases < 1 or args.repeats < 1:
        parser.error("cases and repeats must be positive")
    expected = hashlib.sha256((args.checkpoint / "model.safetensors").read_bytes()).hexdigest()
    if expected != args.checkpoint_sha256.lower():
        parser.error("checkpoint checksum mismatch")
    if args.python_source:
        sys.path.insert(0, str(args.python_source.resolve()))
    import torch
    from gliner2 import AutoExtractor
    torch.set_num_interop_threads(2)
    model = AutoExtractor.from_pretrained(str(args.checkpoint.resolve()), map_location="cpu")
    model.eval()
    schema = model.create_schema().entities(LABELS)
    corpus = json.loads(args.corpus.read_text())
    cases = select_chunks(corpus["chunks"], args.cases)
    configurations = [(1, True), (2, True), (4, True), (2, False)]
    timings = {f"{threads}t_oneDNN_{int(flag)}": [] for threads, flag in configurations}
    reference = None
    differences = {}
    for trial in range(args.repeats):
        order = configurations if trial % 2 == 0 else list(reversed(configurations))
        for threads, one_dnn in order:
            torch.set_num_threads(threads)
            name = f"{threads}t_oneDNN_{int(one_dnn)}"
            with torch.backends.mkldnn.flags(enabled=one_dnn):
                with torch.inference_mode():
                    for case in cases:
                        model.extract(case["text"], schema, threshold=.5,
                                      include_confidence=True, include_spans=True)
                started = time.perf_counter_ns()
                predictions = []
                with torch.inference_mode():
                    for case in cases:
                        predictions.append(model.extract(
                            case["text"], schema, threshold=.5,
                            include_confidence=True, include_spans=True))
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                timings[name].append(elapsed_ms)
                if reference is None:
                    reference = predictions
                semantic = sum(not numerically_equal(normalize(a, True), normalize(b, True))
                               for a, b in zip(reference, predictions))
                numeric = sum(not numerically_equal(a, b) for a, b in zip(reference, predictions))
                differences.setdefault(name, {"semantic": 0, "numeric": 0})
                differences[name]["semantic"] += semantic
                differences[name]["numeric"] += numeric
                print(f"trial={trial} {name} {elapsed_ms:.1f} ms semantic_changes={semantic}", flush=True)
                if semantic:
                    raise RuntimeError(f"{name} changes entity or span outputs on {semantic} cases")
    report = {
        "source_sha256": corpus["source_sha256"],
        "checkpoint_sha256": expected,
        "case_count": len(cases),
        "backend": "PyTorch CPU built with MKL (oneDNN switch does not disable MKL GEMM)",
        "results": {name: {"median_ms": round(statistics.median(values), 3),
                           "samples_ms": [round(x, 3) for x in values],
                           "differences": differences[name]}
                    for name, values in timings.items()}
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
