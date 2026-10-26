#!/usr/bin/env python3
"""Profile actual Rust inference stages against a SHA-pinned checkpoint.

The Rust binary's GLINER_PROFILE_STAGES=1 output is parsed as completed
per-extraction frames. All requests run through the real HTTP inference path;
warmups are excluded, and stage durations are CPU wall-clock observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request

EVENT = re.compile(r"^GLINER_STAGE\t([a-z][a-z0-9_]*)\t([0-9]+)$")


def parse_frames(log_text: str) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    frame: dict[str, int] = {}
    for line in log_text.splitlines():
        match = EVENT.fullmatch(line.strip())
        if not match:
            continue
        label, micros = match.group(1), int(match.group(2))
        frame[label] = frame.get(label, 0) + micros
        if label == "extract_total":
            if not all(k in frame for k in ("prepare", "encoder", "post_encoder", "extract_total")):
                raise ValueError(f"incomplete stage frame: {sorted(frame)}")
            if frame["extract_total"] == 0:
                raise ValueError("zero-duration extraction frame")
            if frame["extract_total"] + 100 < frame["prepare"] + frame["encoder"] + frame["post_encoder"]:
                raise ValueError("nested stage measurements exceed full extraction")
            frames.append(frame)
            frame = {}
    if frame:
        raise ValueError(f"unterminated stage frame: {sorted(frame)}")
    return frames


def quant(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "median_us": statistics.median(ordered),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "samples_us": values,
    }


def request(port: int, body: dict | None = None) -> dict:
    address = f"http://127.0.0.1:{port}/" + ("health" if body is None else "v1/extract")
    req = urllib.request.Request(
        address,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as result:
        return json.load(result)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rust-bin", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    ap.add_argument("--out", type=Path, default=Path("inference-stage-profile.json"))
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--port", type=int, default=18931)
    args = ap.parse_args()
    if args.samples < 2 or args.warmups < 0 or args.threads < 1:
        ap.error("samples >= 2, warmups >= 0, threads >= 1 required")
    if not args.rust_bin.is_file() or not args.cases.is_file():
        ap.error("missing binary or case corpus")
    weights = args.model_dir / "model.safetensors"
    if not weights.is_file():
        ap.error(f"missing checkpoint {weights}")
    with weights.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != args.checkpoint_sha256.lower():
        ap.error(f"pinned checkpoint hash mismatch: got {digest}")
    cases = json.loads(args.cases.read_text())
    log = args.out.with_suffix(".stages.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    run_order = []
    with log.open("w") as capture:
        env = dict(os.environ,
                   GLINER_PROFILE_STAGES="1",
                   RAYON_NUM_THREADS=str(args.threads),
                   OMP_NUM_THREADS=str(args.threads))
        process = subprocess.Popen([
            str(args.rust_bin.resolve()), "--model", str(args.model_dir.resolve()),
            "--threads", str(args.threads),
            "serve", "--host", "127.0.0.1", "--port", str(args.port)
        ], stdout=capture, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError(f"Rust process exited: {log.read_text()[-2500:]}")
                try:
                    request(args.port)
                    break
                except (OSError, urllib.error.HTTPError):
                    time.sleep(0.1)
            else:
                raise TimeoutError("model failed to serve its health endpoint")
            for case in cases:
                body = {key: value for key, value in case.items()
                        if key in ("text", "entities", "relations", "json", "classify")}
                body.update(threshold=0.5, spans=True, confidence=True)
                for index in range(args.warmups + args.samples):
                    output = request(args.port, body)
                    if "result" not in output:
                        raise RuntimeError(f"missing extraction output in {case['name']}")
                    run_order.append((case["name"], index >= args.warmups))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    frames = parse_frames(log.read_text())
    if len(frames) != len(run_order):
        raise RuntimeError(f"missing instrumented responses: {len(frames)} frames for {len(run_order)} requests")
    results = {}
    for (name, retained), events in zip(run_order, frames):
        if retained:
            results.setdefault(name, []).append(events)

    report = {
        "checkpoint_sha256": digest,
        "source_binary": str(args.rust_bin),
        "threads": args.threads,
        "warmups": args.warmups,
        "samples": args.samples,
        "scope": "CPU wall-clock, separate opt-in stage events, excludes startup and model load",
        "cases": {},
    }
    for name, samples in results.items():
        names = sorted(set().union(*(sample.keys() for sample in samples)))
        readings = {
            stage: quant([frame.get(stage, 0) for frame in samples])
            for stage in names
        }
        total = readings["extract_total"]["median_us"]
        encoder = readings["encoder"]["median_us"]
        boundary = readings.get("boundary_total", {"median_us": 0})["median_us"]
        report["cases"][name] = {
            "stages": readings,
            "median_encoder_share": round(encoder / total, 4),
            "median_boundary_share": round(boundary / total, 4),
        }
        print(f"{name:20s} total={total / 1000:.2f}ms "
              f"encoder={encoder / 1000:.2f}ms ({encoder / total:.0%}) "
              f"boundary={boundary / 1000:.2f}ms ({boundary / total:.0%})",
              flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
