#!/usr/bin/env python3
"""Assert native HTTP adaptive batch output matches independent single requests.

Uses real pinned weights, complete JSON comparison and captured evidence.
The test intentionally does not enforce a latency threshold on shared CI.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def send(port: int, path: str, body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{path}",
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"{path}: unexpected HTTP {response.status}")
        return json.load(response)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rust-bin", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--out", type=Path, default=Path("http-batch-proof.json"))
    ap.add_argument("--port", type=int, default=18945)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    weights = args.model_dir / "model.safetensors"
    with weights.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != args.checkpoint_sha256.lower():
        raise ValueError(f"checkpoint SHA-256 mismatch {actual}")

    texts = [
        "Panel LP-1 operates at 480 volts.",
        "Alice works for Acme in Paris.",
        "The main distribution switchboard MSB-1 is rated 480/277 volts, three-phase, four wire. It has a 400 ampere main breaker and serves the electrical panels on the first floor of the building.",
        "Schneider Electric supplies the 1200 amp switchgear in Building C. The installation uses a 2000 kVA transformer with primary 13.8 kV and secondary 480/277 V. Protective devices coordinate with the service entrance breaker, feeder breakers, and panelboard breakers under normal utility conditions.",
        "Charlie visited Denver.", "Bob called Emily in Chicago.",
        "Industrial motor M-2 is a 100 horsepower induction motor rated 460 volts with full-load current 124 A. The feeder conductors and overload protection are selected for the motor nameplate, operating duty, and equipment requirements.",
        "The emergency generator GEN-1 is rated 1500 kW, 480/277 volts, and supplies the life-safety switchboard through an automatic transfer switch.",
    ]
    mixed = [(texts[(i * 7 + i // 5) % len(texts)] + " ") * (i % 3 + 1) for i in range(16)]
    uniform = [f"Alice works for Acme in Denver office number {i}." for i in range(8)]
    cases = [("mixed16", mixed), ("uniform8", uniform), ("multilingual", ["Alice works for Acme.", "Charlie went to Denver.", "Schneider Electric is in France."])]
    schema = {"entities": ["company", "person", "location", "equipment", "electrical rating"],
              "threshold": 0.5, "spans": True, "confidence": True}
    report = {"source": str(args.rust_bin), "weights_sha256": actual, "cases": []}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".server.log")
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads))
    with log_path.open("w") as logfile:
        proc = subprocess.Popen([
            str(args.rust_bin.resolve()), "--model", str(args.model_dir.resolve()),
            "--threads", str(args.threads), "serve", "--host", "127.0.0.1",
            "--port", str(args.port),
        ], env=env, stdout=logfile, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited: {log_path.read_text()[-3000:]}")
                try:
                    send(args.port, "health")
                    break
                except (OSError, urllib.error.HTTPError):
                    time.sleep(0.1)
            else:
                raise TimeoutError("server never became ready")
            for name, documents in cases:
                singles = []
                start = time.perf_counter()
                for text in documents:
                    result = send(args.port, "v1/extract", {"text": text, **schema})
                    singles.append(result["result"])
                serial_ms = round((time.perf_counter() - start) * 1000, 2)
                start = time.perf_counter()
                batch = send(args.port, "v1/extract/batch", {"texts": documents, **schema})
                batch_ms = round((time.perf_counter() - start) * 1000, 2)
                if len(batch["results"]) != len(documents) or batch["results"] != singles:
                    raise AssertionError(f"{name}: full batch outputs differ from independent single forwards")
                proof = {
                    "name": name, "documents": len(documents),
                    "serial_ms": serial_ms, "batch_ms": batch_ms,
                    "full_output_sha256": digest(batch["results"]),
                    "identical_full_json": True,
                }
                report["cases"].append(proof)
                print(f"{name}: {len(documents)} full JSON results equal; "
                      f"serial {serial_ms:.1f}ms, batched {batch_ms:.1f}ms", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
