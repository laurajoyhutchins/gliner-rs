#!/usr/bin/env python3
"""Compare the exact same binary-independent benchmark harness on two heads."""
import json
import sys
from pathlib import Path
baseline, candidate = [Path(a) for a in sys.argv[1:]]
read = lambda p: {x["case"]: x for x in (json.loads(line) for line in p.read_text().splitlines() if line.startswith("{"))}
a, b = read(baseline), read(candidate)
if not a or set(a) != set(b) or set(a) != {"mixed", "uniform"}:
    raise SystemExit(f"unexpected benchmark scenarios: {sorted(a)}, {sorted(b)}")
for name in sorted(a):
    old, new = a[name], b[name]
    if old["documents"] != new["documents"]:
        raise SystemExit(f"{name}: changed workload")
    if old["output_sha256"] != new["output_sha256"]:
        raise SystemExit(f"{name}: full JSON output changed between source revisions")
    ratio = old["default_median_ms"] / new["default_median_ms"]
    print(f"{name}: old-default {old['default_median_ms']:.1f} ms -> new-default "
          f"{new['default_median_ms']:.1f} ms ({ratio:.3f}x), "
          f"forced-bucket {new['bucketed_median_ms']:.1f} ms, "
          f"full-output SHA-256 identical", flush=True)
