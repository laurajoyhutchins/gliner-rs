#!/usr/bin/env python3
"""Compare warm end-to-end Rust inference on baseline and candidate binaries.

Both binaries use the same SHA-256-verified Safetensors checkpoint and corpus.
Run order alternates across pairs; output parity is mandatory, while latency
regression is reported rather than gated because shared CI hosts are noisy.
"""
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


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--cases', type=Path, default=Path(__file__).with_name('cases.json'))
    p.add_argument('--out', type=Path, default=Path('rust-revision-benchmark.json'))
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--pairs', type=int, default=2)
    p.add_argument('--port', type=int, default=18791)
    args = p.parse_args()
    if min(args.threads, args.samples, args.pairs) < 1 or args.warmups < 0:
        p.error('threads/samples/pairs must be positive; warmups nonnegative')
    for path in (args.baseline, args.candidate, args.cases, args.model_dir / 'model.safetensors'):
        if not path.is_file():
            p.error(f'missing file: {path}')
    with (args.model_dir / 'model.safetensors').open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != args.checkpoint_sha256.lower():
        p.error(f'checkpoint SHA-256 mismatch: {digest}')
    return args


def request(port, body=None):
    url = f'http://127.0.0.1:{port}/' + ('health' if body is None else 'v1/extract')
    req = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def peak_bytes(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def run_binary(exe, args, cases, port):
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads))
    cmd = [
        str(exe.resolve()), '--model', str(args.model_dir.resolve()),
        '--threads', str(args.threads), 'serve', '--host', '127.0.0.1', '--port', str(port)
    ]
    log_path = args.out.with_suffix(f'.{exe.name}.{port}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                if proc.poll() is not None:
                    raise RuntimeError(f'{exe.name} exited during startup; inspect {log_path}')
                try:
                    request(port)
                    break
                except (OSError, urllib.error.HTTPError):
                    time.sleep(0.1)
            else:
                raise RuntimeError(f'{exe.name} startup timeout; inspect {log_path}')
            results = {}
            for case in cases:
                body = {k: v for k, v in case.items() if k in ('text', 'entities', 'relations', 'json', 'classify')}
                body.update(threshold=0.5, spans=True, confidence=True)
                for _ in range(args.warmups):
                    request(port, body)
                samples = []
                for _ in range(args.samples):
                    start = time.perf_counter_ns()
                    result = request(port, body)['result']
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                results[case['name']] = {'output': result, 'samples_ms': samples}
            return {'results': results, 'peak_rss_bytes': peak_bytes(proc.pid)}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def compare(a, b, location='$'):
    if type(a) is not type(b):
        return [f'{location}: type mismatch']
    if isinstance(a, dict):
        return sum((compare(a[k], b[k], location + '.' + k)
                    if k in a and k in b else [f'{location}.{k}: missing']
                    for k in sorted(set(a) | set(b))), [])
    if isinstance(a, list):
        issues = [] if len(a) == len(b) else [f'{location}: length mismatch']
        for i, (x, y) in enumerate(zip(a, b)):
            issues += compare(x, y, f'{location}[{i}]')
        return issues
    if isinstance(a, float):
        return [] if math.isclose(a, b, rel_tol=0, abs_tol=1e-4) else [f'{location}: {a} != {b}']
    return [] if a == b else [f'{location}: {a!r} != {b!r}']


def main():
    args = arguments()
    cases = json.loads(args.cases.read_text())
    measurements = {'baseline': [], 'candidate': []}
    for pair in range(args.pairs):
        order = ('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')
        for index, name in enumerate(order):
            print(f'pair {pair + 1}/{args.pairs}: {name}', flush=True)
            measurements[name].append(run_binary(getattr(args, name), args, cases, args.port + pair * 2 + index))
    report = {
        'baseline': str(args.baseline), 'candidate': str(args.candidate),
        'model_sha256': args.checkpoint_sha256, 'threads': args.threads,
        'warmups': args.warmups, 'samples': args.samples, 'pairs': args.pairs,
        'cases': [], 'mismatched_cases': []
    }
    ratios = []
    for case in cases:
        name = case['name']
        reference = measurements['baseline'][0]['results'][name]['output']
        errors = []
        for kind in ('baseline', 'candidate'):
            for run in measurements[kind]:
                errors += compare(reference, run['results'][name]['output'])
        baseline = [sample for run in measurements['baseline']
                    for sample in run['results'][name]['samples_ms']]
        candidate = [sample for run in measurements['candidate']
                     for sample in run['results'][name]['samples_ms']]
        speedup = statistics.median(baseline) / statistics.median(candidate)
        ratios.append(speedup)
        report['cases'].append({
            'name': name, 'baseline_median_ms': statistics.median(baseline),
            'candidate_median_ms': statistics.median(candidate),
            'speedup': speedup, 'baseline_samples_ms': baseline,
            'candidate_samples_ms': candidate, 'differences': errors
        })
        if errors:
            report['mismatched_cases'].append(name)
        print(f'{name:17s} {statistics.median(baseline):7.2f}ms -> '
              f'{statistics.median(candidate):7.2f}ms ({speedup:.3f}x) '
              + ('MISMATCH' if errors else 'same output'), flush=True)
    report['geomean_speedup'] = math.exp(statistics.mean(math.log(x) for x in ratios))
    report['baseline_peak_rss_bytes'] = [r['peak_rss_bytes'] for r in measurements['baseline']]
    report['candidate_peak_rss_bytes'] = [r['peak_rss_bytes'] for r in measurements['candidate']]
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'geometric-mean speedup {report["geomean_speedup"]:.3f}x; '
          f'output mismatches {len(report["mismatched_cases"])}')
    return 1 if report['mismatched_cases'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
