#!/usr/bin/env python3
"""Deterministic-input GLiNER2.5 Rust vs Python semantic parity + warm CPU timing.

Use the same pinned checkpoint, measured in separate phases. This benchmark
measures the Rust HTTP boundary and the Python in-process API; do not interpret
small differences as intrinsic framework speedup. Exit nonzero on semantic
parity failures and optionally on full confidence/metadata parity failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rust-bin', required=True, type=Path, help='gliner server binary, compiled from this revision')
    p.add_argument('--model-dir', required=True, type=Path, help='identical checkpoint directory for both engines')
    p.add_argument('--checkpoint-sha256', required=True, help='required SHA-256 of model.safetensors')
    p.add_argument('--python-source', type=Path, help='optional GLiNER2 source tree (otherwise import installed gliner2)')
    p.add_argument('--cases', type=Path, default=Path(__file__).with_name('cases.json'))
    p.add_argument('--out', type=Path, default=Path('gliner-parity-benchmark.json'))
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--warmups', type=int, default=2)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--port', type=int, default=18761)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--require-full-parity', action='store_true')
    a=p.parse_args()
    if a.rounds < 1 or a.warmups < 0 or a.threads < 1 or not (0 <= a.threshold <= 1):
        p.error('rounds and threads must be positive; warmups >= 0; threshold in [0,1]')
    for path in (a.rust_bin,a.model_dir/'model.safetensors',a.cases):
        if not path.exists(): p.error(f'missing {path}')
    with (a.model_dir/'model.safetensors').open('rb') as f:
        got=hashlib.file_digest(f,'sha256').hexdigest()
    if got.lower()!=a.checkpoint_sha256.lower():p.error(f'checkpoint SHA mismatch: {got}')
    return a,got


def quant(samples):
    s=sorted(samples)
    return {'median_ms':round(statistics.median(s),3),'p95_ms':round(s[math.ceil(.95*len(s))-1],3),'mean_ms':round(statistics.mean(s),3),'samples_ms':[round(v,3) for v in samples]}


def peak_rss(pid):
    try:
        return int(next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('VmHWM:')))*1024
    except (OSError,StopIteration):
        return None


def read_json(url, body=None):
    request=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode('utf-8'),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=60) as response: return json.load(response)


def rust_engine(a,cases):
    log=a.out.with_suffix('.rust.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as capture:
        proc=subprocess.Popen([str(a.rust_bin.resolve()),'--model',str(a.model_dir.resolve()),'--threads',str(a.threads),'serve','--host','127.0.0.1','--port',str(a.port)],stdout=capture,stderr=subprocess.STDOUT,env=dict(os.environ,RAYON_NUM_THREADS=str(a.threads),OMP_NUM_THREADS=str(a.threads)))
        try:
            for _ in range(300):
                if proc.poll() is not None: raise RuntimeError(f'Rust process exited {proc.returncode}: {log.read_text()[-3000:]}')
                try:
                    read_json(f'http://127.0.0.1:{a.port}/health');break
                except Exception: time.sleep(.1)
            else: raise RuntimeError('Rust model load timeout')
            results={}
            for case in cases:
                data={k:v for k,v in case.items() if k in ('text','entities','relations','json','classify')}
                data.update(threshold=a.threshold,spans=True,confidence=True)
                def once():return read_json(f'http://127.0.0.1:{a.port}/v1/extract',data)['result']
                for _ in range(a.warmups):once()
                samples=[]
                for _ in range(a.rounds):
                    start=time.perf_counter_ns();output=once();samples.append((time.perf_counter_ns()-start)/1e6)
                results[case['name']]={'output':output,'timing':quant(samples)}
            return results,peak_rss(proc.pid)
        finally:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()


def python_engine(a,cases):
    os.environ['OMP_NUM_THREADS']=str(a.threads)
    os.environ['MKL_NUM_THREADS']=str(a.threads)
    import torch
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(a.threads)
    if a.python_source:sys.path.insert(0,str(a.python_source.resolve()))
    from gliner2 import AutoExtractor
    start=time.perf_counter()
    model=AutoExtractor.from_pretrained(str(a.model_dir.resolve()),map_location='cpu')
    model.eval()
    load_s=time.perf_counter()-start
    results={}
    for case in cases:
        if 'json' in case:
            specs={name:fields.split(',') for item in case['json'] for name,fields in [item.split('=',1)]}
            schema=model._json_schema(specs)
        else:schema=model.create_schema()
        if 'entities' in case:schema.entities(case['entities'])
        if 'relations' in case:schema.relations(case['relations'])
        if 'classify' in case:
            for task in case['classify']:
                name,labels=task.split('=',1);schema.classification(name,labels.split(','))
        def once():
            with torch.inference_mode():return model.extract(case['text'],schema,threshold=a.threshold,include_confidence=True,include_spans=True)
        for _ in range(a.warmups):once()
        samples=[]
        for _ in range(a.rounds):
            start=time.perf_counter_ns();output=once();samples.append((time.perf_counter_ns()-start)/1e6)
        results[case['name']]={'output':output,'timing':quant(samples)}
    return results,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,load_s


def compare(a,b,skip_confidence=False,path='$'):
    if isinstance(a,dict) and isinstance(b,dict):
        keys=(set(a)|set(b))-({'confidence'} if skip_confidence else set())
        return sum((compare(a[k],b[k],skip_confidence,path+'.'+k) if k in a and k in b else [f'{path}.{k}: missing'] for k in sorted(keys)),[])
    if isinstance(a,list) and isinstance(b,list):
        problems=[] if len(a)==len(b) else [f'{path}: length {len(a)} != {len(b)}']
        for i,(x,y) in enumerate(zip(a,b)): problems+=compare(x,y,skip_confidence,f'{path}[{i}]')
        return problems
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return [] if abs(a-b)<=1e-4 else [f'{path}: {a!r} != {b!r}']
    return [] if a==b else [f'{path}: {a!r} != {b!r}']


def main():
    a,model_sha=parse_args();cases=json.loads(a.cases.read_text())
    rust,rrss=rust_engine(a,cases)
    python,prss,load=python_engine(a,cases)
    rows=[];semantic_fail=full_fail=0
    for case in cases:
        name=case['name'];r=rust[name]['output'];p=python[name]['output']
        semantic=compare(r,p,True);full=compare(r,p,False)
        semantic_fail+=bool(semantic);full_fail+=bool(full)
        rt=rust[name]['timing']['median_ms'];pt=python[name]['timing']['median_ms']
        rows.append({'name':name,'rust':rust[name],'python':python[name],'semantically_equal':not semantic,'full_parity':not full,'semantic_differences':semantic,'full_differences':full,'median_speedup_python_over_rust':round(pt/rt,3)})
        print(f'{name:17s} Rust {rt:7.2f}ms Python {pt:7.2f}ms | semantic={not semantic} full={not full}',flush=True)
    report={'model_sha256':model_sha,'model_dir':str(a.model_dir),'rust_bin':str(a.rust_bin),'python_source':str(a.python_source),'threads':a.threads,'rounds':a.rounds,'warmups':a.warmups,'threshold':a.threshold,'host':platform.platform(),'rust_peak_rss_bytes':rrss,'python_peak_rss_bytes':prss,'python_model_load_seconds':round(load,3),'semantic_failures':semantic_fail,'full_parity_failures':full_fail,'cases':rows}
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(f'semantic {len(rows)-semantic_fail}/{len(rows)}; full {len(rows)-full_fail}/{len(rows)}; report {a.out}')
    return 1 if semantic_fail or (a.require_full_parity and full_fail) else 0

if __name__=='__main__':raise SystemExit(main())
