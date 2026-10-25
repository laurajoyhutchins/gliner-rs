#!/usr/bin/env python3
"""Convert private PDF pages using a pinned Docling CLI, then benchmark GLiNER.

No copyrighted source passages, generated Docling JSON, raw prediction text or
reviewed labels are written outside --private-dir. Source-safe hashes and
aggregate timing can be shared separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from prepare_corpus import make_chunks
from jsonschema import Draft202012Validator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_pages(spec: str) -> list[int]:
    pages = set()
    for term in spec.split(","):
        term = term.strip()
        if not re.fullmatch(r"\d+(?:-\d+)?", term):
            raise ValueError("pages must be comma-separated, 1-based numbers/ranges")
        first, last = map(int, term.split("-")) if "-" in term else (int(term), int(term))
        if first < 1 or last < first or last - first > 1000:
            raise ValueError("invalid page range")
        pages.update(range(first, last + 1))
    if not 1 <= len(pages) <= 200:
        raise ValueError("select between 1 and 200 source pages")
    return sorted(pages)


def extract_page(document: dict, page: int) -> tuple[list[str], int]:
    """Preserve Docling body reading order, including nested groups and tables."""
    if document.get("schema_name") != "DoclingDocument":
        raise ValueError("not a DoclingDocument JSON export")
    if str(page) not in document.get("pages", {}):
        raise ValueError(f"Docling did not preserve original page {page}")
    for name in ("texts", "tables", "groups"):
        if not isinstance(document.get(name), list):
            raise ValueError(f"Docling document lacks {name} collection")
    body = document.get("body")
    if not isinstance(body, dict) or not isinstance(body.get("children"), list):
        raise ValueError("Docling document has no ordered body references")

    visited: set[str] = set()
    blocks: list[str] = []
    table_count = 0

    def page_numbers(item: dict) -> set[int]:
        provenance = item.get("prov")
        if not isinstance(provenance, list) or not provenance:
            raise ValueError("Docling content item lacks page provenance")
        numbers = {entry.get("page_no") for entry in provenance if isinstance(entry, dict)}
        if numbers != {page}:
            raise ValueError(f"Docling content attribution differs from page {page}")
        return numbers

    def visit(node: dict) -> None:
        nonlocal table_count
        if not isinstance(node, dict) or not isinstance(node.get("$ref"), str):
            raise ValueError("invalid Docling document-body reference")
        ref = node["$ref"]
        if ref in visited:
            return
        visited.add(ref)
        parts = ref.split("/")
        if len(parts) != 3 or parts[0] != "#" or parts[1] not in ("texts", "tables", "groups"):
            raise ValueError(f"unsupported Docling content reference: {ref}")
        kind = parts[1]
        if not parts[2].isdigit():
            raise ValueError("invalid Docling content index")
        index = int(parts[2])
        entries = document[kind]
        if index >= len(entries):
            raise ValueError("dangling Docling content reference")
        item = entries[index]
        if not isinstance(item, dict):
            raise ValueError("invalid Docling item")
        if item.get("self_ref", ref) != ref:
            raise ValueError("Docling self_ref disagrees with body reference")

        if kind == "groups":
            children = item.get("children")
            if not isinstance(children, list):
                raise ValueError("Docling group has no ordered children")
            for child in children:
                visit(child)
        elif kind == "texts":
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("Docling text must be a string")
            page_numbers(item)
            if text.strip():
                blocks.append(text)
        else:
            page_numbers(item)
            data = item.get("data") or {}
            cells = data.get("table_cells") if isinstance(data, dict) else None
            if not isinstance(cells, list) or not cells:
                raise ValueError("Docling table has no indexed cell data")
            rows = defaultdict(list)
            for cell in cells:
                if not isinstance(cell, dict):
                    raise ValueError("invalid Docling table cell")
                row = cell.get("start_row_offset_idx")
                col = cell.get("start_col_offset_idx")
                text = cell.get("text")
                if not isinstance(row, int) or not isinstance(col, int) or not isinstance(text, str):
                    raise ValueError("Docling table cell missing row/column/text")
                rows[row].append((col, text))
            for _, columns in sorted(rows.items()):
                blocks.append(" | ".join(value for _, value in sorted(columns)))
            table_count += 1
            for child in item.get("children", []):
                visit(child)

    for child in body["children"]:
        visit(child)
    # Fail closed on unattached body text/table nodes, not Docling furniture.
    for kind in ("texts", "tables"):
        for i, item in enumerate(document[kind]):
            if item.get("content_layer", "body") == "body" and f"#/{kind}/{i}" not in visited:
                raise ValueError(f"unreferenced Docling body {kind} node {i}")
    if not blocks:
        raise ValueError(f"no Docling body content on page {page}")
    return blocks, table_count


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--docling-bin", type=Path, required=True)
    ap.add_argument("--docling-revision", required=True, help="actual Docling git SHA/tag used to build CLI")
    ap.add_argument("--pages", default="40,80,100,230,350,450,550,650")
    ap.add_argument("--mode", choices=("standard", "text-layer"), default="standard")
    ap.add_argument("--private-dir", type=Path, required=True)
    ap.add_argument("--max-chars", type=int, default=550)
    ap.add_argument("--rust-binary", type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--python-source", type=Path)
    ap.add_argument("--checkpoint-sha256")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args(argv)
    if args.max_chars < 100:
        ap.error("--max-chars must be >= 100")
    pages = parse_pages(args.pages)
    private = args.private_dir.resolve()
    private.mkdir(parents=True, exist_ok=True)
    contracts = Path(__file__).resolve().parent / "contracts"
    docling_schema = Draft202012Validator(json.loads((contracts / "docling-export.schema.json").read_text()))
    corpus_schema = Draft202012Validator(json.loads((contracts / "docling-corpus.schema.json").read_text()))
    output_pages = private / "docling-pages"
    output_pages.mkdir(exist_ok=True)
    source_sha = sha256(args.pdf)
    binary_sha = sha256(args.docling_bin)
    blocks_by_page = {}
    page_receipts = []
    total_tables = 0
    for page in pages:
        destination = output_pages / f"page-{page:04d}.json"
        command = [str(args.docling_bin.resolve()), str(args.pdf.resolve()),
                   "--pages", f"{page}-{page}", "--to", "json",
                   "--output-file", str(destination), "--no-progress"]
        if args.mode == "text-layer":
            command.append("--text-layer-only")
        # A missing structured-layout model is a hard failure, never fallback.
        subprocess.run(command, check=True, timeout=180, stdout=subprocess.DEVNULL)
        document = json.loads(destination.read_text())
        docling_schema.validate(document)
        blocks, table_count = extract_page(document, page)
        blocks_by_page[page] = blocks
        total_tables += table_count
        page_receipts.append({
            "page": page, "docling_json_sha256": sha256(destination),
            "text_elements": len(document["texts"]), "tables": table_count,
            "docling_schema_version": document.get("version")
        })
    chunks = make_chunks(blocks_by_page, set(pages), args.max_chars)
    if not chunks:
        raise ValueError("No extractable Docling chunks")
    corpus = {
        "format_version": 1, "source_sha256": source_sha,
        "extraction": "docling_document_json", "pages": pages, "chunks": chunks,
        "docling": {"source_revision": args.docling_revision, "binary_sha256": binary_sha,
                    "mode": args.mode, "page_receipts": page_receipts, "tables": total_tables}
    }
    corpus_schema.validate(corpus)
    corpus_path = private / "corpus.json"
    corpus_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n")
    safe = {
        "source_sha256": source_sha, "docling_binary_sha256": binary_sha,
        "docling_source_revision": args.docling_revision, "docling_mode": args.mode,
        "page_receipts": page_receipts, "pages": pages,
        "chunks": len(chunks), "tables": total_tables,
        "accuracy": "unassessed_without_independent_gold_labels"
    }
    if args.rust_binary or args.model_dir:
        if not all((args.rust_binary, args.model_dir, args.checkpoint_sha256)):
            ap.error("inference requires --rust-binary, --model-dir, --checkpoint-sha256")
        result = private / "inference-report.json"
        command = [sys.executable, str(Path(__file__).with_name("benchmark_regulatory.py")),
                   "--corpus", str(corpus_path), "--rust-binary", str(args.rust_binary),
                   "--model-dir", str(args.model_dir),
                   "--checkpoint-sha256", args.checkpoint_sha256,
                   "--threads", str(args.threads), "--rounds", str(args.rounds),
                   "--out", str(result)]
        if args.python_source:
            command += ["--python-source", str(args.python_source)]
        subprocess.run(command, check=True, timeout=2400)
        report = json.loads(result.read_text())
        safe["parity"] = {
            "cases": report["case_count"], "semantic_matches": report["semantic_equal_count"],
            "full_numeric_matches": report["full_numeric_equal_count"],
            "geomean_python_over_rust": report["geomean_python_over_rust"],
            "total_rust_median_ms": report["total_rust_median_ms"],
            "total_python_median_ms": report["total_python_median_ms"]
        }
    (private / "source-safe-receipt.json").write_text(json.dumps(safe, indent=2) + "\n")
    print(json.dumps(safe, indent=2))
    return safe


if __name__ == "__main__":
    run()
