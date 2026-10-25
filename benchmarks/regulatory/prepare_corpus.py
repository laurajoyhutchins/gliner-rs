#!/usr/bin/env python3
"""Build a PRIVATE, page-provenanced benchmark corpus from PDF or Docling JSON.

Never commit generated corpora from licensed sources. The public benchmark
reports only source hashes, page numbers, aggregate timings, and correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def docling_pages(document: dict) -> dict[int, list[str]]:
    """Parse DoclingDocument JSON text items with prov.page_no provenance."""
    pages: dict[int, list[str]] = defaultdict(list)
    for item in document.get("texts", []):
        text = item.get("text", "")
        provenance = item.get("prov", [])
        if not isinstance(text, str) or not text.strip() or not isinstance(provenance, list):
            continue
        seen: set[int] = set()
        for entry in provenance:
            page = entry.get("page_no") if isinstance(entry, dict) else None
            if isinstance(page, int) and page >= 1 and page not in seen:
                pages[page].append(text.strip())
                seen.add(page)
    return pages


def pdf_pages(source: Path, selected: set[int]) -> dict[int, list[str]]:
    """Explicit fallback for text-bearing PDFs; NOT Docling extraction."""
    import fitz

    pages: dict[int, list[str]] = defaultdict(list)
    with fitz.open(source) as pdf:
        for page in sorted(selected):
            if page < 1 or page > len(pdf):
                raise ValueError(f"page {page} outside 1..{len(pdf)}")
            for block in pdf[page - 1].get_text("blocks"):
                text = block[4].strip()
                if text and not re.fullmatch(r"\s*\d+(?:\s*[–-]\s*\d+)?\s*", text):
                    pages[page].append(text)
    return pages


def make_chunks(pages: dict[int, list[str]], selected: set[int], maximum: int) -> list[dict]:
    chunks: list[dict] = []
    for page in sorted(selected):
        blocks = [re.sub(r"\s+", " ", item).strip() for item in pages.get(page, [])]
        blocks = [item for item in blocks if item and not item.startswith("2017 Edition  NATIONAL ELECTRICAL CODE")]
        current = ""
        for block in blocks:
            while block:
                available = maximum - len(current) - bool(current)
                if available <= 0:
                    chunks.append({"page": page, "text": current})
                    current = ""
                    available = maximum
                if len(block) <= available:
                    current += (" " if current else "") + block
                    block = ""
                else:
                    split = block.rfind(" ", 1, available + 1)
                    if split < available // 2:
                        split = available
                    current += (" " if current else "") + block[:split].rstrip()
                    block = block[split:].lstrip()
                    chunks.append({"page": page, "text": current})
                    current = ""
        if current:
            chunks.append({"page": page, "text": current})
    for number, chunk in enumerate(chunks):
        chunk["id"] = f"p{chunk['page']:04d}-c{number:04d}"
        chunk["sha256"] = digest(chunk["text"].encode())
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=Path)
    src.add_argument("--docling-json", type=Path)
    parser.add_argument("--source-pdf", type=Path, help="required source PDF for Docling JSON provenance")
    parser.add_argument("--pages", required=True, help="1-based PDF physical page numbers, comma separated")
    parser.add_argument("--max-chars", type=int, default=550)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.max_chars < 100:
        parser.error("max-chars must be at least 100")
    selected = {int(n) for n in args.pages.split(",")}
    if not selected or min(selected) < 1:
        parser.error("page numbers must be >= 1")
    source = args.pdf or args.docling_json
    if args.docling_json and not args.source_pdf:
        parser.error("--docling-json requires --source-pdf; Docling JSON is not the source PDF")
    if args.pdf and args.source_pdf:
        parser.error("--source-pdf belongs only with --docling-json")
    if args.docling_json:
        pages = docling_pages(json.loads(source.read_text()))
        kind = "docling_document_json"
    else:
        pages = pdf_pages(source, selected)
        kind = "pymupdf_text_blocks_not_docling"
    chunks = make_chunks(pages, selected, args.max_chars)
    if not chunks:
        raise RuntimeError("No page text extracted")
    record = {"format_version": 1, "source_sha256": digest((args.source_pdf or source).read_bytes()),
              "extraction": kind, "pages": sorted(selected), "chunks": chunks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source_sha256": record["source_sha256"], "extraction": kind,
                      "pages": len(selected), "chunks": len(chunks),
                      "private_file": str(args.out)}))


if __name__ == "__main__":
    main()
